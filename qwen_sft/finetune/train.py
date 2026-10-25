"""Generic SFT trainer for the Qwen3.5 4B base.

Designed for Kaggle (2x T4, 16 GB each, 20 GB ``/kaggle/working``):

    python -m qwen_sft.finetune.train \
        --dataset openthoughts --dataset astra --dataset fable \
        --limit-per-dataset 20000 --epochs 1

Data is normalized to chat messages, rendered with the *model's own* chat
template, and masked so only assistant turns contribute to the loss.

Every hyperparameter that matters on a T4 is a flag; the defaults are a
conservative first run, not a tuned recipe.
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import glob
import json
import sys

from qwen_sft.config import (
    DATA_DIR,
    DEFAULT_BASE_MODEL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATASETS,
    DEFAULT_EPOCHS,
    DEFAULT_GRAD_ACCUM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LIMIT_PER_DATASET,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_SEED,
    INSTRUCTION_PART,
    LORA_TARGET_MODULES,
    PROJECT_ROOT,
    RESPONSE_PART,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT on a chat dataset")
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL,
                        help="Base model repo id or local dir (must have safetensors)")
    parser.add_argument("--dataset", action="append", default=None,
                        help="Registry key, HF dataset id, .jsonl file, or a directory "
                             "of .jsonl (repeatable; default: openthoughts, astra, fable)")
    parser.add_argument("--limit-per-dataset", type=int, default=DEFAULT_LIMIT_PER_DATASET,
                        help="Max conversations kept per source when preparing")
    parser.add_argument("--prepare-dir", default=DATA_DIR, help="Where prepared JSONL lives")
    parser.add_argument("--refresh-data", action="store_true",
                        help="Re-download/re-prepare even if a cached JSONL exists")
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "lora_adapter"),
                        help="Where the final LoRA adapter is saved")
    parser.add_argument("--training-output-dir", default=os.path.join(PROJECT_ROOT, "training_output"),
                        help="Where checkpoints/trainer state are written")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--fresh", action="store_true",
                        help="Start a brand-new run: quarantine old local "
                             "checkpoint-* dirs as stale-checkpoint-* so no later "
                             "resume can inherit the previous run's state.")
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-4bit", action="store_true", help="Load in bf16/fp16 instead of 4-bit")
    parser.add_argument("--no-mask-prompt", action="store_true",
                        help="Train on prompt tokens too (not recommended)")
    parser.add_argument("--hub-repo", default=None, help="Repo id to push the adapter to")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push the adapter after training (needs HF_TOKEN and --hub-repo)")
    parser.add_argument("--push-checkpoints", action=argparse.BooleanOptionalAction, default=None,
                        help="Push the latest checkpoint to --hub-repo at every save. "
                             "Default: on whenever --hub-repo is set and a token is available "
                             "(only last-checkpoint is kept on the Hub).")
    parser.add_argument("--resume-from-hub", action=argparse.BooleanOptionalAction, default=None,
                        help="Resume from last-checkpoint in --hub-repo when it exists. "
                             "Default: on whenever checkpoint pushing is enabled.")
    parser.add_argument("--hub-private", action="store_true",
                        help="Create the Hub repo as private")
    parser.add_argument("--save-total-limit", type=int, default=2,
                        help="How many checkpoints to keep on disk")
    args = parser.parse_args(argv)

    # Automatic mode: --hub-repo is all it takes. Opt out per feature with
    # --no-push-checkpoints / --no-resume-from-hub.
    args.push_checkpoints = bool(args.hub_repo) if args.push_checkpoints is None \
        else args.push_checkpoints
    args.resume_from_hub = args.push_checkpoints if args.resume_from_hub is None \
        else args.resume_from_hub
    return args


def require_cuda() -> None:
    try:
        import torch
    except ImportError:
        print("[FATAL] PyTorch is not installed.")
        raise SystemExit(1)
    if not torch.cuda.is_available():
        print("[FATAL] CUDA is not available. Training needs a GPU (Kaggle: T4/P100).")
        raise SystemExit(1)
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[GPU] {name} ({vram:.1f} GB VRAM)")
    if vram < 12:
        print("[WARN] Under 12 GB VRAM: lower --max-seq-length (1024) or --batch-size stays 1.")


def slugify(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value).strip("-").lower()


def resolve_dataset_files(values, limit, seed, refresh, prepare_dir) -> list[str]:
    """Turn CLI dataset values into local JSONL paths, preparing HF sources."""
    from qwen_sft.datasets.sources import Source, get_source, prepare_source
    from qwen_sft.validation import parse_messages

    files: list[str] = []
    os.makedirs(prepare_dir, exist_ok=True)

    for value in values:
        if os.path.isdir(value):
            found = sorted(glob.glob(os.path.join(value, "*.jsonl")))
            print(f"[DATA] {len(found)} shard(s) in {value}")
            files.extend(found)
            continue
        if os.path.isfile(value):
            files.append(value)
            continue

        try:
            source = get_source(value)
        except KeyError:
            source = Source(
                key=slugify(value),
                repo_id=value,
                normalizer=parse_messages,
                description="ad-hoc Hugging Face dataset",
            )

        out_path = os.path.join(prepare_dir, f"{source.key}.jsonl")
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 0 and not refresh:
            print(f"[DATA] Reusing {out_path} (use --refresh-data to rebuild)")
        else:
            print(f"[DATA] Preparing {source.key} <- {source.repo_id}")
            stats = prepare_source(source, out_path, limit=limit, seed=seed)
            print(f"[DATA] {stats.summary()}")
            if stats.written == 0:
                raise SystemExit(
                    f"[FATAL] {source.key} produced no valid conversations. "
                    "Check the dataset schema with --list."
                )
        files.append(out_path)

    if not files:
        raise SystemExit("[FATAL] No datasets resolved.")
    return files


def format_dataset(dataset, tokenizer):
    """Render each conversation with the model's own chat template."""

    def format_example(example):
        messages = example["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    return dataset.map(format_example, remove_columns=dataset.column_names)


def report_length_stats(dataset, tokenizer, max_seq_length, sample: int = 200) -> None:
    """Warn about truncation before spending GPU time on it.

    Reasoning datasets are long. Anything past ``max_seq_length`` is cut, and a
    trace cut mid-thought teaches the model to stop mid-thought.
    """
    total = len(dataset)
    if total == 0:
        return
    step = max(total // sample, 1)
    texts = [dataset[index]["text"] for index in range(0, total, step)][:sample]
    lengths = sorted(len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts)
    over = sum(1 for length in lengths if length > max_seq_length)
    print(
        f"[LENGTH] sampled {len(lengths)}: median {lengths[len(lengths) // 2]} tokens, "
        f"max {lengths[-1]}, over limit {over}/{len(lengths)} ({over / len(lengths):.0%})"
    )
    if over:
        print(
            f"[WARN] {over / len(lengths):.0%} of sampled samples exceed --max-seq-length "
            f"{max_seq_length} and will be truncated. Raise it (4096 fits a 16 GB T4 with "
            "gradient checkpointing) or lower --limit-per-dataset for cleaner samples."
        )


def load_base(model_name, max_seq_length, load_in_4bit):
    """Load via Unsloth, falling back to the vision loader for VL checkpoints."""
    from unsloth import FastLanguageModel

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        return model, tokenizer, "language"
    except Exception as exc:
        print(f"[LOAD] FastLanguageModel failed ({type(exc).__name__}: {exc})")
        print("[LOAD] Retrying with FastVisionModel (text-only LoRA)...")
        from unsloth import FastVisionModel

        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        # FastVisionModel returns a processor; text-only SFT needs the tokenizer.
        return model, getattr(tokenizer, "tokenizer", tokenizer), "vision"


def attach_lora(model, args, loader_kind):
    from unsloth import FastLanguageModel

    common = dict(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    if loader_kind == "language":
        model = FastLanguageModel.get_peft_model(model, use_rslora=False, **common)
    else:
        from unsloth import FastVisionModel

        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            **common,
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] trainable {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    return model


def validate_hub_args(args) -> None:
    """Fail before loading the model if the Hub options cannot work."""
    if not (args.push_checkpoints or args.resume_from_hub):
        return
    if not args.hub_repo:
        raise SystemExit("[FATAL] --push-checkpoints/--resume-from-hub need --hub-repo.")
    if not args.push_checkpoints:
        return

    from huggingface_hub import get_token

    if not get_token():
        raise SystemExit(
            "[FATAL] --push-checkpoints needs a Hugging Face token. Set HF_TOKEN "
            "(Kaggle secret) or run `hf auth login`."
        )


def resolve_resume_checkpoint(args) -> str | None:
    """Resume path: explicit flag > prompt on local checkpoint > prompt on Hub.

    Falls back to 'continue' when stdin is not interactive (batch reruns), so a
    crashed Kaggle session still picks up where it left off. Every candidate is
    validated first; a torn newest checkpoint resumes from the previous one.
    """
    if args.resume_from_checkpoint:
        problem = _checkpoint_problem(args.resume_from_checkpoint)
        if problem:
            raise SystemExit(
                f"[FATAL] --resume-from-checkpoint {args.resume_from_checkpoint} "
                f"is unusable: {problem}. Remove the flag or point at a complete "
                "checkpoint-* dir."
            )
        return args.resume_from_checkpoint
    if getattr(args, "fresh", False):
        moved = _quarantine_local_checkpoints(args.training_output_dir)
        print(f"[HF] --fresh: quarantined {len(moved)} old checkpoint dir(s) as "
              "stale-checkpoint-*; training starts from step 0.")
        return None
    if not args.resume_from_hub:
        return None

    local, rejected = _best_local_checkpoint(args.training_output_dir)
    for path, reason in rejected:
        print(f"[HF] skipping {os.path.basename(path)}: {reason}")
    if local:
        if _ask_continue(local, "Local training_output"):
            return local
        print("[HF] Starting a fresh run.")
        moved = _quarantine_local_checkpoints(args.training_output_dir)
        if moved:
            print(f"[HF] moved {len(moved)} old checkpoint dir(s) to stale-checkpoint-* "
                  "so the fresh run and later resumes never inherit the old state.")
        else:
            print("[HF] the first save replaces last-checkpoint.")
        return None
    # No local candidate: fetch last-checkpoint from the Hub if it exists.

    import shutil

    import requests
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import (EntryNotFoundError, HfHubHTTPError,
                                       RepositoryNotFoundError)

    hub_dir = os.path.join(args.training_output_dir, "hub_resume")
    # Always download clean: snapshot_download keeps stale files otherwise.
    if os.path.isdir(hub_dir):
        shutil.rmtree(hub_dir)
    try:
        snapshot_download(
            repo_id=args.hub_repo,
            local_dir=hub_dir,
            allow_patterns=["last-checkpoint/*"],
        )
    except (RepositoryNotFoundError, EntryNotFoundError) as exc:
        print(f"[HF] No Hub checkpoint ({type(exc).__name__}); starting fresh.")
        return None
    except (OSError, HfHubHTTPError, requests.RequestException) as exc:
        # Network/auth glitch: never worth crashing the run over before the
        # model is even loaded. Anything usable locally was already considered.
        print(f"[HF] Hub fetch failed ({type(exc).__name__}: {exc}); "
              "continuing from local state.")
        return None

    path = os.path.join(hub_dir, "last-checkpoint")
    if not os.path.isdir(path):
        print("[HF] No last-checkpoint on the Hub; starting fresh.")
        return None
    problem = _checkpoint_problem(path)
    if problem:
        print(f"[HF] Hub last-checkpoint is unusable ({problem}); starting fresh.")
        return None
    if _ask_continue(path, "Hub last-checkpoint"):
        return path
    print("[HF] Starting a fresh run; the first save replaces last-checkpoint.")
    return None


def _checkpoint_step(path) -> int | None:
    try:
        with open(os.path.join(path, "trainer_state.json"), encoding="utf-8") as fh:
            step = json.load(fh).get("global_step")
        return int(step) if step else None
    except (OSError, TypeError, ValueError):
        return None


def _ask_continue(path, source) -> bool:
    step = _checkpoint_step(path)
    shown = f" (step {step:,})" if step else ""
    print(f"[HF] {source} checkpoint found{shown}: {path}")
    try:
        answer = input("    Continue this run? [Enter = yes, n = start fresh] ")
    except (EOFError, OSError):
        print("    No interactive input available; continuing from the checkpoint.")
        return True
    return answer.strip().lower() not in ("n", "no")


_ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def _checkpoint_problem(path) -> str | None:
    """Describe why a checkpoint dir is unsafe to resume from, or None if it is sound.

    Kaggle kills sessions (12-h limit, OOM, crash) exactly mid-save, which leaves
    a checkpoint-* dir with a truncated ``optimizer.pt``. Resuming from a torn
    file fails inside ``torch.load`` and burns the session, so resumes are
    validated first: the trainer state must parse and the adapter weights and
    optimizer file must exist non-empty and load.
    """
    if _checkpoint_step(path) is None:
        return "no parsable trainer_state.json"

    optimizer_path = os.path.join(path, "optimizer.pt")
    if not (os.path.isfile(optimizer_path) and os.path.getsize(optimizer_path) > 0):
        return "optimizer.pt is missing or empty"
    if not any(
        os.path.isfile(os.path.join(path, name)) and os.path.getsize(os.path.join(path, name)) > 0
        for name in _ADAPTER_WEIGHT_FILES
    ):
        return "adapter weights are missing or empty"

    try:
        import torch
    except ImportError:
        return None  # e.g. tensor-free validation; presence checks still held

    try:
        torch.load(optimizer_path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as exc:  # truncated zip/pickle from a killed session
        return f"optimizer.pt is unreadable ({type(exc).__name__}: {exc})"
    return None


def _best_local_checkpoint(training_output_dir):
    """Newest checkpoint-* that validates, plus what was rejected and why.

    Sorted newest first by the step in the directory name. A torn newest
    checkpoint falls back to the previous complete one instead of killing
    the rerun; ``(None, rejected)`` when nothing is usable.
    """
    def step_of(path):
        try:
            return int(os.path.basename(path).rsplit("-", 1)[-1])
        except ValueError:
            return -1

    dirs = sorted(glob.glob(os.path.join(training_output_dir, "checkpoint-*")),
                  key=step_of, reverse=True)
    rejected = []
    for path in dirs:
        problem = _checkpoint_problem(path)
        if problem is None:
            return path, rejected
        rejected.append((path, problem))
    return None, rejected


def _quarantine_local_checkpoints(training_output_dir) -> list[str]:
    """Move checkpoint-* dirs aside after a declined resume.

    A fresh run must not share its output dir with the previous run's
    checkpoints: the first save lands at a low step number next to the stale
    high one, and a later restart can then resume from the OLD run's
    optimizer/scheduler/dataset state (wrong schedule, wrong data position).
    Renaming to ``stale-checkpoint-N`` keeps the weights inspectable while
    keeping them out of the resume glob.
    """
    moved = []
    for path in sorted(glob.glob(os.path.join(training_output_dir, "checkpoint-*"))):
        target = os.path.join(training_output_dir, "stale-" + os.path.basename(path))
        try:
            os.rename(path, target)
            moved.append(target)
        except OSError as exc:
            print(f"[HF] could not quarantine {path}: {exc}")
    return moved


def build_sft_config(args, training_output_dir):
    import torch
    from trl import SFTConfig

    push_checkpoints = bool(args.push_checkpoints and args.hub_repo)
    kwargs = dict(
        output_dir=training_output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=args.epochs,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to="none",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        push_to_hub=push_checkpoints,
        hub_model_id=args.hub_repo if push_checkpoints else None,
        # "checkpoint" pushes at every save and keeps only last-checkpoint on the Hub.
        hub_strategy="checkpoint" if push_checkpoints else "end",
        hub_private_repo=bool(args.hub_private) if push_checkpoints else None,
        dataset_text_field="text",
        packing=False,
        dataset_num_proc=1,
    )
    try:
        return SFTConfig(max_length=args.max_seq_length, **kwargs)
    except TypeError:  # trl < 0.17
        return SFTConfig(max_seq_length=args.max_seq_length, **kwargs)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_cuda()
    validate_hub_args(args)
    resume_path = resolve_resume_checkpoint(args)

    from datasets import concatenate_datasets, load_dataset
    from trl import SFTTrainer

    dataset_values = args.dataset or list(DEFAULT_DATASETS)
    files = resolve_dataset_files(
        dataset_values,
        args.limit_per_dataset,
        args.seed,
        args.refresh_data,
        args.prepare_dir,
    )

    print(f"[LOAD] {args.model} (4-bit={not args.no_4bit}, seq={args.max_seq_length})")
    model, tokenizer, loader_kind = load_base(
        args.model, args.max_seq_length, load_in_4bit=not args.no_4bit
    )
    model = attach_lora(model, args, loader_kind)

    shards = [load_dataset("json", data_files=path, split="train") for path in files]
    dataset = concatenate_datasets(shards).shuffle(seed=args.seed)
    print(f"[DATA] {len(dataset):,} conversations from {len(files)} file(s)")
    dataset = format_dataset(dataset, tokenizer)
    report_length_stats(dataset, tokenizer, args.max_seq_length)
    print(f"[DATA] sample:\n{dataset[0]['text'][:400]}\n---")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=build_sft_config(args, args.training_output_dir),
    )

    if not args.no_mask_prompt:
        from unsloth.chat_templates import train_on_responses_only

        trainer = train_on_responses_only(
            trainer,
            instruction_part=INSTRUCTION_PART,
            response_part=RESPONSE_PART,
        )
        print("[MASK] Loss is computed on assistant turns only.")

    trainer.train(resume_from_checkpoint=resume_path)

    output_dir = os.path.abspath(args.output_dir)
    print(f"[SAVE] adapter -> {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if args.push_to_hub or args.push_checkpoints:
        from huggingface_hub import get_token

        token = get_token()
        if not token:
            print("[HF] No Hugging Face token; skipping adapter push.")
        elif not args.hub_repo:
            print("[HF] --push-to-hub needs --hub-repo; skipping push.")
        else:
            model.push_to_hub(args.hub_repo, token=token)
            tokenizer.push_to_hub(args.hub_repo, token=token)
            print(f"[HF] adapter pushed to https://huggingface.co/{args.hub_repo}")

    print("[DONE] Adapter ready. Merge it into the base with "
          "`python -m qwen_sft.finetune.merge --adapter <dir>`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
