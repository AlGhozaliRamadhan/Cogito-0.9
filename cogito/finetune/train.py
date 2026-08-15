import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse
import glob
import json
import queue
import sys
import threading
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IS_MAIN_PROCESS = int(os.environ.get("RANK", "0")) == 0
HUB_REPO_ID = "ozaa77/Cogito-0.9.1"
CPT_REVISION = "main"
HUB_KEEP_CHECKPOINTS = 2

_checkpoint_upload_queue: "queue.Queue[str | None]" = queue.Queue()
_checkpoint_upload_stop = threading.Event()
_checkpoint_upload_thread: "threading.Thread | None" = None


def parse_messages(example):
    messages = example.get("messages", [])
    if isinstance(messages, str):
        messages = json.loads(messages)
    elif isinstance(messages, list) and messages and isinstance(messages[0], str):
        messages = [json.loads(message) if isinstance(message, str) else message for message in messages]

    if not isinstance(messages, list) or not all(isinstance(message, dict) for message in messages):
        raise ValueError("messages must be a list of message objects")
    return messages


def audit_dataset(dataset, dataset_name):
    from cogito.validation import canonicalize_system_prompt, validate_conversation_structure

    invalid_examples = []
    sources = Counter()

    for index, example in enumerate(dataset):
        try:
            messages = canonicalize_system_prompt(parse_messages(example))
            is_valid, reason = validate_conversation_structure(messages)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            is_valid, reason = False, str(exc)

        if not is_valid:
            if len(invalid_examples) < 5:
                invalid_examples.append(f"record {index}: {reason}")
            continue

        source = str(example.get("oversample") or example.get("source") or "unknown")
        sources[source] += 1

    total = len(dataset)
    valid_total = sum(sources.values())
    if invalid_examples:
        examples = "; ".join(invalid_examples)
        raise ValueError(
            f"Refusing to train on {dataset_name}: {total - valid_total}/{total} records fail "
            f"the canonical Cogito validation ({examples})."
        )

    identity_count = sum(count for source, count in sources.items() if "cogito_identity_core" in source)
    probing_count = sum(count for source, count in sources.items() if "cogito_philosophical_probing" in source)
    identity_ratio = identity_count / valid_total if valid_total else 0
    probing_ratio = probing_count / valid_total if valid_total else 0
    print(
        f"[DATA AUDIT] {dataset_name}: {valid_total} valid records; "
        f"identity={identity_count} ({identity_ratio:.1%}), "
        f"probing={probing_count} ({probing_ratio:.1%})"
    )


def _prune_old_hub_checkpoints(hf_token: str):
    from huggingface_hub import HfApi, RepoFolder

    try:
        api = HfApi()
        entries = api.list_repo_tree(
            repo_id=HUB_REPO_ID,
            revision=CPT_REVISION,
            repo_type="model",
            token=hf_token,
            recursive=False,
        )
        steps = []
        for entry in entries:
            name = getattr(entry, "path", "")
            if isinstance(entry, RepoFolder) and name.startswith("checkpoint-"):
                try:
                    steps.append(int(name.split("-")[-1]))
                except ValueError:
                    continue
        if len(steps) <= HUB_KEEP_CHECKPOINTS:
            return
        stale = sorted(steps)[:-HUB_KEEP_CHECKPOINTS]
        for step in stale:
            print(f"[HF] Pruning old checkpoint-{step} from Hub...")
            try:
                api.delete_folder(
                    repo_id=HUB_REPO_ID,
                    repo_type="model",
                    path_in_repo=f"checkpoint-{step}",
                    revision=CPT_REVISION,
                    token=hf_token,
                )
            except TypeError:
                api.delete_folder(
                    repo_id=HUB_REPO_ID,
                    repo_type="model",
                    folder_path=f"checkpoint-{step}",
                    revision=CPT_REVISION,
                    token=hf_token,
                )
    except Exception as exc:
        print(f"[HF] Prune checkpoints failed (non-fatal): {exc}")


def _checkpoint_upload_worker(hf_token: str):
    from huggingface_hub import upload_folder

    while True:
        ckpt_dir = _checkpoint_upload_queue.get()
        try:
            if ckpt_dir is None or _checkpoint_upload_stop.is_set():
                return
            step = int(os.path.basename(ckpt_dir).split("-")[-1])
            print(f"\n[HF] Pushing checkpoint {step} to '{CPT_REVISION}'...")
            upload_folder(
                repo_id=HUB_REPO_ID,
                folder_path=ckpt_dir,
                path_in_repo=f"checkpoint-{step}",
                revision=CPT_REVISION,
                commit_message=f"checkpoint-{step}",
                token=hf_token,
            )
            print(f"[HF] Checkpoint {step} pushed to '{CPT_REVISION}'.")
            _prune_old_hub_checkpoints(hf_token)
        except Exception as exc:
            print(f"[HF ERROR] Failed to push checkpoint to Hub: {exc}")
        finally:
            _checkpoint_upload_queue.task_done()


def _flush_checkpoint_uploads():
    if _checkpoint_upload_thread is not None:
        _checkpoint_upload_queue.join()


def find_latest_local_checkpoint(root: str) -> str | None:
    if not os.path.isdir(root):
        return None
    dirs = [
        d for d in os.listdir(root)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(root, d))
    ]
    if not dirs:
        return None
    steps = sorted(int(d.split("-")[-1]) for d in dirs)
    return os.path.join(root, f"checkpoint-{steps[-1]}")


def sanitize_resume_checkpoint(ckpt_dir: str) -> str:
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        return ckpt_dir
    removed = []
    for fname in ("optimizer.pt", "scheduler.pt", "scaler.pt", "rng_state_0.pth", "rng_state_1.pth"):
        path = os.path.join(ckpt_dir, fname)
        if os.path.isfile(path):
            try:
                os.remove(path)
                removed.append(fname)
            except OSError as exc:
                print(f"[RESUME] Warning: could not remove {path}: {exc}")
    if removed:
        print(
            "[RESUME] Dropped optimizer state to guarantee a clean optimizer restart "
            f"on this run ({', '.join(removed)}). LoRA weights and step counter kept."
        )
    return ckpt_dir


def collect_checkpoint_step(training_output_dir: str, revision: str = "main") -> str | None:
    from huggingface_hub import snapshot_download

    cache_dir = os.path.join(training_output_dir, "hub_snapshot")
    print(f"\n[HF] Pulling latest checkpoint from {HUB_REPO_ID}@{revision} ...")
    try:
        snapshot_download(
            repo_id=HUB_REPO_ID,
            revision=revision,
            repo_type="model",
            allow_patterns=["checkpoint-*/*", "checkpoint-*"],
            cache_dir=cache_dir,
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception as exc:
        print(f"[HF] Could not download snapshot from Hub: {exc}")
        return None

    latest = None
    for root, dirs, files in os.walk(cache_dir):
        for d in dirs:
            if d.startswith("checkpoint-"):
                cand = os.path.join(root, d)
                if latest is None:
                    latest = cand
                else:
                    step_a = int(os.path.basename(cand).split("-")[-1])
                    step_b = int(os.path.basename(latest).split("-")[-1])
                    if step_a > step_b:
                        latest = cand
    if latest:
        print(f"[HF] Resuming from Hub checkpoint: {latest}")
        return latest

    legacy_root = os.path.join(training_output_dir, "hub_snapshot", "legacy_flat")
    os.makedirs(legacy_root, exist_ok=True)
    state_path = os.path.join(legacy_root, "trainer_state.json")
    if not os.path.isfile(state_path):
        print("[HF] No checkpoint-NNN folder on the Hub; checking for legacy flat checkpoint...")
        try:
            snapshot_download(
                repo_id=HUB_REPO_ID,
                revision=revision,
                repo_type="model",
                allow_patterns=[
                    "adapter_config.json", "adapter_model.safetensors",
                    "optimizer.pt", "scheduler.pt", "scaler.pt",
                    "rng_state_*.pth", "trainer_state.json",
                ],
                local_dir=legacy_root,
                token=os.environ.get("HF_TOKEN"),
            )
        except Exception as exc:
            print(f"[HF] Legacy flat checkpoint download failed: {exc}")
    if os.path.isfile(state_path):
        try:
            with open(state_path, encoding="utf-8") as fh:
                step = json.load(fh).get("global_step", 0)
        except Exception:
            step = 0
        ckpt_dir = os.path.join(training_output_dir, f"checkpoint-{step}")
        if ckpt_dir != legacy_root:
            os.makedirs(ckpt_dir, exist_ok=True)
            import shutil
            for fname in os.listdir(legacy_root):
                src = os.path.join(legacy_root, fname)
                dst = os.path.join(ckpt_dir, fname)
                if os.path.isfile(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
            print(f"[HF] Rebuilt legacy flat checkpoint as {ckpt_dir}")
        return ckpt_dir
    print("[HF] No checkpoint-NNN found on the Hub; starting fresh.")
    return None


def get_chat_eos_token_id(tokenizer):
    chat_eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if chat_eos_token_id is not None and chat_eos_token_id != tokenizer.unk_token_id:
        return chat_eos_token_id
    return tokenizer.eos_token_id


def main():
    try:
        import torch
        if not torch.cuda.is_available():
            print("[FATAL] CUDA not available. This script requires a GPU.")
            print("        Ensure you have PyTorch with CUDA installed.")
            sys.exit(1)
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[GPU] {gpu_name}  —  {gpu_vram:.1f} GB VRAM")
    except ImportError:
        print("[FATAL] PyTorch not installed. Run the environment setup first.")
        sys.exit(1)

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainerCallback
    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "combined_dense_dataset.jsonl"),
        help="Persona-aligned dense JSONL dataset to train on",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--model", type=str, default=None, help="Base model to load (defaults to Abliterated model or Qwen 14B)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for the final LoRA adapter")
    parser.add_argument("--training-output-dir", type=str, default=None, help="Directory for checkpoints and trainer state")
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Optional path to a checkpoint-N dir to resume from (defaults to the latest found in --training-output-dir)",
    )
    parser.add_argument("--no-push-to-hub", action="store_true", help="Keep an isolated comparison run local")
    parser.add_argument(
        "--allow-unbalanced-dataset",
        action="store_true",
        help="Allow a validated dataset with too little identity/probing data (not recommended)",
    )
    args = parser.parse_args()

    push_checkpoints = bool(os.environ.get("HF_TOKEN")) and not args.no_push_to_hub
    output_dir = os.path.abspath(args.output_dir or os.path.join(PROJECT_ROOT, "cogito_0.9_lora"))
    training_output_dir = os.path.abspath(
        args.training_output_dir or os.path.join(PROJECT_ROOT, "cogito_training_output")
    )

    abliterated_model = os.path.join(PROJECT_ROOT, "Qwen3-14B-Cogito-Abliterated")
    if args.model:
        base_model = args.model
    else:
        base_model = abliterated_model if os.path.exists(abliterated_model) else "Qwen/Qwen3-14B"

    max_seq_length = 1024
    print(f"\n{'='*60}")
    print(f"Loading {base_model} in 4-bit quantization...")
    print(f"Max sequence length: {max_seq_length}")
    print(f"{'='*60}\n")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
    )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[LoRA] Trainable: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)\n")

    dataset_arg = args.dataset
    if os.path.isdir(dataset_arg):
        datasets_to_train = sorted(glob.glob(os.path.join(dataset_arg, "*.jsonl")))
        print(f"[DATA] Found {len(datasets_to_train)} shards in directory {dataset_arg}")
    else:
        datasets_to_train = [dataset_arg]

    def format_example(example):
        from cogito.validation import canonicalize_system_prompt
        messages = [
            m for m in canonicalize_system_prompt(parse_messages(example))
            if isinstance(m, dict) and isinstance(m.get("content"), str) and m["content"].strip()
        ]
        if not messages:
            raise ValueError("record has no usable content in any message")
        return {
            "text": tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    class SavePeftModelCallback(TrainerCallback):
        def on_save(self, callback_args, state, control, **kwargs):
            global _checkpoint_upload_thread
            if not IS_MAIN_PROCESS or not push_checkpoints:
                return control
            ckpt_dir = (
                kwargs.get("output_dir")
                or os.path.join(callback_args.output_dir, f"checkpoint-{state.global_step}")
            )
            if not ckpt_dir or not os.path.isdir(ckpt_dir):
                return control
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                return control
            if _checkpoint_upload_thread is None:
                _checkpoint_upload_thread = threading.Thread(
                    target=_checkpoint_upload_worker,
                    args=(hf_token,),
                    daemon=True,
                    name="hub-checkpoint-uploader",
                )
                _checkpoint_upload_thread.start()
            _checkpoint_upload_queue.put(ckpt_dir)
            return control

    class SaveAtEpochEndCallback(TrainerCallback):
        def on_epoch_end(self, callback_args, state, control, **kwargs):
            if not IS_MAIN_PROCESS:
                return control
            control.should_save = True
            return control

    chat_eos_token_id = get_chat_eos_token_id(tokenizer)
    print(f"[TOKENIZER] ChatML end-of-turn token id: {chat_eos_token_id}")

    training_args = SFTConfig(
        output_dir=training_output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        num_train_epochs=args.epochs,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=5,
        logging_first_step=True,
        report_to="none",
        save_strategy="steps",
        save_steps=50,
        save_total_limit=10,
        load_best_model_at_end=False,
        seed=42,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        push_to_hub=False,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    resume_dir = args.resume_from_checkpoint
    if not resume_dir:
        resume_dir = find_latest_local_checkpoint(training_output_dir)
        if resume_dir:
            print(f"\n[RESUME] Local checkpoint found: {resume_dir}")
    if not resume_dir:
        resume_dir = collect_checkpoint_step(training_output_dir, revision="main")
    if resume_dir:
        resume_dir = sanitize_resume_checkpoint(resume_dir)

    try:
        checkpoint_path = resume_dir
        for i, current_dataset_path in enumerate(datasets_to_train):
            print(f"\n{'='*60}")
            print(f"PHASE/SHARD {i+1} OF {len(datasets_to_train)}: {current_dataset_path}")
            print(f"{'='*60}\n")

            if os.path.isfile(current_dataset_path):
                dataset = load_dataset("json", data_files=current_dataset_path, split="train")
            else:
                try:
                    dataset = load_dataset(current_dataset_path, split="train")
                except Exception as e:
                    print(f"\n[WARNING] Standard loading failed ({e}). Attempting robust pandas loading...")
                    import pandas as pd
                    from datasets import Dataset
                    from huggingface_hub import list_repo_files

                    files = [f for f in list_repo_files(current_dataset_path, repo_type="dataset") if f.endswith(".parquet")]
                    dfs = []
                    for f in files:
                        file_url = f"hf://datasets/{current_dataset_path}/{f}"
                        dfs.append(pd.read_parquet(file_url))

                    df = pd.concat(dfs, ignore_index=True)
                    dataset = Dataset.from_pandas(df)

            audit_dataset(dataset, current_dataset_path)
            dataset = dataset.map(format_example, remove_columns=dataset.column_names)

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                args=training_args,
                callbacks=[SaveAtEpochEndCallback(), SavePeftModelCallback()],
            )

            trainer = train_on_responses_only(
                trainer,
                instruction_part="<|im_start|>user\n",
                response_part="<|im_start|>assistant\n",
            )

            train_result = trainer.train(resume_from_checkpoint=checkpoint_path)
            metrics = train_result.metrics
            print(f"\n  -> Completed shard {i+1}. Train loss: {metrics.get('train_loss', 'N/A'):.4f}\n")

            checkpoint_path = None
    except Exception as e:
        print(f"\n[FATAL] Training crashed: {e}")
        raise

    _flush_checkpoint_uploads()
    if IS_MAIN_PROCESS:
        print(f"[SAVE] Saving LoRA adapter to: {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"\n{'='*60}")
        print("ALL DONE.")
        print(f"  LoRA adapter saved to: {output_dir}")
        print(f"  To run inference: python -m cogito")
        print(f"{'='*60}")
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token and not args.no_push_to_hub:
            print(f"\n[HF] HF_TOKEN detected. Pushing completed LoRA adapter to {HUB_REPO_ID} (main branch)...")
            try:
                model.push_to_hub(HUB_REPO_ID, token=hf_token, revision="main")
                tokenizer.push_to_hub(HUB_REPO_ID, token=hf_token, revision="main")
                print(f"[HF] Final push to main successful! Model available at: https://huggingface.co/{HUB_REPO_ID}")
            except Exception as e:
                print(f"[HF ERROR] Failed to push to Hugging Face: {e}")


if __name__ == "__main__":
    main()
