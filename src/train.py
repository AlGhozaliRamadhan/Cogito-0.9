import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IS_MAIN_PROCESS = int(os.environ.get("RANK", "0")) == 0
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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default=os.path.join(PROJECT_ROOT, "combined_dense_dataset.jsonl"),
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

# Checkpoints are persisted to the Hub's main branch during training unless the
# user explicitly asked for a fully-local, isolated run.
PUSH_CHECKPOINTS = bool(os.environ.get("HF_TOKEN")) and not args.no_push_to_hub

GENERATORS_DIR = os.path.join(PROJECT_ROOT, "scripts", "generators")
if GENERATORS_DIR not in sys.path:
    sys.path.insert(0, GENERATORS_DIR)

from validator import COGITO_SYSTEM_PROMPT, validate_conversation_structure

ABLITERATED_MODEL = os.path.join(PROJECT_ROOT, "Qwen2.5-Coder-14B-Cogito-Abliterated")

if args.model:
    BASE_MODEL = args.model
else:
    BASE_MODEL = ABLITERATED_MODEL if os.path.exists(ABLITERATED_MODEL) else "Qwen/Qwen2.5-Coder-14B"

MAX_SEQ_LENGTH = 1024  # Reduced from 2048: activation memory is linear with seq_len — biggest OOM lever on T4 x2
LOAD_IN_4BIT = True                                                   
DTYPE = None                                                                  
print(f"\n{'='*60}")
print(f"Loading {BASE_MODEL} in 4-bit quantization...")
print(f"Max sequence length: {MAX_SEQ_LENGTH}")
print(f"{'='*60}\n")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,           # Reduced from 32: halves LoRA gradient memory on T4 x2
    lora_alpha=32,  # Kept at 2x r (standard ratio)
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


from datasets import load_dataset
dataset_arg = args.dataset

import glob

# Determine datasets to train on
datasets_to_train = []
if os.path.isdir(dataset_arg):
    datasets_to_train = sorted(glob.glob(os.path.join(dataset_arg, "*.jsonl")))
    print(f"[DATA] Found {len(datasets_to_train)} shards in directory {dataset_arg}")
else:
    datasets_to_train = [dataset_arg]

def parse_messages(example):
    """Decode a dataset record without silently repairing malformed turns."""
    messages = example.get("messages", [])
    if isinstance(messages, str):
        messages = json.loads(messages)
    elif isinstance(messages, list) and messages and isinstance(messages[0], str):
        messages = [json.loads(message) if isinstance(message, str) else message for message in messages]

    if not isinstance(messages, list) or not all(isinstance(message, dict) for message in messages):
        raise ValueError("messages must be a list of message objects")
    return messages


def audit_dataset(dataset, dataset_name):
    """Fail before training if the input cannot reinforce the Cogito persona."""
    invalid_examples = []
    sources = Counter()

    for index, example in enumerate(dataset):
        try:
            messages = parse_messages(example)
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
            f"the canonical Cogito validation ({examples}). Rebuild it with "
            "`python data/build_dense_dataset.py`."
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

    if not args.allow_unbalanced_dataset and (identity_ratio < 0.20 or probing_ratio < 0.08):
        raise ValueError(
            f"Refusing an identity-starved dataset ({dataset_name}). "
            f"Got identity={identity_ratio:.1%} (need ≥20%) and probing={probing_ratio:.1%} (need ≥8%). "
            "The HuggingFace dataset must be the multiplied dense build, not the raw shards. "
            "Fix: run `python data/build_dense_dataset.py` locally, then "
            "`python scripts/upload_dense_dataset_to_hub.py` to push the correct 7068-record dataset. "
            "Or pass --allow-unbalanced-dataset to skip this check."
        )


def format_example(example):
    """Convert one already-audited record into the ChatML training format."""
    messages = [
        m for m in parse_messages(example)
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

from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
from unsloth.chat_templates import train_on_responses_only

CPT_REVISION = "main"  # Checkpoints always land on the Hub's main branch, never a new one.

# Checkpoint Hub pushes MUST NOT run synchronously inside the DDP training loop:
# the upload runs only on rank 0 and takes minutes for ~425MB, so rank 1 races
# ahead into the next gradient collective (all_gather/all_reduce) while rank 0 is
# still uploading. NCCL's watchdog then times the desynced collective out after 30
# minutes and kills the whole run (observed on 2x T4). So on_save only enqueues a
# checkpoint dir and returns instantly; a single daemon thread uploads the dirs in
# order. Because upload_folder only commits its revision at the end, a process
# killed mid-upload leaves the Hub at the previous checkpoint, so resume stays safe.
import queue
import threading

_checkpoint_upload_queue: "queue.Queue[str | None]" = queue.Queue()
_checkpoint_upload_stop = threading.Event()


HUB_KEEP_CHECKPOINTS = 2  # Keep the N most recent checkpoints on the Hub; prune the rest.


def _prune_old_hub_checkpoints(hf_token: str):
    """Delete older checkpoint-N/ dirs on the Hub, keeping the HUB_KEEP_CHECKPOINTS
    most recent. Keeps the repo lean (~2 x 424MB instead of one per 50 steps)."""
    from huggingface_hub import HfApi

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
            if entry.type == "folder" and name.startswith("checkpoint-"):
                try:
                    steps.append(int(name.split("-")[-1]))
                except ValueError:
                    continue
        if len(steps) <= HUB_KEEP_CHECKPOINTS:
            return
        stale = sorted(steps)[:-HUB_KEEP_CHECKPOINTS]
        for step in stale:
            print(f"[HF] Pruning old checkpoint-{step} from Hub...")
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
    """Background uploader. None in the queue is the shutdown sentinel."""
    from huggingface_hub import upload_folder

    while True:
        ckpt_dir = _checkpoint_upload_queue.get()
        try:
            if ckpt_dir is None or _checkpoint_upload_stop.is_set():
                return
            step = int(os.path.basename(ckpt_dir).split("-")[-1])
            print(f"\n[HF] Pushing checkpoint {step} to '{CPT_REVISION}'...")
            upload_folder(
                repo_id="ozaa77/Cogito-0.9",
                folder_path=ckpt_dir,
                # Keep each step under its own folder on the Hub. Without this,
                # upload_folder flattens the checkpoint onto the repo root and the
                # next save silently overwrites the previous one, so only the last
                # step survives and resume-from-Hub can never find any checkpoint.
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


_checkpoint_upload_thread: "threading.Thread | None" = None


class SavePeftModelCallback(TrainerCallback):
    """Enqueue each saved checkpoint to the background Hub uploader.

    HF's save_strategy='steps' already writes the full checkpoint (adapter weights,
    trainer_state.json with the step counter, optimizer/scheduler state) to
    --training-output-dir/checkpoint-<step>. We only mirror that same checkpoint to
    the Hub so a killed Kaggle session can resume from it. Checkpoints always land
    on main (CPT_REVISION); a new branch is never created.
    """

    def on_save(self, args, state, control, **kwargs):
        # Fires after ANY save: step multiples of 50 AND epoch-end saves (the two
        # things that set control.should_save). Mirrors the just-written checkpoint
        # to the Hub so a stop at a non-50 step still persists. kwarg 'output_dir' is
        # the dir HF actually wrote this checkpoint to.
        global _checkpoint_upload_thread
        if not IS_MAIN_PROCESS or not PUSH_CHECKPOINTS:
            return control
        ckpt_dir = (
            kwargs.get("output_dir")
            or os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
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
        # Enqueue and return immediately; the worker handles the network I/O.
        _checkpoint_upload_queue.put(ckpt_dir)
        return control


def _flush_checkpoint_uploads():
    """Block until every enqueued checkpoint has been pushed (call at end of run)."""
    if _checkpoint_upload_thread is not None:
        _checkpoint_upload_queue.join()

class SaveAtEpochEndCallback(TrainerCallback):
    """Also checkpoint+push when each epoch completes, not just at 50-step marks.

    HF's save_strategy='steps' only saves at exact step multiples. If an epoch or the
    whole run ends on a step that isn't a multiple of 50, nothing would be recorded.
    This flags should_save at every epoch boundary so the finished state is always
    persisted (and mirrored to the Hub) regardless of step count.
    """

    def on_epoch_end(self, args, state, control, **kwargs):
        if not IS_MAIN_PROCESS:
            return control
        control.should_save = True
        return control


OUTPUT_DIR = os.path.abspath(args.output_dir or os.path.join(PROJECT_ROOT, "cogito_0.9_lora"))
TRAINING_OUTPUT_DIR = os.path.abspath(
    args.training_output_dir or os.path.join(PROJECT_ROOT, "cogito_training_output")
)

def get_chat_eos_token_id(tokenizer):
    """Return ChatML's end-of-turn token, not an arbitrary tokenizer EOS."""
    chat_eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if chat_eos_token_id is not None and chat_eos_token_id != tokenizer.unk_token_id:
        return chat_eos_token_id
    return tokenizer.eos_token_id


CHAT_EOS_TOKEN_ID = get_chat_eos_token_id(tokenizer)
print(f"[TOKENIZER] ChatML end-of-turn token id: {CHAT_EOS_TOKEN_ID}")
# SFTConfig (a TrainingArguments subclass) is used instead of TrainingArguments to
# avoid a pickling error during checkpoint saves when Unsloth patches TRL internals.
# SFTTrainer internally converts TrainingArguments -> SFTConfig, which breaks class
# identity checks in Python's pickle; passing SFTConfig directly bypasses this.
training_args = SFTConfig(
    output_dir=TRAINING_OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,  # Doubled (was 16) to keep effective batch size = 32 after seq_len halved
    learning_rate=2e-5,                                    
    lr_scheduler_type="cosine",                      
    warmup_ratio=0.05,                            
    num_train_epochs=args.epochs,                                        
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",                                              
    weight_decay=0.01,
    max_grad_norm=1.0,                                    
    logging_steps=5,
    logging_first_step=True,
    report_to="none",
    # Native HF checkpointing persists trainer_state.json + optimizer + scheduler,
    # which is what makes true resume-from-checkpoint possible. Checkpoints land in
    # --training-output-dir (training-output), distinct from the final adapter dir.
    save_strategy="steps",
    save_steps=50,
    save_total_limit=10,
    load_best_model_at_end=False,
    seed=42,
    dataloader_pin_memory=True,
    ddp_find_unused_parameters=False,
    push_to_hub=False,  # The completed adapter is pushed manually once, on main.
    # SFTConfig-specific fields (avoids deprecation warnings when passed to SFTTrainer)
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=False,
)

# Where to resume from, in priority order:
#   1. --resume-from-checkpoint flag (explicit local or hub path)
#   2. Latest local checkpoint in --training-output-dir
#   3. Latest checkpoint pulled from the Hub (ozaa77/Cogito-0.9) main branch
#
# A Kaggle session wipes /kaggle/working on restart, so the local fs cannot be
# trusted to survive. The Hub is the source of truth: checkpoints are pushed there
# during training, and pulled back here when the local copy is gone.
HUB_REPO_ID = "ozaa77/Cogito-0.9"


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


def collect_checkpoint_step(revision: str = "main") -> str:
    """Refresh a local snapshot dir from the Hub's checkpoint-N folders.

    Returns the path to the latest checkpoint present on the Hub, synced to
    /kaggle/working so the trainer can resume from the actual adapter weights +
    optimizer + scheduler + step state that were pushed.
    """
    from huggingface_hub import snapshot_download

    cache_dir = os.path.join(TRAINING_OUTPUT_DIR, "hub_snapshot")
    print(f"\n[HF] Pulling latest checkpoint from {HUB_REPO_ID}@{revision} ...")
    snapshot_download(
        repo_id=HUB_REPO_ID,
        revision=revision,
        repo_type="model",
        allow_patterns=["checkpoint-*/*", "checkpoint-*"],
        cache_dir=cache_dir,
        token=os.environ.get("HF_TOKEN"),
    )
    # snapshot_download writes a hashed tree; find the checkpoint subdirs we need.
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

    # Legacy layout: older runs pushed each checkpoint FLAT onto the repo root via
    # upload_folder without path_in_repo, so only the final state survives (adapter +
    # optimizer + scheduler + trainer_state.json, no checkpoint-N/ folder). HF's
    # Trainer.resume_from_checkpoint cannot read a flat dir, so rebuild a proper
    # checkpoint-N/ folder locally from the root files and resume from that.
    legacy_root = os.path.join(TRAINING_OUTPUT_DIR, "hub_snapshot", "legacy_flat")
    os.makedirs(legacy_root, exist_ok=True)
    state_path = os.path.join(legacy_root, "trainer_state.json")
    if not os.path.isfile(state_path):
        print("[HF] No checkpoint-NNN folder on the Hub; checking for legacy flat checkpoint...")
        try:
            snapshot_download(
                repo_id=HUB_REPO_ID,
                revision=revision,
                repo_type="model",
                # Adapter files belong to the flat push and go with the checkpoint;
                # README/.gitattributes/tokenizer extras are excluded by naming.
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
        # trainer_state.json references files by bare name; Trainer requires them
        # relative to the checkpoint dir, which the flat layout already satisfies.
        ckpt_dir = os.path.join(TRAINING_OUTPUT_DIR, f"checkpoint-{step}")
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


# 1) explicit flag wins; then local; then pull from the Hub.
resume_dir = args.resume_from_checkpoint
if not resume_dir:
    resume_dir = find_latest_local_checkpoint(TRAINING_OUTPUT_DIR)
    if resume_dir:
        print(f"\n[RESUME] Local checkpoint found: {resume_dir}")
if not resume_dir:
    resume_dir = collect_checkpoint_step(revision="main")

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
                
                # Fetch all parquet files from the repo
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

        # Shard i finished. The next shard must NOT resume from shard i's checkpoint,
        # or the trainer would think its epoch is already advanced. Reset it so shard
        # i+1 (and any later resume) starts from a clean step counter.
        checkpoint_path = None
except Exception as e:
    print(f"\n[FATAL] Training crashed: {e}")
    raise
# Block until every enqueued checkpoint has actually reached the Hub. The worker
# thread is a daemon, so without this the process would exit and drop the last
# uploads (the epoch-end save) even though training itself finished fine.
_flush_checkpoint_uploads()
if IS_MAIN_PROCESS:
    print(f"[SAVE] Saving LoRA adapter to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n{'='*60}")
    print("ALL DONE.")
    print(f"  LoRA adapter saved to: {OUTPUT_DIR}")
    print(f"  To run inference: python run.py")
    print(f"{'='*60}")
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token and not args.no_push_to_hub:
        print("\n[HF] HF_TOKEN detected. Pushing completed LoRA adapter to the main branch...")
        try:
            model.push_to_hub("ozaa77/Cogito-0.9", token=hf_token, revision="main")
            tokenizer.push_to_hub("ozaa77/Cogito-0.9", token=hf_token, revision="main")
            print("[HF] Final push to main successful! Model available at: https://huggingface.co/ozaa77/Cogito-0.9")
        except Exception as e:
            print(f"[HF ERROR] Failed to push to Hugging Face: {e}")
