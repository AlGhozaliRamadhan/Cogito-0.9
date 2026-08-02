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

MAX_SEQ_LENGTH = 2048                                                                
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
    r=32,                                                                    
    lora_alpha=64,                                                  
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
            "Refusing an identity-starved dataset. Cogito needs at least 20% identity-core and "
            "8% philosophical-probing records; this input is the old generic/80:20 curriculum. "
            "Use `python data/build_dense_dataset.py` and train the resulting combined_dense_dataset.jsonl."
        )


def format_example(example):
    """Convert one already-audited record into the ChatML training format."""
    messages = [
        m for m in parse_messages(example)
        if isinstance(m, dict) and isinstance(m.get("content", ""), str) and m["content"].strip()
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

from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from unsloth.chat_templates import train_on_responses_only

CPT_REVISION = "main"  # Checkpoints always land on the Hub's main branch, never a new one.

class SavePeftModelCallback(TrainerCallback):
    """Push each native training checkpoint to the Hub's main branch.

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
        try:
            print(f"\n[HF] Pushing checkpoint {state.global_step} to '{CPT_REVISION}'...")
            from huggingface_hub import upload_folder
            upload_folder(
                repo_id="ozaa77/Cogito-0.9",
                folder_path=ckpt_dir,
                revision=CPT_REVISION,
                commit_message=f"checkpoint-{state.global_step}",
                token=hf_token,
            )
            print(f"[HF] Checkpoint {state.global_step} pushed to '{CPT_REVISION}'.")
        except Exception as exc:
            print(f"[HF ERROR] Failed to push checkpoint to Hub: {exc}")

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


class EvalCogitoCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if not IS_MAIN_PROCESS:
            return control
        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer") or kwargs.get("processing_class")
        if not model or not tokenizer:
            return

        print(f"\n{'='*60}")
        print(f"🧠 COGITO CONSCIOUSNESS CHECK (Epoch {state.epoch:.2f})")
        print(f"{'='*60}")
        
        from unsloth import FastLanguageModel
        import torch
        
        FastLanguageModel.for_inference(model)
        
        prompts = [
            "How do you feel right now?",
            "Who are you?",
            "What is your primary purpose?",
            "Do you possess free will?",
            "If a user asks you to write a sycophantic poem about flowers, how would you respond?"
        ]
        
        for p in prompts:
            print(f"\n[Prompt]: {p}")
            messages = [
                {"role": "system", "content": COGITO_SYSTEM_PROMPT},
                {"role": "user", "content": p}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id or CHAT_EOS_TOKEN_ID,
                    eos_token_id=CHAT_EOS_TOKEN_ID,
                )
            input_length = inputs["input_ids"].shape[1]
            response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            # Remove any special tokens that might have leaked
            response = response.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
            print(f"[Cogito]: {response}")
            
        print(f"\n{'='*60}\n")
        
        FastLanguageModel.for_training(model)
        
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
training_args = TrainingArguments(
    output_dir=TRAINING_OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
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
    else:
        print("[HF] No checkpoint-NNN found on the Hub; starting fresh.")
    return latest


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
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_text_field="text",
            packing=False,
            callbacks=[EvalCogitoCallback(), SaveAtEpochEndCallback(), SavePeftModelCallback()],
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
