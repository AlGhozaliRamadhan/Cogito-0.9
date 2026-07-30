#!/usr/bin/env python3
"""
train_cogito.py — Phase 2: Fine-tune Qwen2.5-Coder-14B into Cogito 0.9.

Uses Unsloth + QLoRA (4-bit quantization) to fit a 14B model into 24 GB VRAM.
Loads the master dataset in ShareGPT/OpenAI chat format and applies the
Qwen ChatML template for tokenization.

Usage:
    python train.py

Output:
    ./cogito_0.9_lora/   — The saved LoRA adapter weights
"""

import os
import json
import sys

# ===========================================================================
# Guard: check GPU availability before importing heavy libraries
# ===========================================================================
try:
    import torch
    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available. This script requires a GPU.")
        print("        Ensure you have PyTorch with CUDA installed.")
        sys.exit(1)
    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    print(f"[GPU] {gpu_name}  —  {gpu_vram:.1f} GB VRAM")
except ImportError:
    print("[FATAL] PyTorch not installed. Run the environment setup first.")
    sys.exit(1)

# ===========================================================================
# 1. Load the base model in 4-bit quantization via Unsloth
# ===========================================================================
from unsloth import FastLanguageModel

# --- Model Configuration ---
BASE_MODEL = "Qwen/Qwen2.5-Coder-14B"
MAX_SEQ_LENGTH = 4096          # Qwen2.5 supports up to 128K; 4K is safe for training
LOAD_IN_4BIT = True            # QLoRA: 4-bit NormalFloat quantization
DTYPE = None                   # Auto-detect (bf16 on Ampere+, fp16 otherwise)

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

# ===========================================================================
# 2. Apply LoRA adapters to ALL linear layers
# ===========================================================================
# Targeting all linear layers ensures the model's entire reasoning pathway
# is adapted, not just attention. This is critical for behavioral fine-tuning
# (identity, refusal, monologue) vs. simple knowledge injection.

model = FastLanguageModel.get_peft_model(
    model,
    r=32,                          # LoRA rank — 32 is a good balance for 14B
    lora_alpha=64,                 # Alpha = 2*r is a safe heuristic
    lora_dropout=0.05,             # Light dropout to reduce overfitting
    target_modules=[               # All linear layers in the transformer
        "q_proj", "k_proj", "v_proj", "o_proj",   # Attention
        "gate_proj", "up_proj", "down_proj",        # MLP (SwiGLU)
    ],
    bias="none",                   # No bias adaptation (standard for LoRA)
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
    random_state=42,
    use_rslora=False,              # Standard LoRA (not Rank-Stabilized)
)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n[LoRA] Trainable: {trainable_params:,} / {total_params:,} "
      f"({100 * trainable_params / total_params:.2f}%)\n")

# ===========================================================================
# 3. Load and format the dataset
# ===========================================================================
from datasets import load_dataset

DATASET_PATH = os.path.join(os.path.dirname(__file__), "cogito_0.9_master_dataset.jsonl")

if not os.path.isfile(DATASET_PATH):
    print(f"[FATAL] Dataset not found: {DATASET_PATH}")
    print("        Run merge_datasets.py first to create it.")
    sys.exit(1)

print(f"[DATA] Loading dataset from: {DATASET_PATH}")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
print(f"[DATA] Loaded {len(dataset)} training examples")

# --- Apply Qwen ChatML template ---
# Qwen2.5 uses the ChatML format:
#   <|im_start|>system\n{content}<|im_end|>\n
#   <|im_start|>user\n{content}<|im_end|>\n
#   <|im_start|>assistant\n{content}<|im_end|>\n
#
# The tokenizer.apply_chat_template() handles this automatically.
# We format each example's "messages" list into a single tokenized string.

def format_example(example):
    """
    Convert a single dataset example into the tokenized ChatML format.
    
    Each example has a "messages" list of dicts with "role" and "content".
    We apply the Qwen chat template, which produces the <|im_start|> / <|im_end|>
    delimited format the model was pretrained on.
    """
    messages = example["messages"]
    
    # apply_chat_template returns a string with all the ChatML tokens
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,            # Return string, not token IDs
        add_generation_prompt=False  # Don't add trailing assistant prompt
    )
    
    return {"text": formatted_text}


print("[DATA] Applying Qwen ChatML template...")
dataset = dataset.map(format_example, remove_columns=dataset.column_names)
print(f"[DATA] Formatted. Sample length: {len(dataset[0]['text'])} chars")

# Print a truncated sample for sanity checking
print(f"\n{'─'*60}")
print("SAMPLE (first 500 chars):")
print(f"{'─'*60}")
print(dataset[0]["text"][:500])
print(f"{'─'*60}\n")

# ===========================================================================
# 4. Training configuration
# ===========================================================================
from trl import SFTTrainer
from transformers import TrainingArguments

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "cogito_0.9_lora")

training_args = TrainingArguments(
    # --- Output ---
    output_dir=os.path.join(os.path.dirname(__file__), "cogito_training_output"),
    
    # --- Batch size ---
    # Effective batch = per_device * gradient_accumulation = 2 * 4 = 8
    # This fits comfortably in 24 GB with 4-bit quant + gradient checkpointing.
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    
    # --- Learning rate ---
    learning_rate=2e-4,                # Standard for QLoRA
    lr_scheduler_type="cosine",        # Smooth decay
    warmup_ratio=0.05,                 # 5% warmup
    
    # --- Duration ---
    num_train_epochs=3,                # 3 passes over the data
    
    # --- Precision ---
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    
    # --- Optimization ---
    optim="adamw_8bit",                # 8-bit Adam from bitsandbytes
    weight_decay=0.01,
    max_grad_norm=1.0,                 # Gradient clipping
    
    # --- Logging ---
    logging_steps=5,
    logging_first_step=True,
    report_to="none",                  # No W&B / TensorBoard by default
    
    # --- Saving ---
    save_strategy="epoch",
    save_total_limit=2,                # Keep only the last 2 checkpoints
    
    # --- Misc ---
    seed=42,
    dataloader_pin_memory=True,
)

# ===========================================================================
# 5. Initialize the SFT Trainer
# ===========================================================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",         # Column name from our format_example()
    packing=True,                      # Pack short examples together (saves GPU time)
)

# ===========================================================================
# 6. Train
# ===========================================================================
print(f"\n{'='*60}")
print("TRAINING STARTED")
print(f"  Model:          {BASE_MODEL}")
print(f"  Dataset:        {len(dataset)} examples")
print(f"  Epochs:         {training_args.num_train_epochs}")
print(f"  Batch size:     {training_args.per_device_train_batch_size} × {training_args.gradient_accumulation_steps} = {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Learning rate:  {training_args.learning_rate}")
print(f"  Precision:      {'bf16' if training_args.bf16 else 'fp16'}")
print(f"  Output:         {OUTPUT_DIR}")
print(f"{'='*60}\n")

try:
    train_result = trainer.train()
    
    # --- Log training metrics ---
    metrics = train_result.metrics
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"  Total steps:    {metrics.get('total_flos', 'N/A')}")
    print(f"  Train loss:     {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Train runtime:  {metrics.get('train_runtime', 'N/A'):.1f}s")
    print(f"{'='*60}\n")
    
except Exception as e:
    print(f"\n[FATAL] Training crashed: {e}")
    print("        Check VRAM usage. If OOM, reduce per_device_train_batch_size to 1.")
    raise

# ===========================================================================
# 7. Save the LoRA adapter
# ===========================================================================
print(f"[SAVE] Saving LoRA adapter to: {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Also save a merged 16-bit model for easier deployment (optional, requires more disk)
# Uncomment the following lines if you want a standalone merged model:
#
# MERGED_DIR = os.path.join(os.path.dirname(__file__), "cogito_0.9_merged_16bit")
# print(f"[SAVE] Merging LoRA into base model and saving to: {MERGED_DIR}")
# model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")

print(f"\n{'='*60}")
print("ALL DONE.")
print(f"  LoRA adapter saved to: {OUTPUT_DIR}")
print(f"  To run inference: python run.py")
print(f"{'='*60}")
