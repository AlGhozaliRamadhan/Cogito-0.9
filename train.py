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
from unsloth import FastLanguageModel
BASE_MODEL = "Qwen/Qwen2.5-Coder-14B"
MAX_SEQ_LENGTH = 4096                                                                
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
DATASET_PATH = os.path.join(os.path.dirname(__file__), "cogito_0.9_master_dataset.jsonl")
if not os.path.isfile(DATASET_PATH):
    print(f"[FATAL] Dataset not found: {DATASET_PATH}")
    print("        Run merge_datasets.py first to create it.")
    sys.exit(1)
print(f"[DATA] Loading dataset from: {DATASET_PATH}")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
print(f"[DATA] Loaded {len(dataset)} training examples")
def format_example(example):
    """
    Convert a single dataset example into the tokenized ChatML format.
    Each example has a "messages" list of dicts with "role" and "content".
    We apply the Qwen chat template, which produces the <|im_start|> / <|im_end|>
    delimited format the model was pretrained on.
    """
    messages = example["messages"]
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,                                          
        add_generation_prompt=False                                       
    )
    return {"text": formatted_text}
print("[DATA] Applying Qwen ChatML template...")
dataset = dataset.map(format_example, remove_columns=dataset.column_names)
print(f"[DATA] Formatted. Sample length: {len(dataset[0]['text'])} chars")
print(f"\n{'─'*60}")
print("SAMPLE (first 500 chars):")
print(f"{'─'*60}")
print(dataset[0]["text"][:500])
print(f"{'─'*60}\n")
from trl import SFTTrainer
from transformers import TrainingArguments
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "cogito_0.9_lora")
training_args = TrainingArguments(
    output_dir=os.path.join(os.path.dirname(__file__), "cogito_training_output"),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,                                    
    lr_scheduler_type="cosine",                      
    warmup_ratio=0.05,                            
    num_train_epochs=3,                                        
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",                                              
    weight_decay=0.01,
    max_grad_norm=1.0,                                    
    logging_steps=5,
    logging_first_step=True,
    report_to="none",                                                   
    save_strategy="epoch",
    save_total_limit=2,                                                  
    seed=42,
    dataloader_pin_memory=True,
    push_to_hub=True if os.environ.get("HF_TOKEN") else False,
    hub_model_id="ozaa77/Cogito-0.9",
    hub_strategy="checkpoint",
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",                                                
    packing=True,                                                                     
)
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
print(f"[SAVE] Saving LoRA adapter to: {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n{'='*60}")
print("ALL DONE.")
print(f"  LoRA adapter saved to: {OUTPUT_DIR}")
print(f"  To run inference: python run.py")
print(f"{'='*60}")
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    print(f"\n[HF] HF_TOKEN detected. Pushing LoRA adapter to Hugging Face Hub...")
    try:
        model.push_to_hub("ozaa77/Cogito-0.9", token=hf_token)
        tokenizer.push_to_hub("ozaa77/Cogito-0.9", token=hf_token)
        print(f"[HF] Push successful! Model available at: https://huggingface.co/ozaa77/Cogito-0.9")
    except Exception as e:
        print(f"[HF ERROR] Failed to push to Hugging Face: {e}")
