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
    gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[GPU] {gpu_name}  —  {gpu_vram:.1f} GB VRAM")
except ImportError:
    print("[FATAL] PyTorch not installed. Run the environment setup first.")
    sys.exit(1)
from unsloth import FastLanguageModel
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ozaa77/Cogito-0.9-dataset", help="Hugging Face dataset ID to train on")
parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
parser.add_argument("--model", type=str, default=None, help="Base model to load (defaults to Abliterated model or Qwen 14B)")
args = parser.parse_args()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABLITERATED_MODEL = os.path.join(PROJECT_ROOT, "Qwen2.5-Coder-14B-Cogito-Abliterated")

if args.model:
    BASE_MODEL = args.model
else:
    BASE_MODEL = ABLITERATED_MODEL if os.path.exists(ABLITERATED_MODEL) else "Qwen/Qwen2.5-Coder-14B"

MAX_SEQ_LENGTH = 8192                                                                
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
    r=128,                                                                    
    lora_alpha=256,                                                  
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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_arg = args.dataset

import glob

# Determine datasets to train on
datasets_to_train = []
if os.path.isdir(dataset_arg):
    datasets_to_train = sorted(glob.glob(os.path.join(dataset_arg, "*.jsonl")))
    print(f"[DATA] Found {len(datasets_to_train)} shards in directory {dataset_arg}")
else:
    datasets_to_train = [dataset_arg]

def format_example(example):
    """
    Convert a single dataset example into the tokenized ChatML format.
    """
    messages = example.get("messages", [])
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,                                          
        add_generation_prompt=False                                       
    )
    return {"text": formatted_text}

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import train_on_responses_only

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "cogito_0.9_lora")
training_args = TrainingArguments(
    output_dir=os.path.join(PROJECT_ROOT, "cogito_training_output"),
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
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
    save_strategy="epoch",
    save_total_limit=2,                                                  
    seed=42,
    dataloader_pin_memory=True,
    push_to_hub=True if os.environ.get("HF_TOKEN") else False,
    hub_model_id="ozaa77/Cogito-0.9",
    hub_strategy="checkpoint",
)

try:
    for i, current_dataset_path in enumerate(datasets_to_train):
        print(f"\n{'='*60}")
        print(f"PHASE/SHARD {i+1} OF {len(datasets_to_train)}: {current_dataset_path}")
        print(f"{'='*60}\n")
        
        if os.path.isfile(current_dataset_path):
            dataset = load_dataset("json", data_files=current_dataset_path, split="train")
        else:
            dataset = load_dataset(current_dataset_path, split="train")
            
        dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_text_field="text",                                                
            packing=False,
        )
        
        trainer = train_on_responses_only(
            trainer,
            instruction_part=["<|im_start|>user\n", "<|im_start|>system\n"],
            response_part="<|im_start|>assistant\n",
        )
        
        # We only auto-resume if it's the very first execution and a checkpoint exists
        last_checkpoint = None
        if i == 0 and os.path.isdir(training_args.output_dir):
            checkpoints = glob.glob(os.path.join(training_args.output_dir, "checkpoint-*"))
            if checkpoints:
                last_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
                print(f"\n[RESUME] Auto-resuming from checkpoint: {last_checkpoint}")
                
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        metrics = train_result.metrics
        print(f"\n  -> Completed shard {i+1}. Train loss: {metrics.get('train_loss', 'N/A'):.4f}\n")
except Exception as e:
    print(f"\n[FATAL] Training crashed: {e}")
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
