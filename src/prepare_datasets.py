import os
import shutil
import json
import re

def is_valid_content(content):
    if not isinstance(content, str):
        try:
            content = str(content)
        except Exception:
            return False
            
    content_stripped = content.strip()
    if not content_stripped:
        return False
        
    lower_content = content_stripped.lower()
    if lower_content.startswith("<!doctype html") or lower_content.startswith("<html") or (
        "<body" in lower_content and "<div" in lower_content and "</a>" in lower_content):
        return False
        
    try:
        parsed = json.loads(content_stripped)
        if isinstance(parsed, (dict, list)):
            return False
    except ValueError:
        pass
        
    if len(content_stripped) > 15000:
        return False
        
    return True

def append_hf_datasets():
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets library not found.")
        return
        
    hf_datasets = [
        ("jason-oneal/mitre-stix-cve-exploitdb-dataset-alpaca-chatml-harmony", "chatml"),
        ("AlicanKiraz0/All-CVE-Records-Training-Dataset", None),
        ("hitoshura25/cvefixes", None),
        ("Luoberta/cve_train", None),
        ("ChipHolmes/All-CVE-Records-Training-Dataset-archive", None),
        ("Manusagents/GPT-5.5-Gemini-3.1-Pro-Grok-4-Claude-Fable-5-Mythos-5-Qwen-3.7-Max-and-more-Distillation-Dataset", None),
        ("Glint-Research/Fable-5-traces", None),
        ("nvidia/OpenCodeInstruct", None),
        ("SYNTHETIC-2-SFT-verified", None),
        ("KIMI-K2.5-1000000x", None),
        ("MathX-5M", None),
        ("Phips/dense-reasoning-coding-1k", None),
        ("ethanker/agentic_coding_dataset", None),
        ("kaist-ai/Multifaceted-Collection-small", None),
        ("kulia-moon/deeprethink", None),
        ("Raymond-dev-546730/Open-CoT-Reasoning-Mini", None),
        ("Mahadih534/multilingual-chain-of-thought-cot-datasets-for-fine-tuning", None),
        ("NexLM/agenticcode", None),
        ("zeli2024/CorrectBench", None)
    ]
    
    shard_dir = os.path.join("data", "hf_shards")
    os.makedirs(shard_dir, exist_ok=True)
    
    print(f"Preparing Hugging Face datasets sequentially into {shard_dir}...")
    for ds_path, config in hf_datasets:
        print(f"Processing {ds_path}...")
        safe_name = ds_path.replace("/", "_").replace("-", "_").lower()
        output_file = os.path.join(shard_dir, f"{safe_name}.jsonl")
        
        # Skip if already processed to save time on Kaggle
        if os.path.exists(output_file):
            print(f"  -> Already exists, skipping.")
            continue
            
        count = 0
        is_cve = "cve" in ds_path.lower()
        
        try:
            if config:
                ds = load_dataset(ds_path, config, split="train", streaming=True)
            else:
                ds = load_dataset(ds_path, split="train", streaming=True)
                
            with open(output_file, "w", encoding="utf-8") as f:
                for i, row in enumerate(ds):
                    if i >= 100: break
                    
                    messages = []
                    
                    if is_cve:
                        cve_id = row.get("cve_id") or row.get("id") or "this vulnerability"
                        description = row.get("description") or row.get("text") or row.get("output") or ""
                        if not description or not is_valid_content(description): continue
                            
                        messages.append({"role": "user", "content": f"Tell me about {cve_id}."})
                        messages.append({"role": "assistant", "content": f"<confidence>0.3</confidence>\n<thought>I do not have this CVE memorized. I must search.</thought>\n<action>web_search</action>\n<query>{cve_id}</query>"})
                        fake_search = description[:500] if len(description) > 500 else description
                        messages.append({"role": "system", "content": f"Search Results:\n{fake_search}"})
                        messages.append({"role": "assistant", "content": f"<confidence>0.95</confidence>\n<thought>I have found information regarding {cve_id} and will provide an answer.</thought>\n<action>answer</action>\n{description}"})
                        
                    elif "input" in row and "output" in row:
                        user_input, assistant_output = row.get("input"), row.get("output")
                        if not user_input or not assistant_output: continue
                        if not is_valid_content(user_input) or not is_valid_content(assistant_output): continue
                            
                        messages.append({"role": "user", "content": str(user_input)})
                        messages.append({"role": "assistant", "content": f"<confidence>0.9</confidence>\n<thought>Generating code to satisfy the user request.</thought>\n<action>generate_code</action>\n<code>\n{assistant_output}\n</code>"})
                        
                    elif "messages" in row:
                        msgs = row["messages"]
                        valid = True
                        temp_messages = []
                        for m in msgs:
                            role, content = m.get("role", ""), m.get("content", "")
                            if not is_valid_content(content):
                                valid = False
                                break
                            if role == "user":
                                temp_messages.append({"role": "user", "content": content})
                            elif role == "assistant":
                                temp_messages.append({"role": "assistant", "content": f"<action>answer</action>\n{content}"})
                        if valid and temp_messages: messages = temp_messages
                            
                    elif "instruction" in row and "output" in row:
                        instruction, output = row.get("instruction"), row.get("output")
                        if not instruction or not output: continue
                        if not is_valid_content(instruction) or not is_valid_content(output): continue
                        messages.append({"role": "user", "content": str(instruction)})
                        messages.append({"role": "assistant", "content": f"<action>answer</action>\n{output}"})
                        
                    if messages:
                        f.write(json.dumps({"messages": messages, "source": ds_path}, ensure_ascii=False) + "\n")
                        count += 1
            print(f"  -> Saved {count} records to {output_file}")
        except Exception as e:
            print(f"  -> Error processing {ds_path}: {e}")

def main():
    os.makedirs("data", exist_ok=True)
    append_hf_datasets()
    print("[DONE] All Hugging Face datasets prepared as separate shards.")

if __name__ == "__main__":
    main()
