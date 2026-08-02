"""
merge_datasets.py — Merge all Cogito 0.9 dataset shards into a single master JSONL.
Enforces an 80/20 ratio between Agentic/Coding data and Personality data.
"""
import json
import os
import random
import sys

generators_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "generators"))
if generators_dir not in sys.path:
    sys.path.append(generators_dir)

from validator import (
    canonicalize_system_prompt,
    validate_conversation_structure,
)

DATASET_DIR = os.path.join(os.path.dirname(__file__), "raw")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "cogito_0.9_master_dataset.jsonl")

AGENTIC_SHARDS = [
    "cogito_agentic_tools.jsonl",
    "cogito_execution_engine.jsonl",
]

PERSONALITY_SHARDS = [
    "cogito_identity_core.jsonl",
    "cogito_retrieval_filter.jsonl",
    "cogito_human_conversations.jsonl",
    "cogito_personality_quirks.jsonl",
    "cogito_heated_conversations.jsonl",
    "cogito_philosophical_probing.jsonl",
]

def validate_record(record: dict) -> bool:
    messages = record.get("messages")
    is_valid, _ = validate_conversation_structure(messages)
    return is_valid

def load_shards(shard_list):
    records = []
    stats = {}
    for shard_name in shard_list:
        path = os.path.join(DATASET_DIR, shard_name)
        if not os.path.isfile(path):
            continue
        loaded, accepted, rejected = 0, 0, 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                loaded += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    rejected += 1
                    continue
                record["messages"] = canonicalize_system_prompt(record.get("messages"))
                if validate_record(record):
                    record["source"] = shard_name
                    records.append(record)
                    accepted += 1
                else:
                    rejected += 1
        stats[shard_name] = {"loaded": loaded, "accepted": accepted, "rejected": rejected}
    return records, stats

def main():
    print("Loading Agentic/Coding Shards...")
    agentic_records, agentic_stats = load_shards(AGENTIC_SHARDS)
    
    print("Loading Personality Shards...")
    personality_records, personality_stats = load_shards(PERSONALITY_SHARDS)
    
    num_agentic = len(agentic_records)
    num_personality = len(personality_records)
    
    print(f"\n[RAW COUNTS] Agentic: {num_agentic}, Personality: {num_personality}")
    
    if num_agentic == 0:
        print("[FATAL] No valid agentic records found.")
        sys.exit(1)
        
    # We want Agentic to be 80% of the total dataset.
    # So Personality (20%) should be 1/4 of Agentic.
    target_personality_count = num_agentic // 4
    
    if num_personality > target_personality_count:
        print(f"[*] Downsampling Personality data from {num_personality} to {target_personality_count} to maintain 80/20 ratio.")
        random.seed(42)
        personality_records = random.sample(personality_records, target_personality_count)
    elif num_personality < target_personality_count:
        print(f"[WARNING] Not enough personality data to make 20%. Have {num_personality}, need {target_personality_count}.")
        print(f"          Agentic will be > 80%. Consider generating more personality data.")
        
    all_records = agentic_records + personality_records
    random.seed(42)
    random.shuffle(all_records)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print(f"\n{'='*60}")
    print(f"Master dataset written: {OUTPUT_FILE}")
    print(f"Total examples: {len(all_records)}")
    final_agentic = len(agentic_records)
    final_personality = len(personality_records)
    print(f"  Agentic/Coding (80% target): {final_agentic} ({final_agentic/len(all_records)*100:.1f}%)")
    print(f"  Personality (20% target): {final_personality} ({final_personality/len(all_records)*100:.1f}%)")
    
    print("\n[Stats per Shard]")
    all_stats = {**agentic_stats, **personality_stats}
    for shard, s in all_stats.items():
        print(f"  {shard}: {s['accepted']} accepted (out of {s['loaded']})")
    print(f"{'='*60}")
    
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print(f"\n[HF] HF_TOKEN detected. Pushing master dataset to Hugging Face Hub...")
        try:
            from datasets import load_dataset
            ds = load_dataset("json", data_files=OUTPUT_FILE, split="train")
            ds.push_to_hub("ozaa77/Cogito-0.9-dataset", token=hf_token)
            print(f"[HF] Dataset push successful! Available at: https://huggingface.co/datasets/ozaa77/Cogito-0.9-dataset")
        except Exception as e:
            print(f"[HF ERROR] Failed to push dataset to Hugging Face: {e}")

if __name__ == "__main__":
    main()
