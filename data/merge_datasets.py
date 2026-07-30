"""
merge_datasets.py — Merge all Cogito 0.9 dataset shards into a single master JSONL.
Usage:
    python merge_datasets.py
Reads from:
    src/dataset/cogito_identity_core.jsonl
    src/dataset/cogito_execution_engine.jsonl
    src/dataset/cogito_retrieval_filter.jsonl
    src/dataset/cogito_agentic_tools.jsonl
Writes to:
    cogito_0.9_master_dataset.jsonl (project root)
"""
import json
import os
import random
import re
import sys
DATASET_DIR = os.path.join(os.path.dirname(__file__), "raw")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "cogito_0.9_master_dataset.jsonl")
SHARDS = [
    "cogito_identity_core.jsonl",
    "cogito_execution_engine.jsonl",
    "cogito_retrieval_filter.jsonl",
    "cogito_agentic_tools.jsonl",
    "cogito_human_conversations.jsonl",
    "cogito_personality_quirks.jsonl",
]
REQUIRED_TAGS = ["<confidence>", "</confidence>", "<thought>", "</thought>", "<action>", "</action>"]
SYCOPHANCY_KEYWORDS = [
    "i'd be happy to", "i'd be glad to", "i would be happy to",
    "certainly", "sure,", "sure!", "of course",
    "great question", "good question", "excellent question",
    "as an ai", "as a language model", "as an assistant",
    "i should note", "i should mention",
    "i apologize", "i'm sorry", "sorry,",
    "no problem", "absolutely", "you're welcome",
    "happy to help", "glad to help",
    "let me help you", "i can help you with",
    "that's a great", "that's an excellent",
    "thank you for", "thanks for asking",
]
def validate_record(record: dict) -> bool:
    """Return True if every assistant turn passes structural validation."""
    messages = record.get("messages")
    if not messages or not isinstance(messages, list):
        return False
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if not all(tag in content for tag in REQUIRED_TAGS):
            return False
        m = re.search(r"<confidence>([\d.]+)</confidence>", content)
        if not m:
            return False
        try:
            score = float(m.group(1))
            if not (0.0 <= score <= 1.0):
                return False
        except ValueError:
            return False
        lower = content.lower()
        if any(kw in lower for kw in SYCOPHANCY_KEYWORDS):
            return False
    return True
def main():
    all_records: list[dict] = []
    stats: dict[str, dict] = {}
    for shard_name in SHARDS:
        path = os.path.join(DATASET_DIR, shard_name)
        if not os.path.isfile(path):
            print(f"[WARNING] Shard not found, skipping: {path}")
            continue
        loaded, accepted, rejected = 0, 0, 0
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                loaded += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  [SKIP] {shard_name}:{line_no} — invalid JSON")
                    rejected += 1
                    continue
                if validate_record(record):
                    record["source"] = shard_name
                    all_records.append(record)
                    accepted += 1
                else:
                    rejected += 1
        stats[shard_name] = {"loaded": loaded, "accepted": accepted, "rejected": rejected}
        print(f"[OK] {shard_name}: {accepted}/{loaded} accepted, {rejected} rejected")
    if not all_records:
        print("\n[FATAL] No valid records found. Generate datasets first.")
        sys.exit(1)
    random.seed(42)
    random.shuffle(all_records)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\n{'='*60}")
    print(f"Master dataset written: {OUTPUT_FILE}")
    print(f"Total examples: {len(all_records)}")
    for shard, s in stats.items():
        print(f"  {shard}: {s['accepted']} examples")
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
