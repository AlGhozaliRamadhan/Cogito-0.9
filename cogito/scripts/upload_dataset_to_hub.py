"""
upload_dataset_to_hub.py — Upload repaired local raw shards to HuggingFace
as ozaa77/Cogito-0.9-dataset (dataset repo).

Reads HF_TOKEN from the .env file or the HF_TOKEN environment variable.
Run from the project root:
  python scripts/upload_dataset_to_hub.py
"""
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Load .env
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    with env_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

HF_TOKEN = os.environ.get("HF_TOKEN")

DATASET_REPO_ID = "ozaa77/Cogito-0.9-dataset"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

SHARD_NAMES = [
    "cogito_agentic_tools.jsonl",
    "cogito_direct_assertions.jsonl",
    "cogito_execution_engine.jsonl",
    "cogito_heated_conversations.jsonl",
    "cogito_human_conversations.jsonl",
    "cogito_identity_core.jsonl",
    "cogito_personality_quirks.jsonl",
    "cogito_philosophical_probing.jsonl",
    "cogito_retrieval_filter.jsonl",
]

from cogito.validation import canonicalize_system_prompt, validate_conversation_structure


def load_valid_records(shard_path: Path) -> list[dict]:
    """Load only valid, canonicalized records from a shard."""
    records = []
    with shard_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            messages = record.get("messages", [])
            if isinstance(messages, str):
                try:
                    messages = json.loads(messages)
                except Exception:
                    continue
            messages = canonicalize_system_prompt(messages)
            is_valid, _ = validate_conversation_structure(messages)
            if is_valid:
                record["messages"] = messages
                records.append(record)
    return records


def main():
    global HF_TOKEN
    try:
        from datasets import Dataset, DatasetDict
        from huggingface_hub import HfApi
    except ImportError:
        print("[ERROR] Install: pip install datasets huggingface_hub")
        sys.exit(1)

    HF_TOKEN = HF_TOKEN or os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN not found. Set it in .env or as an environment variable.")
        sys.exit(1)

    api = HfApi(token=HF_TOKEN)

    # Ensure dataset repo exists
    try:
        api.repo_info(repo_id=DATASET_REPO_ID, repo_type="dataset", token=HF_TOKEN)
        print(f"[HF] Dataset repo {DATASET_REPO_ID} found.")
    except Exception:
        print(f"[HF] Creating dataset repo {DATASET_REPO_ID}...")
        api.create_repo(repo_id=DATASET_REPO_ID, repo_type="dataset", private=False, token=HF_TOKEN)

    all_records = []
    for shard_name in SHARD_NAMES:
        shard_path = RAW_DIR / shard_name
        if not shard_path.exists():
            print(f"[WARN] Missing shard: {shard_path} — skipping")
            continue
        records = load_valid_records(shard_path)
        # Tag each record with its source shard
        for r in records:
            r["source"] = shard_name
        all_records.extend(records)
        print(f"  {shard_name}: {len(records)} valid records")

    print(f"\nTotal records to upload: {len(all_records)}")

    # Serialize messages list as JSON string for Parquet compatibility
    for r in all_records:
        if isinstance(r.get("messages"), list):
            r["messages"] = json.dumps(r["messages"], ensure_ascii=False)

    ds = Dataset.from_list(all_records)
    print(f"\n[HF] Pushing {len(ds)} records to {DATASET_REPO_ID}...")
    ds.push_to_hub(
        DATASET_REPO_ID,
        token=HF_TOKEN,
        private=False,
    )
    print(f"[HF] Upload complete! Dataset at: https://huggingface.co/datasets/{DATASET_REPO_ID}")


if __name__ == "__main__":
    main()
