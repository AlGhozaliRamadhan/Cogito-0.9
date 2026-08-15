"""
upload_dense_hub.py — Upload the multiplied combined_dense_dataset.jsonl
to ozaa77/Cogito-0.9-dataset on HuggingFace.
"""
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_REPO_ID = "ozaa77/Cogito-0.9-dataset"
DENSE_DATASET_PATH = PROJECT_ROOT / "data" / "combined_dense_dataset.jsonl"

from cogito.validation import canonicalize_system_prompt, validate_conversation_structure


def load_dense_records(path: Path) -> list[dict]:
    records = []
    invalid = 0
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as e:
                print(f"  Line {line_no}: JSON error — {e}")
                invalid += 1
                continue

            messages = record.get("messages", [])
            if isinstance(messages, str):
                try:
                    messages = json.loads(messages)
                except Exception:
                    invalid += 1
                    continue

            messages = canonicalize_system_prompt(messages)
            is_valid, reason = validate_conversation_structure(messages)
            if not is_valid:
                print(f"  Line {line_no}: invalid — {reason}")
                invalid += 1
                continue

            record["messages"] = messages
            records.append(record)

    print(f"Loaded {len(records)} valid records ({invalid} invalid/skipped)")
    return records


def check_ratios(records: list[dict]) -> None:
    from collections import Counter
    sources: Counter = Counter()
    for r in records:
        source = str(r.get("oversample") or r.get("source") or "unknown")
        sources[source] += 1

    total = len(records)
    identity = sum(v for k, v in sources.items() if "cogito_identity_core" in k)
    probing = sum(v for k, v in sources.items() if "cogito_philosophical_probing" in k)
    print(f"\n[RATIO CHECK] total={total}  identity={identity} ({identity/total:.1%})  probing={probing} ({probing/total:.1%})")


def main():
    global HF_TOKEN
    try:
        from datasets import Dataset
        from huggingface_hub import HfApi
    except ImportError:
        print("[ERROR] Install: pip install datasets huggingface_hub")
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            with env_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() == "HF_TOKEN":
                            hf_token = v.strip()
                            break

    if not hf_token:
        print("[ERROR] HF_TOKEN not found. Set it in .env or as an environment variable.")
        sys.exit(1)

    if not DENSE_DATASET_PATH.exists():
        print(f"[ERROR] {DENSE_DATASET_PATH} not found.")
        print("Run first: python -m cogito.datasets.build_dense")
        sys.exit(1)

    api = HfApi(token=hf_token)

    print(f"Loading {DENSE_DATASET_PATH} ...")
    records = load_dense_records(DENSE_DATASET_PATH)
    check_ratios(records)

    normalised = []
    for r in records:
        messages = r.get("messages", [])
        if isinstance(messages, str):
            messages = json.loads(messages)
        normalised.append({
            "messages": messages,
            "source": str(r.get("oversample") or r.get("source") or "unknown"),
        })

    ds = Dataset.from_list(normalised)
    print(f"\n[HF] Pushing {len(ds)} records to {DATASET_REPO_ID} ...")
    ds.push_to_hub(
        DATASET_REPO_ID,
        token=HF_TOKEN,
        private=False,
    )
    print(f"[HF] Upload complete! Dataset at: https://huggingface.co/datasets/{DATASET_REPO_ID}")


if __name__ == "__main__":
    main()
