"""
upload_dense_dataset_to_hub.py — Upload the multiplied combined_dense_dataset.jsonl
to ozaa77/Cogito-0.9-dataset on HuggingFace.

This is the correct file to train on: it has shard multipliers already applied so the
identity/probing ratios satisfy audit_dataset() thresholds (≥20% identity, ≥8% probing).

Run from the project root:
  python scripts/upload_dense_dataset_to_hub.py
"""
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATORS_DIR = PROJECT_ROOT / "scripts" / "generators"
if str(GENERATORS_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATORS_DIR))

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
if not HF_TOKEN:
    print("[ERROR] HF_TOKEN not found. Set it in .env or as an environment variable.")
    sys.exit(1)

DATASET_REPO_ID = "ozaa77/Cogito-0.9-dataset"
DENSE_DATASET_PATH = PROJECT_ROOT / "combined_dense_dataset.jsonl"

if not DENSE_DATASET_PATH.exists():
    print(f"[ERROR] {DENSE_DATASET_PATH} not found.")
    print("Run first: python data/build_dense_dataset.py")
    sys.exit(1)

from validator import canonicalize_system_prompt, validate_conversation_structure


def load_dense_records(path: Path) -> list[dict]:
    """Load and validate every record from the dense dataset."""
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
    """Print the identity/probing ratios audit_dataset() will see."""
    from collections import Counter
    sources: Counter = Counter()
    for r in records:
        source = str(r.get("oversample") or r.get("source") or "unknown")
        sources[source] += 1

    total = len(records)
    identity = sum(v for k, v in sources.items() if "cogito_identity_core" in k)
    probing  = sum(v for k, v in sources.items() if "cogito_philosophical_probing" in k)
    print(f"\n[RATIO CHECK] total={total}  identity={identity} ({identity/total:.1%})  probing={probing} ({probing/total:.1%})")
    ok = True
    if identity / total < 0.20:
        print(f"  [FAIL] identity ratio {identity/total:.1%} < 20%")
        ok = False
    if probing / total < 0.08:
        print(f"  [FAIL] probing ratio {probing/total:.1%} < 8%")
        ok = False
    if ok:
        print("  [PASS] Both ratios satisfy audit_dataset() thresholds.")


def main():
    try:
        from datasets import Dataset
        from huggingface_hub import HfApi
    except ImportError:
        print("[ERROR] Install: pip install datasets huggingface_hub")
        sys.exit(1)

    api = HfApi(token=HF_TOKEN)

    print(f"Loading {DENSE_DATASET_PATH} ...")
    records = load_dense_records(DENSE_DATASET_PATH)
    check_ratios(records)

    # Serialize messages list as JSON string for Parquet compatibility
    for r in records:
        if isinstance(r.get("messages"), list):
            r["messages"] = json.dumps(r["messages"], ensure_ascii=False)

    ds = Dataset.from_list(records)
    print(f"\n[HF] Pushing {len(ds)} records to {DATASET_REPO_ID} ...")
    ds.push_to_hub(
        DATASET_REPO_ID,
        token=HF_TOKEN,
        private=False,
    )
    print(f"[HF] Upload complete! Dataset at: https://huggingface.co/datasets/{DATASET_REPO_ID}")


if __name__ == "__main__":
    main()
