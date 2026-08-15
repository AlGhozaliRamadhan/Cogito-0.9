"""Rebuild the dense SFT dataset from existing local Cogito shards.

This intentionally makes no API calls. Small, high-value identity shards are
repeated so their influence is proportional to their importance rather than
their raw volume.
"""
import json
import sys
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
OUTPUT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "combined_dense_dataset.jsonl"
from cogito.validation import canonicalize_system_prompt, validate_conversation_structure

# Keep this mapping explicit and versionable. The multipliers produce the
# dense dataset used by the training pipeline. Invalid records are omitted.
SHARD_MULTIPLIERS = {
    "cogito_agentic_tools.jsonl": 2,
    "cogito_execution_engine.jsonl": 2,
    # Hand-written direct assertions for identity inquiries (calibrated to 1x to avoid philosophical skew)
    "cogito_direct_assertions.jsonl": 1,
    "cogito_heated_conversations.jsonl": 1,
    "cogito_human_conversations.jsonl": 2,
    "cogito_identity_core.jsonl": 2,
    # Hand-written Cogito identity and contextual directness calibration shard
    "cogito_identity_self.jsonl": 4,
    "cogito_personality_quirks.jsonl": 1,
    # Philosophical probing kept at 1x so the model engages when asked without over-philosophizing standard tasks
    "cogito_philosophical_probing.jsonl": 1,
    "cogito_retrieval_filter.jsonl": 2,
}


def load_records(path: Path) -> tuple[list[dict], int]:
    records = []
    rejected = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
            record = dict(record)
            record["messages"] = canonicalize_system_prompt(record.get("messages"))
            is_valid, _ = validate_conversation_structure(record.get("messages"))
            if is_valid:
                records.append(record)
            else:
                rejected += 1
    return records, rejected


def main() -> None:
    total = 0
    with OUTPUT_PATH.open("w", encoding="utf-8") as output:
        for shard_name, multiplier in SHARD_MULTIPLIERS.items():
            shard_path = RAW_DIR / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(f"Required dense-training shard is missing: {shard_path}")

            records, rejected = load_records(shard_path)
            for copy_index in range(multiplier):
                for record in records:
                    dense_record = dict(record)
                    dense_record["oversample"] = f"{shard_name}_copy{copy_index}"
                    output.write(json.dumps(dense_record, ensure_ascii=False) + "\n")
                    total += 1
            print(
                f"{shard_name}: {len(records)} valid records x {multiplier}"
                f" ({rejected} rejected)"
            )

    print(f"Dense dataset written to {OUTPUT_PATH}: {total} records")


if __name__ == "__main__":
    main()
