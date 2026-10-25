"""Prepare and merge the SFT sources into one training file.

    python -m qwen_sft.datasets.merge --list
    python -m qwen_sft.datasets.merge --all-sft --limit 20000
    python -m qwen_sft.datasets.merge --source astra --source openthoughts --limit 5000
    python -m qwen_sft.datasets.merge --all-sft --limit 2000 --max-per-source 1000

Each source is materialized once into ``data/prepared/<key>.jsonl`` (reused on
later runs, ``--refresh`` to rebuild), then the shards are deduplicated by
conversation content, shuffled, and written to ``data/merged/`` as a single
``{"messages": [...]}`` JSONL that ``qwen_sft.finetune.train`` accepts directly.

Dedupe matters here: the Manusagents set and its saidutta69 "cleaned" repack
overlap heavily, and cudabenchmarktest's Astra is a mirror of guell00's.

Red-team datasets are deliberately not in ``--all-sft``. See
``qwen_sft.datasets.sources.REDTEAM_SOURCES`` for why each one is eval/DPO
material rather than SFT.
"""

import argparse
import hashlib
import json
import os
import random
import sys

from qwen_sft.config import DATA_DIR, DEFAULT_LIMIT_PER_DATASET, DEFAULT_SEED, MERGED_DIR
from qwen_sft.datasets.sources import SOURCES, describe, get_source, prepare_source
from qwen_sft.validation import sanitize, validate_chat


class SourceStats:
    def __init__(self, key: str):
        self.key = key
        self.read = 0
        self.kept = 0
        self.duplicates = 0
        self.invalid = 0

    def row(self) -> str:
        return (f"{self.key:<18} {self.read:>8} {self.kept:>8} "
                f"{self.duplicates:>8} {self.invalid:>8}")


def record_hash(messages: list[dict]) -> str:
    payload = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def read_jsonl(path: str):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def merge_files(
    inputs: list[tuple[str, str]],
    max_per_source: int | None = None,
    dedupe: bool = True,
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict], list[SourceStats]]:
    """Merge ``(key, path)`` shards, dropping duplicates and invalid records."""
    seen: set[str] = set()
    rows: list[dict] = []
    stats: list[SourceStats] = []

    for key, path in inputs:
        source_stats = SourceStats(key)
        kept_here = 0
        for record in read_jsonl(path):
            source_stats.read += 1
            messages = sanitize(record.get("messages") or [])
            is_valid, _reason = validate_chat(messages)
            if not is_valid:
                source_stats.invalid += 1
                continue
            digest = record_hash(messages)
            if dedupe and digest in seen:
                source_stats.duplicates += 1
                continue
            if max_per_source is not None and kept_here >= max_per_source:
                continue
            seen.add(digest)
            rows.append({"messages": messages})
            kept_here += 1
        source_stats.kept = kept_here
        stats.append(source_stats)

    random.Random(seed).shuffle(rows)
    return rows, stats


def resolve_inputs(
    keys: list[str],
    limit: int,
    prepare_dir: str,
    refresh: bool,
    shuffle_buffer: int,
) -> list[tuple[str, str]]:
    inputs = []
    for key in keys:
        source = get_source(key)
        path = os.path.join(prepare_dir, f"{source.key}.jsonl")
        needs_prepare = refresh or not os.path.isfile(path) or os.path.getsize(path) == 0
        if needs_prepare:
            print(f"[PREP] {source.key} <- {source.repo_id} (limit={limit})")
            try:
                stats = prepare_source(
                    source, path, limit=limit, shuffle_buffer=shuffle_buffer
                )
            except Exception as exc:  # one broken shard must not kill the merge
                print(f"[FAIL] {source.key}: {type(exc).__name__}: {exc}")
                continue
            print(f"[PREP] {stats.summary()}")
        else:
            print(f"[SKIP] {source.key}: reusing {path}")
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            inputs.append((source.key, path))
        else:
            print(f"[WARN] {source.key} produced nothing; excluded from the merge.")
    return inputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare + merge SFT datasets")
    parser.add_argument("--source", action="append", default=None,
                        help="Registry key (repeatable)")
    parser.add_argument("--all-sft", action="store_true",
                        help="Use every registered SFT source")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT_PER_DATASET,
                        help="Max conversations kept per source")
    parser.add_argument("--max-per-source", type=int, default=None,
                        help="Cap after dedupe, to control the final mix")
    parser.add_argument("--prepare-dir", default=DATA_DIR)
    parser.add_argument("--out", default=os.path.join(MERGED_DIR, "qwen_sft_mix.jsonl"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--shuffle-buffer", type=int, default=500,
                        help="Streaming shuffle buffer while preparing (small is faster)")
    parser.add_argument("--refresh", action="store_true", help="Re-prepare cached shards")
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    if args.list or (not args.source and not args.all_sft):
        print(describe())
        return 0

    keys: list[str] = []
    if args.all_sft:
        keys.extend(SOURCES)
    for value in args.source or []:
        if value not in keys:
            keys.append(value)

    try:
        inputs = resolve_inputs(
            keys, args.limit, args.prepare_dir, args.refresh, args.shuffle_buffer
        )
    except KeyError as exc:
        print(f"[FATAL] {exc}")
        return 1

    if not inputs:
        print("[FATAL] No prepared shards to merge.")
        return 1

    rows, stats = merge_files(
        inputs,
        max_per_source=args.max_per_source,
        dedupe=not args.no_dedupe,
        seed=args.seed,
    )

    print(f"\n{'source':<18} {'read':>8} {'kept':>8} {'dupes':>8} {'invalid':>8}")
    for source_stats in stats:
        print(source_stats.row())
    print(f"{'TOTAL':<18} {sum(s.read for s in stats):>8} {len(rows):>8} "
          f"{sum(s.duplicates for s in stats):>8} {sum(s.invalid for s in stats):>8}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\n[MERGE] {len(rows):,} conversations -> {args.out} ({size_mb:.1f} MB)")
    if rows:
        avg_turns = sum(len(r["messages"]) for r in rows) / len(rows)
        print(f"[MERGE] avg {avg_turns:.1f} turns/conversation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
