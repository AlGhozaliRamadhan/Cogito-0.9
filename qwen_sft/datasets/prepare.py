"""Materialize a Hugging Face source into local JSONL.

    python -m qwen_sft.datasets.prepare --list
    python -m qwen_sft.datasets.prepare --source openthoughts --limit 20000
    python -m qwen_sft.datasets.prepare --source all --limit 5000
    python -m qwen_sft.datasets.prepare --source open-thoughts/OpenThoughts-114k --limit 500

Output is ``data/prepared/<name>.jsonl``, one ``{"messages": [...]}`` object per
line, which is exactly what ``qwen_sft.finetune.train`` consumes.

Unauthenticated Hub requests are rate limited; set ``HF_TOKEN`` for large pulls.
"""

import argparse
import os
import sys

from qwen_sft.config import DATA_DIR, DEFAULT_LIMIT_PER_DATASET, DEFAULT_SEED
from qwen_sft.datasets.sources import SOURCES, describe, get_source, prepare_source


def output_path_for(key: str, out_dir: str) -> str:
    return os.path.join(out_dir, f"{key}.jsonl")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare SFT datasets as chat JSONL")
    parser.add_argument("--source", action="append", default=None,
                        help="Registry key, HF dataset id, or 'all' (repeatable)")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT_PER_DATASET,
                        help="Max conversations to keep per source")
    parser.add_argument("--out-dir", default=DATA_DIR, help="Where to write <key>.jsonl")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-streaming", action="store_true",
                        help="Download the full split before sampling (needs disk)")
    parser.add_argument("--shuffle-buffer", type=int, default=2000,
                        help="Streaming shuffle buffer (0 = take the first rows in order)")
    parser.add_argument("--list", action="store_true", help="Show known sources and exit")
    args = parser.parse_args(argv)

    if args.list or not args.source:
        print(describe())
        return 0

    requested = []
    for value in args.source:
        if value.strip().lower() == "all":
            requested.extend(SOURCES)
        else:
            requested.append(value)

    failures = 0
    for value in requested:
        try:
            source = get_source(value)
        except KeyError as exc:
            print(f"[SKIP] {exc}")
            failures += 1
            continue

        out_path = output_path_for(source.key, args.out_dir)
        print(f"[PREP] {source.key} <- {source.repo_id} (limit={args.limit})")
        try:
            stats = prepare_source(
                source.key,
                out_path,
                limit=args.limit,
                seed=args.seed,
                streaming=not args.no_streaming,
                shuffle_buffer=args.shuffle_buffer,
            )
        except Exception as exc:  # network, auth, or schema drift
            print(f"[FAIL] {source.key}: {type(exc).__name__}: {exc}")
            failures += 1
            continue
        print(f"[OK]   {stats.summary()}")
        if stats.written == 0:
            print(f"[WARN] {source.key} produced no usable records; check the schema.")
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
