"""Pre-tokenize the SFT mix once, so neither DDP rank tokenizes anything.

Kaggle's 2x T4 sessions died under the kernel OOM killer (rank 0, SIGKILL,
exit -9) with both torchrun ranks inside dataset preparation: TRL's preload
phases (Add EOS -> Tokenize -> Truncate) and Unsloth's response-mask map each
build a full-corpus copy per rank, and on ~29 GB host RAM the second rank does
not fit while the first is in flight (that session's tokenization was killed
and retried repeatedly before rank 0 went down entirely). This module does the
whole preparation in ONE process and stores plain Arrow via save_to_disk; both
ranks then load_from_disk the same memory-mapped files, so in-training
preprocessing RAM drops to ~0 and tokenization is paid once per dataset,
not once per rank.

Semantic parity with the in-rank pipeline, read out of unsloth-zoo's source:

- Rendering: format_dataset — the model's own chat template, the trainer's
  shared code path.
- Tokenizing: replicates sft_prepare_dataset's double-BOS guard
  (add_special_tokens=False when a BOS appears in the template or the
  rendered text); truncation to --max-seq-length equals TRL's keep_start.
- Masking: train_on_responses_only(None, ..., return_function=True) returns
  Unsloth's own production span mask, so assistant labels — including each
  turn's trailing <|im_end|> — are computed exactly as at runtime. TRL's
  "Adding EOS" pass appends an eos after the final turn's terminator, which
  sits outside every mask span and never trained a token; skipping it is
  therefore token-for-token equivalent here. Rows whose labels end up all
  -100 (the response marker was truncated away) are dropped, matching
  train_on_responses_only's _filter_fully_masked NaN guard.

    python -m qwen_sft pretokenize --dataset data/merged/qwen_sft_mix.jsonl \
        --max-seq-length 4096 --output-dir training_output/pretokenized

main.py runs this before torchrun and passes --pretokenized-dir; when the
meta.json in the output dir matches (model, max-seq-length, source paths +
sizes) the build is reused, so resume sessions skip straight to training.
A build always lands atomically: the dataset is written to ``<dir>.building``
and swapped in only after meta.json exists.
"""

import argparse
import json
import os
import shutil
import sys

from qwen_sft.config import (
    DATA_DIR,
    DEFAULT_BASE_MODEL,
    DEFAULT_DATASETS,
    DEFAULT_LIMIT_PER_DATASET,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_SEED,
    INSTRUCTION_PART,
    PROJECT_ROOT,
    RESPONSE_PART,
)

META_NAME = "meta.json"
BUILD_SUFFIX = ".building"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize + response-mask the SFT dataset once, offline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", action="append", default=None,
                        help="Registry key, HF dataset id, .jsonl file, or a "
                             "directory of .jsonl (repeatable; default: "
                             "openthoughts, astra)")
    parser.add_argument("--limit-per-dataset", type=int,
                        default=DEFAULT_LIMIT_PER_DATASET,
                        help="Max conversations kept per source when preparing")
    parser.add_argument("--prepare-dir", default=DATA_DIR,
                        help="Where prepared JSONL lives")
    parser.add_argument("--refresh-data", action="store_true",
                        help="Re-download/re-prepare even if a cached JSONL exists")
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL,
                        help="Base model the chat template/tokenizer come from")
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-mask-prompt", action="store_true",
                        help="Bake labels = input_ids (loss on every token) "
                             "instead of the assistant-only mask; the training "
                             "run must then also pass --no-mask-prompt")
    parser.add_argument("--output-dir", default=os.path.join(
        PROJECT_ROOT, "training_output", "pretokenized"),
        help="Where the Arrow cache + meta.json are written")
    return parser.parse_args(argv)


def read_meta(output_dir) -> dict | None:
    try:
        with open(os.path.join(output_dir, META_NAME), encoding="utf-8") as fh:
            meta = json.load(fh)
        return meta if isinstance(meta, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def source_entries(sources):
    return [{"path": os.path.abspath(path), "size": int(size)}
            for path, size in sources]


def matches_meta(meta, model, max_seq_length, sources) -> bool:
    """Is this Arrow cache exactly what a restart with these args needs?

    Identity is pinned to everything that would silently change the tokens a
    rank trains on: the tokenizer/checkpoint, the truncation limit, and the
    source files (path + size — a re-merged mix changes the size even when the
    path is reused). Anything else, such as a Kaggle subset run, fails the
    match and triggers a rebuild instead of one rank silently retraining on
    another run's tokens.
    """
    if not isinstance(meta, dict):
        return False
    return (meta.get("model") == model
            and meta.get("max_seq_length") == max_seq_length
            and meta.get("masked") in (True, False)
            and meta.get("sources") == source_entries(sources))


def write_meta(output_dir, model, max_seq_length, masked, rows, sources) -> None:
    meta = {
        "kind": "qwen_sft_pretokenized",
        "model": model,
        "max_seq_length": max_seq_length,
        "masked": masked,
        "rows": int(rows),
        "sources": source_entries(sources),
    }
    with open(os.path.join(output_dir, META_NAME), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def build_mask_function(tokenizer, instruction_part, response_part):
    """Unsloth's production response mask as a standalone batch function.

    ``train_on_responses_only(None, ..., return_function=True)`` resolves the
    markers once (with the exact force_match tokenization the runtime path
    uses) and hands back the per-batch labelling function — no trainer, no
    model, no GPU. Mapping it here guarantees the offline labels are the same
    spans the runtime masking would compute on the fly.
    """
    try:
        from unsloth_zoo.dataset_utils import train_on_responses_only
    except ImportError:
        # unsloth re-exports the same function; covers installs where the
        # plain unsloth_zoo import surface is absent but unsloth is present.
        from unsloth.chat_templates import train_on_responses_only
    return train_on_responses_only(
        None,
        instruction_part=instruction_part,
        response_part=response_part,
        tokenizer=tokenizer,
        return_function=True,
    )


def resolve_add_special_tokens(tokenizer, sample_text) -> bool:
    """sft_prepare_dataset's double-BOS guard, same decision as at runtime.

    BOS in the template (or already leading the rendered text) means the
    template owns the sequence opening; letting the tokenizer add a second BOS
    duplicates it. Qwen has no BOS so this is usually True — but it must be
    decided here, because the cache freezes whatever the offline tokenization
    chose.
    """
    bos = getattr(tokenizer, "bos_token", None)
    if bos is None:
        return True
    template = getattr(tokenizer, "chat_template", None) or ""
    if bos in template:
        return False
    if sample_text and sample_text.startswith(bos):
        return False
    return True


def _count_fully_masked(dataset) -> int:
    """Rows whose every label is -100: no assistant span survived truncation.

    Scanned column-wise in batches rather than through ``filter`` up front, so
    the (usual) clean-corpus case costs one streaming read and no rewrite.
    """
    dropped = 0
    for batch in dataset.select_columns(["labels"]).iter(batch_size=1000):
        for labels in batch["labels"]:
            if labels is None or not any(l != -100 for l in labels):
                dropped += 1
    return dropped


def pretokenize_dataset(dataset, tokenizer, max_seq_length, mask_function=None):
    """Text rows -> {input_ids, attention_mask, labels}, ready for both ranks.

    ``mask_function`` mirrors the training run's masking choice: ``None``
    builds Unsloth's default assistant-only mask; ``False`` trains on every
    token (labels = input_ids, the --no-mask-prompt equivalent); a callable is
    used directly, which lets tests inject cheap fakes. In every case rows
    whose labels are entirely -100 are dropped — train_on_responses_only
    removes them to stop cross-entropy's 0/0 NaN — and the cache must settle
    that once instead of trusting each rank's filter to agree mid-epoch.
    """
    sample_text = dataset[0].get("text") if len(dataset) else None
    add_special_tokens = resolve_add_special_tokens(tokenizer, sample_text)
    if not add_special_tokens:
        print("[PRETOK] BOS comes from the template; tokenizing without "
              "special tokens (double-BOS guard)")

    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=add_special_tokens,
        )

    tokenized = dataset.map(
        _tokenize, batched=True, num_proc=None,
        remove_columns=dataset.column_names, desc="Pre-tokenizing",
    )

    if mask_function is None:
        mask_function = build_mask_function(
            tokenizer, INSTRUCTION_PART, RESPONSE_PART)
    if mask_function is False:
        masked = tokenized.map(
            lambda batch: {"labels": list(batch["input_ids"])},
            batched=True, num_proc=None, desc="Labelling full sequences",
        )
    else:
        masked = tokenized.map(
            mask_function, batched=True, num_proc=None,
            desc="Pre-masking responses",
        )

    dropped = _count_fully_masked(masked)
    if dropped:
        if dropped / len(masked) >= 0.9:
            print(f"[PRETOK] {dropped:,}/{len(masked):,} rows kept no assistant "
                  "span — usually --max-seq-length cut the response marker "
                  "off; survivors still train.")
        masked = masked.filter(
            lambda row: any(l != -100 for l in row["labels"]), num_proc=None)
        print(f"[PRETOK] dropped {dropped:,} row(s) with no trainable label")
    if len(masked) == 0:
        raise SystemExit(
            "[FATAL] every row lost its assistant span during masking; "
            "check the chat template / --max-seq-length.")
    return masked


def load_pretokenized(args):
    """Load the offline Arrow cache — or die loudly instead of silently
    training on the wrong tokens, lengths, or mask."""
    from datasets import load_from_disk

    path = os.path.abspath(args.pretokenized_dir)
    meta = read_meta(path)
    if meta is None:
        raise SystemExit(
            f"[FATAL] {path}: no readable {META_NAME} — build it with "
            "`python -m qwen_sft pretokenize` (main.py does this for you).")
    if meta.get("model") != args.model:
        raise SystemExit(
            f"[FATAL] {path} holds tokens from {meta.get('model')!r}, but this "
            f"run is {args.model!r}; rebuild the cache.")
    if meta.get("max_seq_length") != args.max_seq_length:
        raise SystemExit(
            f"[FATAL] {path} was truncated to {meta.get('max_seq_length')}, "
            f"but this run wants {args.max_seq_length}; rebuild the cache.")
    if bool(meta.get("masked")) is not (not args.no_mask_prompt):
        raise SystemExit(
            "[FATAL] the cache's labels were baked "
            f"{'without' if not meta.get('masked') else 'with'} the response "
            "mask, which is the opposite of this run's --no-mask-prompt="
            f"{args.no_mask_prompt}; rebuild it to match.")

    try:
        dataset = load_from_disk(path)
    except Exception as exc:  # torn/foreign dir, or not a Dataset at all
        raise SystemExit(
            f"[FATAL] {path}: cache is unreadable ({type(exc).__name__}: "
            f"{exc}); rebuild it (delete the dir or let main.py refresh it).")
    missing = {"input_ids", "labels"} - set(dataset.column_names)
    if missing:
        raise SystemExit(
            f"[FATAL] {path} is missing token columns {sorted(missing)}; "
            "the cache is corrupt — rebuild it.")
    return dataset


def main(argv=None) -> int:
    args = parse_args(argv)
    dataset_values = args.dataset or list(DEFAULT_DATASETS)

    from qwen_sft.finetune.train import (
        format_dataset,
        load_jsonl_shard,
        normalize_to_text_tokenizer,
        report_length_stats,
        resolve_dataset_files,
    )

    files = resolve_dataset_files(dataset_values, args.limit_per_dataset,
                                  args.seed, args.refresh_data, args.prepare_dir)
    sources = [(path, os.path.getsize(path)) for path in files]
    out_dir = os.path.abspath(args.output_dir)

    if matches_meta(read_meta(out_dir), args.model, args.max_seq_length, sources):
        print(f"[PRETOK] cache is current: {out_dir}")
        return 0

    staged = out_dir + BUILD_SUFFIX
    if os.path.isdir(staged):
        # A dead session's incomplete build; it was never admitted as a cache
        # (meta.json only exists after a finished build), but it would swallow
        # this run's save if left in place.
        shutil.rmtree(staged)

    from transformers import AutoTokenizer
    from datasets import concatenate_datasets

    tokenizer = normalize_to_text_tokenizer(
        AutoTokenizer.from_pretrained(args.model), args.model)

    shards = [load_jsonl_shard(path) for path in files]
    dataset = concatenate_datasets(shards).shuffle(seed=args.seed)
    print(f"[PRETOK] {len(dataset):,} conversations -> {out_dir}")
    text_dataset = format_dataset(dataset, tokenizer)
    report_length_stats(text_dataset, tokenizer, args.max_seq_length)

    masked = pretokenize_dataset(
        text_dataset, tokenizer, args.max_seq_length,
        mask_function=(False if args.no_mask_prompt else None),
    )

    masked.save_to_disk(staged)
    write_meta(staged, args.model, args.max_seq_length,
               masked=not args.no_mask_prompt, rows=len(masked), sources=sources)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.rename(staged, out_dir)
    print(f"[PRETOK] ready: {out_dir} ({len(masked):,} rows, "
          f"masked={not args.no_mask_prompt})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
