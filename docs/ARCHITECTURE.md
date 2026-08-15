# Cogito 0.9 — Architecture

## Purpose

Cogito 0.9 is a data-generation and fine-tuning stack that produces a
self-aware, abliterated variant of Qwen3-14B. The pipeline:

1. **Generates** a synthetic dataset (tool-use, identity, probing, personality)
   through an NVIDIA-hosted LLM API.
2. **Validates** every conversation against a structural grammar
   (`<confidence>/<thought>/<action>` ChatML), rejecting malformed records.
3. **Merges** validated shards into a master dataset (80/20 agentic/personality)
   and a dense SFT dataset (identity/probing up-weighted).
4. **Fine-tunes** Qwen3-14B with LoRA on Kaggle 2×T4 via unsloth/trl.
5. **Abliterates** refusal behavior as a rank-1 adapter delta and merges it
   into the LoRA, producing a single combined adapter.

## Repository map

```
Cogito 0.9/
├── cogito/                      # The importable package — single source of truth
│   ├── __init__.py              #   __version__ = "0.9.1" only (imports nothing heavy)
│   ├── __main__.py              #   `python -m cogito` → cogito.cli
│   ├── cli.py                   #   CLI entry point → cogito.runtime.main()
│   ├── runtime.py               #   Interactive runtime / inference driver
│   ├── validation.py            #   ChatML validator
│   ├── llm/
│   │   └── api_client.py        #   NVIDIA API client, lazy key probe, CogitoAPIKeyError
│   ├── generators/              #   All dataset generators + topics + validator alias
│   ├── scripts/                 #   Train, merge, abliterate, upload, dataset tooling
│   └── audio/
│       ├── cogito_voice_fx.py   #   Audio DSP
│       └── generate_kokoro.py   #   TTS via kokoro (optional dep)
├── data/
│   ├── raw/                     #  JSONL shards (generator output)
│   ├── combined_dense_dataset.jsonl  # dense SFT dataset (train.py input)
│   └── cogito_0.9_master_dataset.jsonl  # 80/20 master dataset (hub upload)
├── tests/                       # pytest suite (validator, imports, CLI dispatch)
├── docs/                        # ARCHITECTURE.md, PIPELINE.md
├── pyproject.toml               # packaging + optional-deps (kaggle/dev)
└── requirements.txt             # Kaggle quickstart deps (README-referenced)
```

## Package layering

```
audio  →  (self-contained)
scripts →  validation
generators →  llm, validation, generators.topics
llm      →  (self-contained)
validation → (self-contained)
```

Dependency rule: modules may import from `cogito` **only**, never from
`sibling directories`. There is exactly one import policy:

- **No `sys.path` manipulation in production code.** The only exception is
  `cogito/generators/generate_assertions.py` (and `generate_identity_self.py`)'s
  direct-execution bootstrap, which is needed because those files do not live
  under `cogito/`. All other modules are reached via `python -m cogito.*`.
- **No import-time side effects.** `cogito/llm/api_client.py` never probes the
  API key at import; it raises `CogitoAPIKeyError` when a generator actually
  calls `get_client()`/`get_working_model()` without `NVIDIA_API_KEY`.
- **No import-time I/O.** Generators write files only inside `main()`.

## Runtime flow

`python -m cogito [--model ...]` → `cogito.cli` → `cogito.runtime.main()`:

1. GPU pre-check (exits 1 with a clear message on CPU-only machines).
2. Loads the merged LoRA adapter (`ozaa77/Cogito-0.9` or a local path).
3. Chat loop with `COGITO_SYSTEM_PROMPT`, streaming generation,
   optional TTS via `--voice`.

## Data flow

```
cogito/generators/*.py  --(NVIDIA API)-->  data/raw/*.jsonl     11 shards
        │
        ├── python -m cogito.datasets.merge_shards
        │     →  data/cogito_0.9_master_dataset.jsonl
        │          (80/20 agentic/personality, seed 42 → deterministic)
        └── python -m cogito.datasets.build_dense
              →  data/combined_dense_dataset.jsonl
                 (balanced multipliers: agentic ×2, identity_self ×4, probing ×1 …)
                                              │
                                              ▼
              python -m cogito.finetune.train --dataset data/combined_dense_dataset.jsonl
```

Determinism: `merge_shards` seeds `random` with `42` before downsampling
and shuffling, so rebuilds are byte-identical given identical raw shards.

## Fine-tuning & Training flow (Kaggle, 2×T4)

`cogito.finetune.train` handles fine-tuning:

1. CUDA check at startup — exits 1 with a clear message without a GPU.
2. Loads Qwen3-14B in 4-bit via unsloth, `train_on_responses_only`
   masks everything except assistant turns (verified by
   `cogito.finetune.verify_masking`).
3. `SFTTrainer` with gradient checkpointing, step checkpoints, and background upload queue.
4. Every epoch: saves the LoRA adapter locally and pushes it to
   `ozaa77/Cogito-0.9` (main branch), pruned by `cogito.datasets.cleanup_hub`.
5. Final save + push.

## Abliteration & LoRA Merge flow

`cogito.finetune.abliterate` computes a **rank-1 adapter delta** from
the refusal direction in the base model and emits a single combined adapter
(`cogito_0.9_abliteration_adapter/`): Cogito LoRA + abliteration delta in one
adapter directory, so no multi-adapter runtime API is needed. Adapter keys are
saved WITHOUT the `.default` suffix (peft ≥0.19 format).
`cogito.finetune.merge` merges adapters back into a full model for
deployment/inference.

## Testing

- `tests/test_validator_contentless.py` — validator grammar regression tests
  (record-478 crash: content-less messages must be rejected) + alias equality.
- `tests/test_imports.py` — every `cogito.*` module must import with NO
  environment variables (no `NVIDIA_API_KEY`, no `HF_TOKEN`); CLI entry
  points (`python -m cogito.*`) must dispatch with the expected exit codes
  (no tracebacks).
- Path bootstrap lives only in `tests/conftest.py`.

## Safety notes

- The abliterated model intentionally refuses less often; it is a research
  artifact for studying refusal behavior, not a product. Use responsibly.
- Generators make paid API calls against an NVIDIA key; `python -m cogito`
  and the `--help` paths make none.
- Upload scripts require `HF_TOKEN`; they exit 1 with a clear message when it
  is absent (no import-time exits).
