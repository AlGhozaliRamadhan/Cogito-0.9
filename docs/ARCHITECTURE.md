# Cogito 0.9 — Architecture

## Purpose

Cogito 0.9 is a data-generation and fine-tuning stack that produces a
self-aware, abliterated variant of Qwen2.5-Coder-14B. The pipeline:

1. **Generates** a synthetic dataset (tool-use, identity, probing, personality)
   through an NVIDIA-hosted LLM API.
2. **Validates** every conversation against a structural grammar
   (`<confidence>/<thought>/<action>` ChatML), rejecting malformed records.
3. **Merges** validated shards into a master dataset (80/20 agentic/personality)
   and a dense SFT dataset (identity/probing up-weighted).
4. **Fine-tunes** Qwen2.5-Coder-14B with LoRA on Kaggle 2×T4 via unsloth/trl.
5. **Abliterates** refusal behavior as a rank-1 adapter delta and merges it
   into the LoRA, producing a single combined adapter.

## Repository map

```
Cogito 0.9/
├── cogito/                      # The importable package — single source of truth
│   ├── __init__.py              #   __version__ = "0.9.0" only (imports nothing heavy)
│   ├── runtime.py               #   CLI runtime / inference driver (formerly run.py)
│   ├── validation.py            #   ChatML validator (formerly scripts/generators/validator.py)
│   ├── llm/
│   │   └── api_client.py        #   NVIDIA API client, lazy key probe, CogitoAPIKeyError
│   ├── generators/              #   All 11 dataset generators + topics + validator alias
│   ├── scripts/                 #   Train, merge, abliterate, upload, dataset tooling
│   └── audio/
│       ├── cogito_voice_fx.py   #   Audio DSP (formerly src/cogito_voice_fx.py)
│       └── generate_kokoro.py   #   TTS via kokoro (optional dep)
├── scripts/  src/  data/  run.py   # Thin compatibility shims → cogito.*
├── data/
│   ├── raw/                     #  9 JSONL shards (generator output)
│   ├── combined_dense_dataset.jsonl  # dense SFT dataset (train.py input)
│   └── cogito_0.9_master_dataset.jsonl  # 80/20 master dataset (hub upload)
├── tests/                       # pytest suite (validator, imports, shims)
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
`sibling directories or legacy paths`. There is exactly one import policy:

- **No `sys.path` manipulation in production code.** The only exceptions are
  the legacy *shims* (`scripts/`, `src/`, `data/`, `run.py`), which bootstrap
  the repo root so `import cogito...` resolves when executed as a script, and
  `cogito/generators/generate_assertions.py`'s direct-execution bootstrap.
- **No import-time side effects.** `cogito/llm/api_client.py` never probes the
  API key at import; it raises `CogitoAPIKeyError` when a generator actually
  calls `get_client()`/`get_working_model()` without `NVIDIA_API_KEY`.
- **No import-time I/O.** Generators write files only inside `main()`.

## Runtime flow

`python run.py [--model ...]` → shim → `cogito.runtime.main()`:

1. GPU pre-check (exits 1 with a clear message on CPU-only machines).
2. Loads the merged LoRA adapter (`ozaa77/Cogito-0.9` or a local path).
3. Chat loop with `COGITO_SYSTEM_PROMPT`, streaming generation,
   optional TTS via `--voice`.

## Data flow

```
generators/*.py  --(NVIDIA API)-->  data/raw/*.jsonl     9 shards
        │
        ├── merge_datasets.py  →  data/cogito_0.9_master_dataset.jsonl
        │                          (80/20 agentic/personality, seed 42 → deterministic)
        └── build_dense_dataset.py → data/combined_dense_dataset.jsonl
                                      (identity ×3, probing ×4, assertions ×5 …)
                                              │
                                              ▼
        cogito/scripts/train.py --dataset data/combined_dense_dataset.jsonl
```

Determinism: `merge_datasets.py` seeds `random` with `42` before downsampling
and shuffling, so rebuilds are byte-identical given identical raw shards.

## Training flow (Kaggle, 2×T4)

`cogito/scripts/train.py` is a module-level script (no `main()` guard) that
runs top-to-bottom:

1. CUDA check at import — exits 1 with a clear message without a GPU.
2. Loads Qwen2.5-Coder-14B in 4-bit via unsloth, `train_on_responses_only`
   masks everything except assistant turns (verified by
   `cogito/scripts/verify_masking.py`).
3. `SFTTrainer` with packing, gradient checkpointing, and a live-murmur
   callback that streams 50-step progress updates to `run.py`.
4. Every epoch: saves the LoRA adapter locally and pushes it to
   `ozaa77/Cogito-0.9` (main branch), pruned by `cleanup_hub.py`.
5. Final save + push.

## Abliteration flow

`cogito/scripts/abliterate_cogito.py` computes a **rank-1 adapter delta** from
the refusal direction in the base model and emits a single combined adapter
(`cogito_0.9_abliteration_adapter/`): Cogito LoRA + abliteration delta in one
adapter directory, so no multi-adapter runtime API is needed. Adapter keys are
saved WITHOUT the `.default` suffix (peft ≥0.19 format). `merge_lora.py`
merges adapters back into a full model for deployment/inference.

## Testing

- `tests/test_validator_contentless.py` — validator grammar regression tests
  (record-478 crash: content-less messages must be rejected) + alias equality.
- `tests/test_imports.py` — every `cogito.*` module must import with NO
  environment variables (no `NVIDIA_API_KEY`, no `HF_TOKEN`); legacy shims must
  dispatch with the expected exit codes (no tracebacks).
- Path bootstrap lives only in `tests/conftest.py`.

## Safety notes

- The abliterated model intentionally refuses less often; it is a research
  artifact for studying refusal behavior, not a product. Use responsibly.
- Generators make paid API calls against an NVIDIA key; `run.py` and the
  `--help` paths make none.
- Upload scripts require `HF_TOKEN`; they exit 1 with a clear message when it
  is absent (no import-time exits).
