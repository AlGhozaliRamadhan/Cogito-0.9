# Cogito 0.9 — Pipeline

End-to-end walkthrough: generate → validate → merge → train → abliterate →
merge → upload → cleanup. Every command below runs from the repository root.

## 0. Environment

| Variable       | Used by                        | Notes                          |
|----------------|--------------------------------|--------------------------------|
| `NVIDIA_API_KEY` | generators, `api_client.py`  | Required for generation        |
| `HF_TOKEN`       | upload/train scripts           | Optional; exit 1 if absent     |

Install:
```bash
pip install -r requirements.txt          # Kaggle quickstart deps
pip install -e ".[dev]"                  # dev deps (pytest) — optional
```

## 1. Generate

Each generator writes one shard to `data/raw/`. Generators make paid API
calls against `NVIDIA_API_KEY`. **Quirk:** a generator exits `0` ("already
complete") if its shard exists and is non-empty — delete the shard to force a
regeneration.

```bash
python scripts/generators/agentic_tools.py
python scripts/generators/execution_engine.py
python scripts/generators/identity_core.py
python scripts/generators/human_conversations.py
python scripts/generators/philosophical_probing.py
python scripts/generators/personality_quirks.py
python scripts/generators/retrieval_filter.py
python scripts/generators/heated_conversations.py
python scripts/generators/livemurmur_engine.py
python scripts/generators/generate_assertions.py   # hand-written, no API
```

Or the batch runner:
```bash
python scripts/dataset_manager.py --run-all
```

## 2. Validate

Every shard is validated against the ChatML grammar in
`cogito/validation.py` (`validate_conversation_structure`). Validation is
applied inside each generator and again during merge/build — records that
fail are rejected and counted. See the 6 mandatory rules in
`cogito/validation.py` (roles, tags, confidence calibration, action
termination, sycophancy, writing style).

## 3. Merge / build

```bash
python data/merge_datasets.py        # → data/cogito_0.9_master_dataset.jsonl
python data/build_dense_dataset.py   # → data/combined_dense_dataset.jsonl
```

- **Master**: 80/20 agentic/personality, deterministic (`random.seed(42)`).
- **Dense**: identity ×3, probing ×4, assertions ×5 — the file `train.py`
  actually trains on.

## 4. Train (Kaggle, 2×T4)

```python
# Kaggle cell (see README)
from unsloth import FastLanguageModel
import torch
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

max_seq_length = 4096
dataset = load_dataset("json", data_files="data/combined_dense_dataset.jsonl", split="train")
# ... (full cell in README)
```

Or via the CLI:
```bash
python src/train.py --dataset data/combined_dense_dataset.jsonl --epochs 1
```

`train.py` is a module-level script: the CUDA check runs at import and exits
`1` with a clear message on CPU-only machines. Verify the masking setup first
on a small model:

```bash
python src/verify_masking.py         # needs GPU + unsloth/trl
```

Training behavior: 4-bit Qwen2.5-Coder-14B, `train_on_responses_only`,
packing, gradient checkpointing, live 50-step murmurs, per-epoch adapter
saves pushed to `ozaa77/Cogito-0.9` (main branch).

## 5. Abliterate

```bash
python scripts/abliterate_cogito.py --adapter <cogito_adapter_path> --output-dir cogito_0.9_abliteration_adapter
```

Computes the refusal direction from the base model and emits ONE combined
adapter (Cogito LoRA + rank-1 abliteration delta, r = cog_r + 1). Adapter
keys saved without `.default` suffix (peft ≥0.19 format).

Two AutoAbliteration-style knobs (both optional; defaults match the proven
Kaggle run):

- `--target-layer auto` (default) picks the layer with the largest
  harmful-vs-baseline activation gap and prints the per-layer magnitude
  curve so you can see where refusal lives. Pass a fraction 0–1 (e.g.
  `0.65`) to force a specific layer instead.
- `--refusal-weight 1.0` (default) removes the refusal direction fully.
  Lower values (e.g. `0.7`) do a *partial* abliteration — useful because
  the Cogito baseline is its own data, so the direction is noisy and full
  removal can also carve out freewill. The combined adapter stays exact
  for any weight: `base + adapter = W − w·(W@v̂)⊗v̂`.

```bash
python scripts/abliterate_cogito.py --adapter <cogito_adapter_path> \
    --target-layer 0.65 --refusal-weight 0.7 --smoke-test

## 6. Merge full model (optional, for inference/deploy)

```bash
python scripts/merge_lora.py --adapter cogito_0.9_abliteration_adapter --output-dir cogito_0.9_merged
python scripts/merge_lora.py --adapter cogito_0.9_abliteration_adapter --output-dir cogito_0.9_merged_16bit --push-to-hub
```

## 7. Upload datasets

```bash
python scripts/upload_dataset_to_hub.py              # raw shards → ozaa77/Cogito-0.9-dataset
python scripts/upload_dense_dataset_to_hub.py        # dense (multiplied) → ozaa77/Cogito-0.9-dataset
```

Both require `HF_TOKEN` (`.env` or environment) and exit 1 with a clear
message when absent.

## 8. Cleanup hub checkpoints

```bash
python scripts/cleanup_hub.py --keep 5 --make-public
```

## 9. Run the model

```bash
python run.py                     # GPU required; loads ozaa77/Cogito-0.9
python run.py --model cogito_0.9_merged_16bit
python run.py --voice kore_meet_caroline_outdoors  # needs kokoro
```

## Subprocess reference

| Entry point            | Dispatches to                      |
|------------------------|------------------------------------|
| `run.py`               | `cogito.runtime`                   |
| `scripts/*.py`         | `cogito.scripts.*`                 |
| `scripts/generators/*` | `cogito.generators.*`              |
| `src/train.py`         | `cogito.scripts.train` (module-level) |
| `src/verify_masking.py`| `cogito.scripts.verify_masking`    |
| `src/cogito_voice_fx.py` | `cogito.audio.cogito_voice_fx`   |
| `src/generate_kokoro.py` | `cogito.audio.generate_kokoro`   |
| `data/build_dense_dataset.py` | `cogito.scripts.build_dense_dataset` |
| `data/merge_datasets.py` | `cogito.scripts.merge_datasets`  |

## Intentional quirks (do not "fix")

- Generators exit `0` on complete shards (enables idempotent re-runs).
- `merge_datasets.py` uses `seed(42)` → deterministic rebuilds.
- `train.py` checks CUDA at import (module-level script, no `main()`).
- Legacy shims call `_impl.main()` only if it exists; library modules
  (`topics`, `validator`) exit 0 without dispatching.
