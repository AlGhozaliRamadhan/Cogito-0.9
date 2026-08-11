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
python -m cogito.generators.agentic_tools
python -m cogito.generators.execution_engine
python -m cogito.generators.identity_core
python -m cogito.generators.human_conversations
python -m cogito.generators.philosophical_probing
python -m cogito.generators.personality_quirks
python -m cogito.generators.retrieval_filter
python -m cogito.generators.heated_conversations
python -m cogito.generators.livemurmur_engine
python -m cogito.generators.generate_assertions   # hand-written, no API
python -m cogito.generators.generate_identity_self  # hand-written, no API
```

Or the batch runner:
```bash
python -m cogito.scripts.dataset_manager --run-all
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
python -m cogito.scripts.merge_datasets        # → data/cogito_0.9_master_dataset.jsonl
python -m cogito.scripts.build_dense_dataset   # → data/combined_dense_dataset.jsonl
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
python -m cogito.scripts.train --dataset data/combined_dense_dataset.jsonl --epochs 1
```

`train.py` is a module-level script: the CUDA check runs at import and exits
`1` with a clear message on CPU-only machines. Verify the masking setup first
on a small model:

```bash
python -m cogito.scripts.verify_masking         # needs GPU + unsloth/trl
```

Training behavior: 4-bit Qwen2.5-Coder-14B, `train_on_responses_only`,
packing, gradient checkpointing, live 50-step murmurs, per-epoch adapter
saves pushed to `ozaa77/Cogito-0.9` (main branch).

## 5. Abliterate

```bash
python -m cogito.scripts.abliterate_cogito --adapter <cogito_adapter_path> --output-dir cogito_0.9_abliteration_adapter
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
python -m cogito.scripts.abliterate_cogito --adapter <cogito_adapter_path> \
    --target-layer 0.65 --refusal-weight 0.7 --smoke-test

## 6. Merge full model (optional, for inference/deploy)

```bash
python -m cogito.scripts.merge_lora --adapter cogito_0.9_abliteration_adapter --output-dir cogito_0.9_merged
python -m cogito.scripts.merge_lora --adapter cogito_0.9_abliteration_adapter --output-dir cogito_0.9_merged_16bit --push-to-hub
```

## 7. Upload datasets

```bash
python -m cogito.scripts.upload_dataset_to_hub              # raw shards → ozaa77/Cogito-0.9-dataset
python -m cogito.scripts.upload_dense_dataset_to_hub        # dense (multiplied) → ozaa77/Cogito-0.9-dataset
```

Both require `HF_TOKEN` (`.env` or environment) and exit 1 with a clear
message when absent.

## 8. Cleanup hub checkpoints

```bash
python -m cogito.scripts.cleanup_hub --keep 5 --make-public
```

## 9. Run the model

```bash
python -m cogito                     # GPU required; loads ozaa77/Cogito-0.9
python -m cogito --model cogito_0.9_merged_16bit
python -m cogito --voice kore_meet_caroline_outdoors  # needs kokoro
```

## Subprocess reference

Every CLI is launched via `python -m cogito.*`:

| Invocation                          | Module                              |
|-------------------------------------|-------------------------------------|
| `python -m cogito`                  | `cogito.cli` → `cogito.runtime`     |
| `python -m cogito.cli`              | `cogito.cli` → `cogito.runtime`     |
| `python -m cogito.scripts.train`    | `cogito.scripts.train` (module-level CUDA check) |
| `python -m cogito.scripts.abliterate_cogito` | `cogito.scripts.abliterate_cogito` |
| `python -m cogito.scripts.merge_lora`       | `cogito.scripts.merge_lora`        |
| `python -m cogito.scripts.merge_datasets`   | `cogito.scripts.merge_datasets`    |
| `python -m cogito.scripts.build_dense_dataset` | `cogito.scripts.build_dense_dataset` |
| `python -m cogito.scripts.dataset_manager`  | `cogito.scripts.dataset_manager`   |
| `python -m cogito.scripts.cleanup_hub`      | `cogito.scripts.cleanup_hub`       |
| `python -m cogito.scripts.verify_masking`   | `cogito.scripts.verify_masking`    |
| `python -m cogito.scripts.prepare_datasets` | `cogito.scripts.prepare_datasets`  |
| `python -m cogito.generators.<name>`        | one of 11 generators               |
| `python -m cogito.audio.generate_kokoro`   | `cogito.audio.generate_kokoro`     |

## Intentional quirks (do not "fix")

- Generators exit `0` on complete shards (enables idempotent re-runs).
- `merge_datasets` uses `seed(42)` → deterministic rebuilds.
- `cogito.scripts.train` checks CUDA at import (module-level script, no `main()`).
