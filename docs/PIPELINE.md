# Pipeline

## 1. Choose a trainable base

The linked HauhauCS repo is GGUF-only. Train the safetensors repack of the same
uncensored weights (the default). Do not point this at the official
`Qwen/Qwen3.5-4B`: you would lose the uncensored behaviour before training even
starts. There is no automatic substitute — a missing repack is a hard failure.

## 2. Prepare and merge data

```bash
python -m qwen_sft.datasets.merge --list
python -m qwen_sft.datasets.merge --all-sft --limit 20000
```

The merge prepares each source into `data/prepared/<key>.jsonl`, drops
duplicate conversations (the Astra mirror and the Manusagents/cleaned overlap
collapse), and writes `data/merged/qwen_sft_mix.jsonl`. Do it one source at a
time if you want to inspect each shard:

```bash
python -m qwen_sft prepare --source openthoughts --limit 20000
python -m qwen_sft prepare --source distill_cleaned --limit 5000
```

Preparation streams the split, shuffles with a seeded buffer, normalizes each
record to chat messages, validates it, and writes `data/prepared/<key>.jsonl`.
Cached files are reused by the trainer; pass `--refresh` after raising `--limit`
to rebuild them.

Sampling controls:

- `--limit` caps how many conversations are kept (default 20 000).
- `--no-streaming` downloads the whole split first — avoid for `fable`/`am`.
- `HF_TOKEN` raises the anonymous rate limit.

## 3. Train

```bash
python -m qwen_sft train \
    --dataset data/merged/qwen_sft_mix.jsonl \
    --epochs 1 --max-seq-length 4096
```

To include Fable (agentic tool loops, 40+ turns, tens of thousands of tokens),
raise the sequence budget and keep it a minority of the mix:

```bash
python -m qwen_sft prepare --source fable --limit 2000
python -m qwen_sft train --dataset openthoughts --dataset astra --dataset fable \
    --max-seq-length 4096 --limit-per-dataset 8000
```

Pipeline inside the trainer: resolve files → load 4-bit base → attach LoRA →
render with the model's chat template → mask prompts → SFT → save adapter.

Outputs:

- `lora_adapter/` — final adapter (~50-150 MB for r=16)
- `training_output/checkpoint-N/` — adapter + trainer state, two kept

On Kaggle, keep the vocabulary small and the run short. A 4B at seq 2048,
batch 1, grad-accum 16 fits a single 16 GB T4; `torchrun --nproc_per_node=2`
uses both if you want the extra throughput.

## 4. Test

```bash
python -m qwen_sft chat --adapter ./lora_adapter
```

Compare against the base on the same prompts with `--adapter` omitted. Format
changes are the expected result; reasoning gains need a held-out benchmark, not
a vibe check.

## 5. Publish / merge

```bash
# adapter
python -m qwen_sft train --dataset openthoughts --epochs 1 \
    --hub-repo you/qwen35-4b-sft --push-to-hub

# standalone model (CPU merge, push without local 8 GB copy)
python -m qwen_sft merge --adapter you/qwen35-4b-sft \
    --push-repo you/qwen35-4b-sft-merged --skip-local-save
```

The merged safetensors model can then be quantized to GGUF for llama.cpp/Ollama
with `llama.cpp`'s `convert_hf_to_gguf.py` and `llama-quantize`.

## Known limits

- **Truncation:** on the merged mix the median is ~1.4k tokens and the p90 is
  ~10.9k, so set `--max-seq-length 4096` at minimum; 31% of samples still exceed
  it, and the agentic Fable and Claude Code sessions will be cut.
- **Mix dominance:** sample counts decide the blend. 1.4M AM rows against 20k
  Astra rows is not a mix, it is AM with garnish.
- **Forgetting:** more than ~2 epochs on a narrow distribution erodes general chat.
- **Licences:** check each card. AM-DeepSeek-R1-Distilled-1.4M is CC-BY-NC-4.0.
