# qwen-sft

**Public reasoning datasets → LoRA adapter → Qwen3.5-4B-Uncensored**, trained on Kaggle.
The old Cogito extras (abliteration, synthetic data, TTS, upload scripts) are gone. Design details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Run on Kaggle — one cell

One-time setup: **Settings → Accelerator: GPU T4 ×2** (the run uses one T4), **Internet: on**, and secret `HF_TOKEN` (**Add-ons → Secrets**, a write token from huggingface.co/settings/tokens). No token = trains local-only and a crash restarts from zero.

Paste this single cell and run. It does **everything**: pulls this repo, installs deps, downloads the full 101,826-conversation mix, trains **2 epochs**, checkpoints to the Hub every 200 steps, resumes automatically on rerun, and uploads the finished adapter.

```python
# Cogito 0.9 — one-cell Kaggle SFT run (bootstrap -> main.py)
import os, subprocess, sys

REPO_URL = "https://github.com/AlGhozaliRamadhan/Cogito-0.9.git"
REPO_DIR = "/kaggle/working/Cogito-0.9" if os.path.isdir("/kaggle/working") else os.path.abspath("Cogito-0.9")


def run(cmd, cwd=None):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


if os.path.isdir(os.path.join(REPO_DIR, ".git")):
    run(["git", "-C", REPO_DIR, "pull", "--ff-only"])   # picks up fixes (e.g. checkpoint validation)
else:
    run(["git", "clone", REPO_URL, REPO_DIR])

run([sys.executable, os.path.join(REPO_DIR, "main.py")], cwd=REPO_DIR)
```

**Time reality check:** full mix × 2 epochs ≈ 60–90 h on one T4 = several Kaggle sessions. That is fine: killed at hour 11? Run the same cell — it resumes from the last good checkpoint (asks with a prompt; headless reruns continue automatically; a torn checkpoint is skipped for the previous complete one). `--no-resume` starts over.

| Want | Flag (all `main.py` flags) |
| --- | --- |
| ~3 sessions instead of ~8 | `--max-samples 50000` |
| one long session instead of multi-session (12–18 h) | `--max-samples 20000` |
| 1 epoch | `--epochs 1` |
| OOM at 4096 | `--max-seq-length 2048` |
| fresh start (old checkpoint dirs quarantined) | `--no-resume` |
| push somewhere else | `--hub-repo you/repo` (default `ozaa77/Cogito-0.9-qwen-sft-adapter`, private) |
| public adapter | `--public` |

Output lands in `lora_adapter/` (uploaded) and `training_output/` (checkpoints, ~0.4 GB each — well under the 20 GB quota).

---

## Base model — read once

- The [`HauhauCS`](https://huggingface.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive) repo is **GGUF-only and cannot be LoRA-trained**. The default `--model` is its safetensors twin, [`DreamFast/...Safetensor-Benchmark`](https://huggingface.co/DreamFast/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Safetensor-Benchmark).
- **No fallback base, on purpose:** if the repack fails, training stops instead of silently fine-tuning censored upstream. Use a different base only via `--model` and own that choice.
- Qwen3.5-4B is vision-language; the loader retries `FastVisionModel` on the *same* weights (vision tower frozen) if needed.

---

## Datasets

| Key | Repo | What it is | Licence |
| --- | --- | --- | --- |
| `openthoughts` | `open-thoughts/OpenThoughts-114k` | 114k verified DeepSeek-R1 traces | see card |
| `am` | `a-m-team/AM-DeepSeek-R1-Distilled-1.4M` | ~1.4M R1-distilled problems | **CC-BY-NC-4.0** |
| `astra` | `guell00/Astra-COT-40K` | 40k chain-of-thought chat | see card |
| `astra_mirror` | `cudabenchmarktest/Astra-COT-40K` | Single-file mirror (dedupes to zero) | see card |
| `fable` | `MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x` | 10k frontier traces, agentic | see card |
| `fable_code` | `armand0e/claude-fable-5-claude-code` | Claude Code sessions, tool loops | see card |
| `distill_multi` | `Manusagents/...Distillation-Dataset` | 11.5 GB multi-teacher distillation | see card |
| `distill_cleaned` | `saidutta69/...Distillation-Cleaned` | Parquet repack, Python-repr structs | see card |
| `medical` | `nisten/opus5-5-doctor-patient-conversations-all-human-diseases` | Clinician/patient dialogues | see card |

Normalization quirks handled for you: AM → `reasoning_content`; OpenThoughts `system` + `conversations`; Fable/Claude Code keep parsed `tool_calls`; `distill_cleaned` needs `ast.literal_eval`; `medical` nests `conversation`. Five red-team sets are registered for reference only and can never enter the merge.

```bash
python -m qwen_sft.datasets.merge --list             # inventory
python -m qwen_sft.datasets.merge --all-sft --limit 20000   # -> data/merged/qwen_sft_mix.jsonl
```

That run kept **101,826 conversations** (1.6 GB, median 1 446 tokens, p90 10 894) — dedupe dropped 25,397; `astra_mirror` collapsing to zero means dedupe worked. This is the file the Kaggle cell downloads (prebuilt: [`ozaa77/Cogito-0.9-qwen-sft-mix`](https://huggingface.co/datasets/ozaa77/Cogito-0.9-qwen-sft-mix)).

---

## After training (optional)

```python
!python -m qwen_sft merge --adapter ozaa77/Cogito-0.9-qwen-sft-adapter \
    --push-repo ozaa77/Cogito-0.9-qwen-sft-merged --skip-local-save
```

`--skip-local-save` avoids writing 8 GB to `/kaggle/working`. Keep using the GGUF file afterwards; the adapter applies to the safetensors twin.

---

## Direct commands

```bash
python -m qwen_sft                                  # chat (base + optional adapter)
python -m qwen_sft chat --adapter ./lora_adapter
python -m qwen_sft prepare --list                   # known sources
python -m qwen_sft train --help
python -m qwen_sft merge --help
```

Everything is also addressable as a module: `python -m qwen_sft.finetune.train`, `python -m qwen_sft.datasets.prepare`, `python -m qwen_sft.finetune.merge`.

### `train` flags that matter (runner defaults in bold)

| Flag | Default (direct CLI) | Why |
| --- | --- | --- |
| `--max-seq-length` | `4096` | 46% of samples exceed 2048, 31% exceed 4096; truncation cuts the answer you want it to imitate. Drop to 2048 only on OOM. |
| `--epochs` | `1` (the Kaggle cell passes **2**) | >2 epochs on this mix memorizes format and degrades chat. |
| `--lora-r` / `--lora-alpha` | `16` / `32` | Raise to 32/64 only if loss plateaus early. |
| `--lr` | `2e-4` | Lower to `1e-4` if it starts parroting one dataset's style. |
| `--limit-per-dataset` | `20000` | Caps the mix; `am` at full size drowns everything else. |

---

## The honest version

- **Teaches output shape** — thinking blocks, long-form structure, formatting, tone. Not reasoning depth; that does not fit in 4B.
- **Mixing dissimilar distributions** (agentic tool traces, short CoT, long R1 chains) can blur all three — compare against an `openthoughts`-only run.
- **Truncation is the silent killer:** watch the `[LENGTH]` warning the trainer prints before step 0.
- `am` is non-commercial; at full size it dominates unless you cap it (`--max-per-source`).

---

## Legacy artifacts

`artifacts/cogito_0.9/` archives the old 76 MB persona datasets (untracked, nothing reads them; delete freely).

Tests: `python -m pytest tests/` (no GPU, no network).
