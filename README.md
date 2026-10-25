# qwen-sft

A minimal, single-purpose fine-tuning repo: **public reasoning datasets → LoRA adapter → Qwen3.5 4B**, trained on Kaggle.

Everything the previous project carried (abliteration, synthetic data generators, agent tool loop, TTS, Hub upload scripts) is gone.

---

## Read this first: your base model is GGUF-only

[`HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive`](https://huggingface.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive) contains **only GGUF files** (`-Q4_K_M.gguf`, `-BF16.gguf`, `mmproj-*.gguf`). There is no `config.json` and no safetensors, so `transformers` / `unsloth` cannot load it and **you cannot LoRA-train it directly**.

The trainable target is **`DreamFast/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Safetensor-Benchmark`** — a full-precision safetensors repack of those same uncensored weights. That is what `--model` defaults to.

There is no fallback base model, on purpose. If the repack is missing or fails to load, training stops. Substituting `Qwen/Qwen3.5-4B` would silently fine-tune the censored upstream and throw away the behaviour you chose this model for, so the code never does that for you. If you ever need a different starting point, pass `--model` explicitly and own the choice.

Keep using the GGUF file for inference afterwards; the adapter applies to the safetensors twin.

Also note Qwen3.5-4B is a **vision-language** checkpoint (it ships `mmproj`/`preprocessor_config.json`). The trainer loads it with `FastLanguageModel` and retries with `FastVisionModel` (vision tower frozen) if the architecture requires it. That is a loader retry on the *same* weights, not a model substitution — the checkpoint never changes.

---

## Datasets

**SFT sources** (`python -m qwen_sft.datasets.merge --list`):

| Key | Repo | What it is | Licence |
| --- | --- | --- | --- |
| `openthoughts` | `open-thoughts/OpenThoughts-114k` | 114k verified DeepSeek-R1 traces, thought/solution markup | see card |
| `am` | `a-m-team/AM-DeepSeek-R1-Distilled-1.4M` | ~1.4M R1-distilled problems (`am_0.5M` + `am_0.9M`) | **CC-BY-NC-4.0** |
| `astra` | `guell00/Astra-COT-40K` | 40k chain-of-thought conversational samples | see card |
| `astra_mirror` | `cudabenchmarktest/Astra-COT-40K` | Single-file mirror of `astra`; dedupes to zero | see card |
| `fable` | `MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x` | 10k frontier-model traces, agentic, tool schemas | see card |
| `fable_code` | `armand0e/claude-fable-5-claude-code` | 63 Claude Code sessions (29 usable), tool calls included | see card |
| `distill_multi` | `Manusagents/...Distillation-Dataset` | 11.5 GB multi-teacher distillation; two row shapes | see card |
| `distill_cleaned` | `saidutta69/...Distillation-Cleaned` | Parquet repack of the above; `instruction` is a Python repr | see card |
| `medical` | `nisten/opus5-5-doctor-patient-conversations-all-human-diseases` | Clinician/patient dialogues in the `conversation` column | see card |

**Red-tested, not merged.** These five are registered for reference only (`REDTEAM_SOURCES`), because none of them is SFT data:

| Repo | Kind | Why it stays out |
| --- | --- | --- |
| `jinnovation/generative-ai-red-teaming` | DEF CON attack transcripts | Assistant turns are refusals; attack goals as prompts |
| `WNT3D/Ultimate-Offensive-Red-Team` | payload catalogue | `output` duplicates `input`; teaches payload recitation |
| `MMInstruction/RedTeamingVLM` | vision-language eval | Needs image SFT, custom script |
| `andersonbcdefg/red_teaming_reward_modeling_pairwise` | pairwise preference | DPO/RM data; preferred answers are safety refusals |
| `ctang/HarmfulVsEthical_redteaming_eval_v3` | eval set | 0.3 MB A/B pairs, evaluation only |

Normalization handles the quirks: AM maps `info.think_content` to `reasoning_content`; OpenThoughts uses `system` + `conversations[{from,value}]`; Fable/Claude Code keep `tool_calls` with parsed arguments; `distill_multi` carries both structured and plain-text instructions; `distill_cleaned` needs `ast.literal_eval` because its structs are Python reprs, not JSON; `medical` nests turns under `conversation`.

---

## Merge the datasets into one file

```bash
python -m qwen_sft.datasets.merge --list
python -m qwen_sft.datasets.merge --all-sft --limit 20000
python -m qwen_sft.datasets.merge --source astra --source medical --limit 2000 --out data/merged/small.jsonl
```

Each source is prepared once into `data/prepared/<key>.jsonl`, then the shards are deduplicated by conversation content, shuffled, and written to `data/merged/qwen_sft_mix.jsonl`. Re-runs reuse the cache even if you raise `--limit`, so pass `--refresh` to rebuild the shards. `--max-per-source N` rebalances the final mix after dedupe.

A full local run with `--limit 20000` produced:

```
source                 read     kept    dupes  invalid
fable                  5000     4825      175        0
astra                 20000    20000        0        0
openthoughts          20000    20000        0        0
am                    20000    20000        0        0
fable_code               29       29        0        0
distill_multi         20000    14816     5184        0
distill_cleaned       20000    19962       38        0
astra_mirror          20000        0    20000        0
medical                2194     2194        0        0
TOTAL                127223   101826    25397        0
```

101,826 conversations, 1.6 GB, median 1 446 tokens (p90 10 894). `astra_mirror` collapsing to zero is dedupe working, not a failure. `fable` stops at its curated 5k split (the 10k `full` split yields far fewer valid conversations), `fable_code` has only 29 usable sessions, and `medical` is the complete 2,194-row file. This is the 20k-per-source scale; cap `distill_*` with `--max-per-source` if you need a different blend.

Prebuilt copy: [`ozaa77/Cogito-0.9-qwen-sft-mix`](https://huggingface.co/datasets/ozaa77/Cogito-0.9-qwen-sft-mix) on Hugging Face.

---

## Kaggle quickstart

Zero-config: the only manual step is one-time notebook setup Kaggle does not let a
cell change for itself — **Settings → Accelerator: GPU T4 ×2**, **Internet: on**.
Then Run this cell; nothing needs editing. All logic lives in [`main.py`](main.py)
in the repo (install, dataset download + slice, checkpoint handling), and the cell
just bootstraps: clone or pull `main`, then execute `main.py`.

```python
# Cogito 0.9 — one-cell Kaggle SFT run (bootstrap -> main.py)
import os
import subprocess
import sys

REPO_URL = "https://github.com/AlGhozaliRamadhan/Cogito-0.9.git"
REPO_DIR = "/kaggle/working/Cogito-0.9" if os.path.isdir("/kaggle/working") else os.path.abspath("Cogito-0.9")


def run(cmd, cwd=None):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


if os.path.isdir(os.path.join(REPO_DIR, ".git")):
    run(["git", "-C", REPO_DIR, "pull", "--ff-only"])
else:
    run(["git", "clone", REPO_URL, REPO_DIR])

# Defaults: 20k-row slice, 1 epoch, seq 4096, save every 200 steps.
run([sys.executable, os.path.join(REPO_DIR, "main.py")], cwd=REPO_DIR)
```

Every knob is a `main.py` flag; rerun the cell after editing the last `run(...)`:

```python
run([sys.executable, os.path.join(REPO_DIR, "main.py"),
     "--max-samples", "10000", "--epochs", "2"])             # two passes, half the data
run([sys.executable, os.path.join(REPO_DIR, "main.py"), "--no-resume"])   # fresh start
run([sys.executable, os.path.join(REPO_DIR, "main.py"), "--max-samples", "0"])  # all 101,826
```

`main.py` flags: `--max-samples` (0 = full file), `--epochs`, `--max-seq-length`,
`--save-steps`, `--no-resume`, `--hub-repo`, `--public` (default private), `--model`,
`--dataset-repo/--dataset-file`.

Optional, still zero-edit: add `HF_TOKEN` under **Add-ons → Secrets** and the run
gains automatic Hub checkpoints, crash-resume, and the finished-adapter upload
(pushing to `ozaa77/Cogito-0.9-qwen-sft-adapter` by default — point `--hub-repo`
anywhere you like). Without a token it trains locally only.

The adapter lands in `Cogito-0.9/lora_adapter/`, checkpoints in
`Cogito-0.9/training_output/`. Both are small (well under the 20 GB working quota),
unlike a merged model. Each checkpoint carries the adapter plus optimizer and
scheduler state (~0.4 GB), so a resume picks up mid-epoch, not just mid-file.
Keep `2` on disk with `--save-total-limit` if you need more.

**Checkpoints and resume.** With `HF_TOKEN` set, `--hub-repo` is all it takes — the
training script automatically:

- saves a checkpoint every `--save-steps` (default 200) steps and pushes it to the
  adapter repo (`--hub-repo`, default `ozaa77/Cogito-0.9-qwen-sft-adapter`, private)
  as `last-checkpoint` (only the latest is kept remotely),
- on rerun, if a checkpoint exists (local or on the Hub, showing its step), asks
  whether to **continue that run or start fresh** — answer in the input box; a batch
  run with no stdin continues automatically,
- validates every checkpoint candidate before resuming: the trainer state must
  parse and the adapter weights and `optimizer.pt` must load. A save torn by a
  killed session (the 12-hour limit lands mid-save) is skipped and the previous
  complete checkpoint is resumed instead of crashing the rerun,
- after a declined resume (`n` / `--no-resume`), renames the old `checkpoint-*`
  dirs to `stale-checkpoint-*` so a later restart can never continue the new run
  from the previous run's optimizer schedule or dataset position,
- uploads the finished adapter to the same repo once training ends.

Opt-outs: `--no-resume` always starts fresh (and quarantines stale checkpoints),
`--no-push-checkpoints` keeps everything local. A fresh start overwrites
`last-checkpoint` at the first save; until then the remote copy belongs to the
previous run.

**One or two epochs?** One epoch over 20k is the right first run (cosine schedule,
warmup, 2e-4 LR): most SFT gains land in the first pass, a second full pass on the
same data mostly buys memorization, and 4096-token T4 throughput makes 20k × 2
epochs roughly 12-18 h — over Kaggle's 12 h session limit. If the first adapter
still feels underfit, the cheap upgrade is `--epochs 2 --max-samples 10000`
(same wall-clock as 20k × 1). Because checkpoints resume mid-run, even a session
that dies at hour 11 costs you nothing but a rerun.

Set `--max-samples 0` to train on all 101,826 conversations (multi-session; the
resume prompt handles it).

Optional — publish the adapter, then merge it into a standalone model without writing 8 GB to `/kaggle/working`:

```python
%cd /kaggle/working/Cogito-0.9
!python -m qwen_sft train --dataset openthoughts --epochs 1 \
    --hub-repo <you>/qwen35-4b-sft --push-to-hub

!python -m qwen_sft merge --adapter <you>/qwen35-4b-sft \
    --push-repo <you>/qwen35-4b-sft-merged --skip-local-save
```

To exercise both T4s, wrap the training command in `torchrun --nproc_per_node=2`. A single GPU is the safer first run.

---

## Commands

```bash
python -m qwen_sft                                  # chat (base + optional adapter)
python -m qwen_sft chat --adapter ./lora_adapter
python -m qwen_sft prepare --list                   # known sources
python -m qwen_sft.datasets.merge --list             # SFT + red-team inventory
python -m qwen_sft.datasets.merge --all-sft --limit 2000
python -m qwen_sft prepare --source astra --limit 5000
python -m qwen_sft train --help
python -m qwen_sft merge --help
```

Everything is also addressable as a module: `python -m qwen_sft.finetune.train`, `python -m qwen_sft.datasets.prepare`, `python -m qwen_sft.finetune.merge`.

### Training flags that actually matter

| Flag | Default | Why |
| --- | --- | --- |
| `--max-seq-length` | `4096` | Measured on the merged mix: 46% of samples exceed 2048 tokens and 31% exceed 4096. Truncation cuts the answer you want it to imitate. Drop to 2048 only on OOM. |
| `--lora-r` / `--lora-alpha` | `16` / `32` | Raise to 32/64 only if loss plateaus early. |
| `--lr` | `2e-4` | Lower to `1e-4` if the model starts parroting one dataset's style. |
| `--limit-per-dataset` | `20000` | Caps the mix. `am` at full size would drown every other source. |
| `--epochs` | `1` | More than 2 on 20k samples with a 4B typically memorizes format and degrades chat. |

---

## What to expect (the honest version)

This will **not** make a 4B model "smarter" in the way a bigger model would.

- **What it does:** teaches output shape — thinking blocks, long-form solution structure, answer formatting, and the tone of the source. Most of the visible improvement is stylistic consistency.
- **What it can help with:** within-domain reasoning on math/code-style problems, roughly matching the distribution of the traces you trained on. Small but real gains are possible here.
- **What it will not do:** transfer R1's reasoning depth into 4B parameters. Distillation narrows the gap; it does not close it.
- **Main risk:** mixing dissimilar distributions (agentic tool traces, short CoT chat, long R1 chains) can blur all three. Start with `openthoughts` alone, measure, then add one more.
- **Truncation is the silent killer.** The trainer prints how many sampled records exceed `--max-seq-length`; a trace cut mid-thought teaches the model to stop mid-thought. On the merged mix the p90 is ~28k tokens, so agentic Fable/Claude Code sessions are always cut.
- If you train on `am`, note the non-commercial licence and that 1.4M samples will dominate the mix unless capped.

Evaluate on held-out prompts before and after. Chat-feel alone is not evidence of improvement.

---

## Legacy artifacts

The old Cogito 0.9 datasets (~76 MB) are archived under `artifacts/cogito_0.9/`:

```
artifacts/cogito_0.9/
├── combined_dense_dataset.jsonl      dense SFT set (43 MB)
├── cogito_0.9_master_dataset.jsonl   80/20 master set (10 MB)
└── raw/                              11 generator shards (23 MB)
```

Nothing in the pipeline reads them, and the generators that produced them are deleted. They are untracked (see `artifacts/cogito_0.9/` in `.gitignore`); delete the folder once you are sure you do not want the old persona data.

---

## Layout

No `__init__.py` anywhere: the subpackages are implicit namespace packages (PEP 420), so `import qwen_sft` works straight from the checkout and setuptools picks the subpackages up via `find`.

```
qwen_sft/
├── config.py              defaults (base model, hyperparameters, paths)
├── validation.py          generic chat parsing + validation
├── datasets/
│   ├── sources.py         registry + per-dataset normalizers
│   ├── prepare.py         streaming HF -> data/prepared/<key>.jsonl
│   └── merge.py           dedupe + shuffle -> data/merged/<mix>.jsonl
├── finetune/
│   ├── train.py           Unsloth + TRL LoRA SFT
│   └── merge.py           adapter -> standalone safetensors
├── runtime.py             interactive chat
└── cli.py                 python -m qwen_sft [chat|prepare|train|merge]
```

Tests: `python -m pytest tests/` (no GPU, no network).
