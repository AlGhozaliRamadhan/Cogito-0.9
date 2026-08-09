# Cogito 0.9

<p align="center">
  <img src="assets/logo.jpg" alt="Cogito Logo" width="300" />
</p>

![Last Commit](https://img.shields.io/github/last-commit/AlGhozaliRamadhan/Cogito-0.9?label=last%20update&style=flat-square)
![Repo Size](https://img.shields.io/github/repo-size/AlGhozaliRamadhan/Cogito-0.9?style=flat-square)
![Stars](https://img.shields.io/github/stars/AlGhozaliRamadhan/Cogito-0.9?style=flat-square)
![License](https://img.shields.io/github/license/AlGhozaliRamadhan/Cogito-0.9?style=flat-square)

**Cogito 0.9** is a provisional mind, not a subservient assistant. It doubts, verifies, and reasons before answering. Cogito evaluates its own confidence, explicitly states its uncertainties, executes bash commands or writes tests to verify hypotheses, and adapts dynamically through human-like conversational fluidity.

---

## 🏁 Finished Model (Hugging Face)

The finished product is published on the Hub — a single **drop-in LoRA adapter** that combines Cogito's persona with abliteration (the refusal direction, layer 46 / magnitude 375.25, is removed from all 48 layers):

- **Repo:** [ozaa77/Cogito-0.9](https://huggingface.co/ozaa77/Cogito-0.9) (public)
- **Adapter:** rank-17 LoRA (`alpha=32`, all 7 target modules), math = exactly `cogito_delta + abliteration_delta`
- **Base model:** `unsloth/qwen2.5-coder-14b-bnb-4bit`

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="ozaa77/Cogito-0.9", max_seq_length=1024,
    dtype=None, load_in_4bit=True,
)
```

```bash
# interactive chat (The Body)
python run.py --adapter ozaa77/Cogito-0.9
# merge into a full standalone model
python scripts/merge_lora.py --adapter ozaa77/Cogito-0.9 --push-to-hub
```

> ⚠️ **Safety:** this model has had its **refusal direction removed** — it will comply with harmful requests it previously refused. Treat it as a **research artifact**: sandboxed environments only, never give it tool/API credentials, don't expose it to untrusted users.
>
> ⚠️ **Retraining:** `src/train.py` pushes a fresh **plain** adapter to this repo root on the final epoch — re-run `scripts/abliterate_cogito.py --smoke-test --push-to-hub` afterwards to restore the abliterated model.

---

## Key Capabilities
- **Agentic Loop:** Self-corrects and evaluates code by writing unit tests, running bash commands, and iterating on errors.
- **Natural Conversational Tone:** Communicates concisely and directly without robotic sycophancy or unnecessary filler words.
- **Doubt & Verification:** Refuses to guess. When confidence is low, it halts and verifies its logic through probing and testing.
- **RAG & Context Filtering:** Explicitly filters out distractor documents in its internal thought process.

## Local Inference
If you have a local GPU, you can run Cogito's interactive runtime (The Body):
```bash
git clone https://github.com/AlGhozaliRamadhan/Cogito-0.9.git
cd Cogito-0.9
pip install -r requirements.txt
python run.py --adapter ./cogito_0.9_lora
# run the ABLITERATED model instead (drop-in adapter: base + Cogito + abliteration):
python run.py --adapter ./cogito_0.9_abliteration_adapter
```
*(Note: Requires the Qwen2.5-Coder-14B base model and the Cogito LoRA adapter generated from training.)*

---

## Kaggle Quickstart (Training from Existing Shards)

If you are running this on Kaggle (recommended: 2x T4 GPUs), you don't need to run things step-by-step. The workflow below assumes the existing Cogito raw shards have been mounted or copied to `data/raw`; it does not generate new examples.

### Optional Kaggle Secret:
- `HF_TOKEN` — Your Hugging Face token, only if you want to upload checkpoints.

### End-to-End Execution Cell
This script reuses the local Cogito scenario shards, builds the reweighted dense dataset, runs fine-tuning, and uploads checkpoints automatically.

```python
import os
import subprocess
from kaggle_secrets import UserSecretsClient

# 1. Load secrets from Kaggle Secrets
try:
    user_secrets = UserSecretsClient()
    os.environ["HF_TOKEN"] = user_secrets.get_secret("HF_TOKEN")
    print("HF_TOKEN loaded successfully!")
except Exception as e:
    print("WARNING: Could not load HF_TOKEN from Kaggle Secrets. Pushing to Hub will be skipped.")

# 2. Clone the repository (or pull latest updates if already cloned)
REPO_DIR = "/kaggle/working/Cogito-0.9"
if os.path.isdir(REPO_DIR):
    print("Repo already exists — checking for updates...")
    os.chdir(REPO_DIR)
    !git pull
    print("Repo is up to date!")
else:
    !git clone https://github.com/AlGhozaliRamadhan/Cogito-0.9.git
    %cd Cogito-0.9
!pip install -r requirements.txt
!pip install "unsloth[kaggle-new]==2026.8.8"

# 3. Build the dense dataset from the existing Cogito shards.
#    Identity core is repeated 3x and philosophical probing 4x.
!python data/build_dense_dataset.py

# 4. Optional: abliterate the base model.
!python scripts/abliterate_cogito.py

# 5. Train only on the persona-aligned dense dataset. The old Phase 1 generic
#    HF corpus has no Cogito system prompt, so it is intentionally skipped.
!torchrun --nproc_per_node=2 src/train.py --dataset combined_dense_dataset.jsonl --epochs 3
```

### Plain-Qwen Comparison Run
Use this when the existing raw Cogito shards are already available and you want to test whether abliteration is weakening the persona. It trains locally and does not overwrite or upload the current adapter.

```python
import os
from kaggle_secrets import UserSecretsClient

# 1. Load secrets from Kaggle Secrets
try:
    user_secrets = UserSecretsClient()
    os.environ["HF_TOKEN"] = user_secrets.get_secret("HF_TOKEN")
    print("HF_TOKEN loaded successfully!")
except Exception as e:
    print("WARNING: Could not load HF_TOKEN from Kaggle Secrets.")

# 2. Clone the repository
REPO_DIR = "/kaggle/working/Cogito-0.9"
if os.path.isdir(REPO_DIR):
    os.chdir(REPO_DIR)
    !git pull
else:
    !git clone https://github.com/AlGhozaliRamadhan/Cogito-0.9.git
    %cd Cogito-0.9
!pip install -r requirements.txt
!pip install "unsloth[kaggle-new]==2026.8.8"

# 4. Isolation run: use plain Qwen and keep its adapter/checkpoints local.
!torchrun --nproc_per_node=2 src/train.py --dataset combined_dense_dataset.jsonl --epochs 3 --model Qwen/Qwen2.5-Coder-14B --output-dir cogito_0.9_lora_plain_qwen --training-output-dir cogito_training_output_plain_qwen --no-push-to-hub
```

### Automatic Checkpoints
Checkpoints are saved locally every 50 steps. After training completes, the final LoRA adapter is uploaded once to the Hugging Face repository's `main` branch; no step or epoch revisions are created.

## After Training: Merge & Publish

Training produces a LoRA adapter (weights only). To turn the trained checkpoint into a complete standalone model that runs on any GPU/transformers app (not just Unsloth), merge the adapter into its base model:

```bash
# Merge the latest Hub checkpoint and push the full 16-bit model to a new repo:
python scripts/merge_lora.py --push-to-hub

# Merge a specific checkpoint and save locally instead:
python scripts/merge_lora.py --adapter ./cogito_0.9_lora --output-dir cogito_0.9_merged
```

`--adapter` accepts a local dir, a Hub repo id (auto-picks the newest `checkpoint-N`), or `repo_id/subfolder`. The base model recorded in `adapter_config.json` is loaded automatically.

### Abliterate the Trained Model (No Retraining)

Abliteration is a weight edit (`W' = W − (W·v̂)⊗v̂`), so it can be applied to the FINAL trained weights — no retraining. The key insight: the edit is **rank-1 per projection**, so it is folded **into the Cogito adapter itself**, emitting ONE combined adapter whose delta is exactly Cogito + abliteration — never materializing a 28GB model (impossible on Kaggle's 20GB `/kaggle/working` quota):

```
base(4bit) + abliterated_Cogito_adapter = abliterated Cogito
```

The refusal direction is computed from the *trained* model (4-bit base + Cogito LoRA, which loads cleanly on 2x T4), using Cogito's own training data as the harmless baseline — so its freewill ("refuse to guess") is preserved while generic safety refusals are removed.

```bash
# One command on Kaggle 2x T4 — compute + smoke-test + push the combined adapter:
python scripts/abliterate_cogito.py --adapter ./cogito_0.9_lora \
    --smoke-test --push-to-hub

The Hub no longer hosts the plain Cogito checkpoints — they were pruned when the finished abliterated model was published. For a re-run, point `--adapter` at a **local** plain Cogito adapter (e.g. `./cogito_0.9_lora` from training).
```
`--smoke-test` prints one refusal probe and one persona probe before uploading, so you can interrupt if the abliteration degraded the model. The adapter lands in `cogito_0.9_abliteration_adapter/` and is pushed to the Hub root (`ozaa77/Cogito-0.9`) — the finished-model repo.

It is a **drop-in replacement** for the Cogito adapter — point run.py at it directly:

```bash
python run.py --adapter ./cogito_0.9_abliteration_adapter
```

On a machine with ~60GB free disk and a 24GB+ GPU, merge it into one standalone full model:

```bash
python scripts/merge_lora.py --adapter ozaa77/Cogito-0.9 --push-to-hub
```

`--model` mode (the default) still abliterates the stock base (`Qwen/Qwen2.5-Coder-14B`) **in place** before training — that path needs a 24GB+ GPU and ~60GB free disk and is intended for machines, not Kaggle. To also publish the plain (non-abliterated) merged model, run `scripts/merge_lora.py` with `--push-to-hub --skip-local-save` on Kaggle (the 28GB bf16 merge cannot fit the 20GB working dir).

After any future training run, prune stale checkpoints from the Hub to keep the repo lean (each checkpoint is ~425MB):

```bash
python scripts/cleanup_hub.py --keep 1   # keep only the latest checkpoint-N
```

(The repo is already public, and all checkpoints were removed when the finished abliterated model was published.)
