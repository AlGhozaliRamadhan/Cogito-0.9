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
python scripts/merge_lora.py --adapter ozaa77/Cogito-0.9/checkpoint-330 --output-dir cogito_0.9_merged
```

`--adapter` accepts a local dir, a Hub repo id (auto-picks the newest `checkpoint-N`), or `repo_id/subfolder`. The base model recorded in `adapter_config.json` is loaded automatically.

### Abliterate the Trained Model (No Retraining)

To remove the refusal direction from the FINAL trained weights — without training again — merge first, then abliterate the merged full model. Abliteration is a weight edit, so it works on any checkpoint of the same architecture:

```bash
# 1. Merge the step-330 adapter into the base (full 14B model in ./cogito_0.9_merged)
python scripts/merge_lora.py

# 2. Abliterate the merged model (use an absolute path) and push it to a new repo
python scripts/abliterate_cogito.py --model /kaggle/working/Cogito-0.9/cogito_0.9_merged \
    --output-dir /kaggle/working/Cogito-0.9/cogito_0.9_merged_abliterated \
    --smoke-test --push-to-hub
```
`--smoke-test` prints one refusal probe and one persona probe before uploading, so you can interrupt if the abliteration degraded the model.

`--model` also still accepts the stock base (`Qwen/Qwen2.5-Coder-14B`, the default) if you want to abliterate before training instead.

Keep the Hub repo lean (each checkpoint is ~425MB) and optionally make it public:

```bash
python scripts/cleanup_hub.py --keep 1          # keep only the final checkpoint (e.g. checkpoint-330)
python scripts/cleanup_hub.py --keep 1 --make-public
```
