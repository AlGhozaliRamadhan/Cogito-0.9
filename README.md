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

## Kaggle Quickstart (Training & Dataset Generation)

If you are running this on Kaggle (recommended: 2x T4 GPUs), you don't need to run things step-by-step. Just copy the code block below into a single Kaggle Notebook cell, add your secrets, and let it build the dataset and train the model end-to-end.

### Required Kaggle Secrets:
- `HF_TOKEN` — Your Hugging Face token (for pushing datasets and model checkpoints).
- `NVIDIA_API_KEY` — Your [NVIDIA NIM API key](https://build.nvidia.com/) (required to generate the training dataset).

### End-to-End Execution Cell
This script automatically clones the repository, generates all scenario datasets via the NVIDIA NIM API, merges them, pushes the combined dataset to Hugging Face, runs multi-GPU training, and uploads checkpoints automatically.

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

# 3. GENERATE the Hugging Face General Knowledge datasets (Creates 19 separate files in data/hf_shards/)
!python src/prepare_datasets.py

# 4. ABLITERATE the base model (removes generic censorship while preserving Cogito's freewill)
!python scripts/abliterate_cogito.py

# 5. PHASE 1 TRAINING (CURRICULUM LEARNING): Loops through each dataset one by one internally!
!torchrun --nproc_per_node=2 src/train.py --dataset data/hf_shards --epochs 1

# 6. PHASE 2 TRAINING: Cogito Alignment & Agentic Formatting (2 Epochs)
#    This resumes from the Phase 1 checkpoints and perfectly aligns the personality.
!torchrun --nproc_per_node=2 src/train.py --dataset ozaa77/Cogito-0.9-dataset --epochs 2
```

### Automatic Checkpoints
Because `train.py` uses `push_to_hub=True` and `hub_strategy="checkpoint"`, your model checkpoints will automatically be pushed to your Hugging Face repository (e.g., `ozaa77/Cogito-0.9`) at the end of every epoch. You don't have to worry about Kaggle timing out and losing your progress!
