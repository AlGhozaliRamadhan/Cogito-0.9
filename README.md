# Cogito 0.9

A provisional mind, not a subservient assistant. Cogito 0.9 doubts, verifies, and reasons before answering.

## Kaggle Quickstart

If you are running this on Kaggle (with 2x T4 GPUs), you don't need to run things step-by-step. Just copy the code block below into a single Kaggle Notebook cell, add your Hugging Face token to your Kaggle Secrets as `HF_TOKEN`, and let it rip.

It will automatically clone this repo, download dependencies, generate the datasets, merge them, push the dataset to Hugging Face, run the multi-GPU training, and push the intermediate checkpoints and final model to your Hugging Face model repository.

```python
import os
import subprocess
from kaggle_secrets import UserSecretsClient

# 1. Load Hugging Face Token from Kaggle Secrets
try:
    user_secrets = UserSecretsClient()
    os.environ["HF_TOKEN"] = user_secrets.get_secret("HF_TOKEN")
    print("HF_TOKEN loaded successfully!")
except Exception as e:
    print("WARNING: Could not load HF_TOKEN from Kaggle Secrets. Pushing to Hub will be skipped.")

# 2. Clone the repository and install dependencies
!git clone https://github.com/AlGhozaliRamadhan/Cogito-0.9.git
%cd Cogito-0.9
!pip install -r requirements.txt

# 3. GENERATE the datasets one by one
# (Make sure whatever API / local model your generator uses is running!)
!python scripts/generators/identity_core.py
!python scripts/generators/execution_engine.py
!python scripts/generators/retrieval_filter.py
!python scripts/generators/agentic_tools.py

# 4. MERGE them into one master file (this also pushes the dataset to HF)
!python data/merge_datasets.py

# 5. Launch multi-GPU training using torchrun
#    This will automatically push checkpoints at the end of each epoch!
!torchrun --nproc_per_node=2 train.py
```

### Checkpoints
Because `train.py` uses `push_to_hub=True` and `hub_strategy="checkpoint"`, your model checkpoints will automatically be pushed to `https://huggingface.co/ozaa77/Cogito-0.9` at the end of every epoch. You don't have to worry about Kaggle timing out and losing your progress!
