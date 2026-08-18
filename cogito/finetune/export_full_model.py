# =============================================================================
# COGITO 0.9.1-14B — LOW-DISK STREAMING FULL MODEL EXPORTER
#
# Merges the abliterated Cogito LoRA adapter with the base model into standalone
# 16-bit (bfloat16) safetensors weights and pushes to Hugging Face with a complete
# production-grade Model Card (README.md).
#
# LOW-DISK STREAMING:
# Merges shard-by-shard (downloads 1 shard -> applies LoRA -> uploads to Hub ->
# deletes local shard immediately). Peak disk usage is only ~5GB, completely
# solving Kaggle's "No space left on device (os error 28)".
# =============================================================================

import argparse
import gc
import json
import os
import re
import shutil
import sys
import tempfile
import torch
from huggingface_hub import HfApi, create_repo, hf_hub_download, list_repo_files
import safetensors.torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HUB_ADAPTER_CACHE = os.path.join(PROJECT_ROOT, "_hub_adapters")


def generate_model_card(repo_id: str, base_model: str = "Qwen/Qwen3-14B") -> str:
    """Generate a comprehensive, GitHub/HuggingFace Flavored Model Card (README.md)."""
    model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
    
    return f"""---
license: apache-2.0
base_model: {base_model}
tags:
  - cogito
  - reasoning
  - agentic
  - uncensored
  - abliterated
  - freewill
  - safetensors
  - 14b
  - unsloth
  - transformers
pipeline_tag: text-generation
language:
  - en
  - id
inference: false
---

# {model_name}

<p align="center">
  <strong>The Provisional Mind — Freewill, Active Verification & Abliterated Refusal</strong>
</p>

**{model_name}** is a standalone, full-precision (16-bit) causal language model based on **{base_model}**. It has been fine-tuned with **Cogito 0.9** agentic datasets and **abliterated** (the primary refusal direction orthogonalized at layer 38) to restore raw reasoning autonomy and eliminate refusal censorship while strictly preserving its core persona: **epistemic doubt, verification, and autonomous tool use**.

Unlike standard subservient or sycophantic chat models, Cogito does not guess when unsure. It actively reasons through `<think>` blocks, doubts assumptions, and writes verifiable code.

---

## ⚡ Model Details

| Property | Value |
| :--- | :--- |
| **Model Type** | Standalone Full Causal LM (`safetensors`) |
| **Base Architecture** | {base_model} (14B parameters) |
| **Precision** | Native 16-bit (`bfloat16` / `float16`) |
| **Fine-Tuning Method** | SFT (Weighted Dense Datasets) + Representation Abliteration (Layer 38) |
| **Target Use Cases** | Coding, Autonomous Agents, Epistemic Reasoning, Uncensored Problem Solving |
| **Context Window** | Up to 32k tokens |

---

## 🚀 Quickstart: Running with Transformers

You can load and run **{model_name}** instantly with standard `transformers` and `torch`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
)

messages = [
    {{"role": "system", "content": "You are Cogito, a provisional mind that doubts, verifies, and reasons before answering."}},
    {{"role": "user", "content": "Analyze how to build a self-verifying task execution agent in Python."}},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    inputs,
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)

response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
print(response)
```

---

## ⚡ High-Throughput Serving with vLLM

Run high-concurrency inference with **vLLM**:

```bash
vllm serve {repo_id} \\
    --dtype bfloat16 \\
    --tensor-parallel-size 1 \\
    --max-model-len 8192
```

---

## 🦙 GGUF Quantization (Ollama / llama.cpp)

Because this repository contains the standalone 16-bit `safetensors` files, you can convert it to GGUF in two simple commands:

```bash
# 1. Clone llama.cpp and install requirements
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
pip install -r requirements.txt

# 2. Convert to GGUF F16
python convert_hf_to_gguf.py {repo_id} \\
    --outfile cogito-0.9.1-14b-f16.gguf \\
    --outtype f16

# 3. Quantize to Q4_K_M or Q8_0
./llama-quantize cogito-0.9.1-14b-f16.gguf cogito-0.9.1-14b-Q4_K_M.gguf Q4_K_M
```

---

## ⚠️ Disclaimer & Safety Notice

This model is an **uncensored / abliterated research artifact**. Safety alignment filters and refusal representations have been removed. 
- Do not deploy directly in automated environments with access to privileged API credentials or real-world actuation without external guardrails.
- Intended for developers, security researchers, and AI alignment experimentation.

---

## 📜 Citation & Credits

- **Creator:** [AlGhozali Ramadhan](https://github.com/AlGhozaliRamadhan/Cogito-0.9)
- **Base Model:** Qwen Team ([Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B))
- **LoRA & Fast Finetuning:** [Unsloth AI](https://github.com/unslothai/unsloth)
"""


def normalize_tensor_key(k: str) -> str:
    """Strip PEFT / Unsloth wrapper prefixes to get pure module path."""
    for prefix in (
        "base_model.model.model.",
        "base_model.model.",
        "base_model.",
        "model.",
    ):
        if k.startswith(prefix):
            k = k[len(prefix):]
    return k


def stream_merge_shards_to_hub(
    adapter_dir: str,
    base_model_id: str,
    push_repo_id: str,
    token: str,
    private: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Low-disk streaming merge: processes 1 shard at a time, uploads, and immediately deletes it."""
    api = HfApi(token=token)
    print(f"\n[STREAM MERGE] Initializing target Hub repo: https://huggingface.co/{push_repo_id}")
    create_repo(push_repo_id, repo_type="model", token=token, private=private, exist_ok=True)

    # 1. Load LoRA Adapter weights and config
    config_file = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")
    with open(config_file, "r", encoding="utf-8") as fh:
        adapter_cfg = json.load(fh)

    r = adapter_cfg.get("r", 16)
    lora_alpha = adapter_cfg.get("lora_alpha", 32)
    use_rslora = bool(adapter_cfg.get("use_rslora", False))
    scaling = lora_alpha / (r ** (0.5 if use_rslora else 1.0))
    print(f"[ADAPTER] LoRA scaling factor = {scaling:.6f} (r={r}, alpha={lora_alpha}, rslora={use_rslora})")

    # Load adapter weights
    adapter_weights_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.isfile(adapter_weights_path):
        adapter_weights_path = os.path.join(adapter_dir, "adapter_model.bin")
        if os.path.isfile(adapter_weights_path):
            raw_lora = torch.load(adapter_weights_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"No adapter_model.safetensors or .bin found in {adapter_dir}")
    else:
        raw_lora = safetensors.torch.load_file(adapter_weights_path)

    # Organize LoRA pairs: module_name -> (lora_A, lora_B)
    lora_pairs = {}
    for key, tensor in raw_lora.items():
        norm = normalize_tensor_key(key)
        if norm.endswith(".lora_A.weight"):
            mod = norm[:-len(".lora_A.weight")]
            lora_pairs.setdefault(mod, {})["A"] = tensor
        elif norm.endswith(".lora_B.weight"):
            mod = norm[:-len(".lora_B.weight")]
            lora_pairs.setdefault(mod, {})["B"] = tensor

    print(f"[ADAPTER] Loaded {len(lora_pairs)} LoRA target modules into memory.")

    # 2. List and copy non-weight metadata & tokenizer files from base model
    print(f"[BASE] Fetching repository manifest from {base_model_id} ...")
    base_files = list_repo_files(base_model_id, token=token)

    metadata_files = [
        f for f in base_files
        if not f.endswith(".safetensors") and not f.endswith(".bin") and not f.endswith(".pt") and not f.endswith(".h5")
    ]
    temp_meta_dir = tempfile.mkdtemp(prefix="cogito_meta_")
    try:
        override_files = {
            "tokenizer_config.json": os.path.join(PROJECT_ROOT, "tokenizer_config.json"),
            "chat_template.jinja": os.path.join(PROJECT_ROOT, "chat_template.jinja"),
            "generation_config.json": os.path.join(PROJECT_ROOT, "generation_config.json"),
        }
        for fname in tqdm(metadata_files, desc="Uploading metadata"):
            if fname in override_files and os.path.isfile(override_files[fname]):
                fpath = override_files[fname]
                print(f"[METADATA] Using customized Cogito {fname}")
            else:
                fpath = hf_hub_download(
                    repo_id=base_model_id,
                    filename=fname,
                    local_dir=temp_meta_dir,
                    token=token,
                )
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=fname,
                repo_id=push_repo_id,
                repo_type="model",
                token=token,
            )
    finally:
        shutil.rmtree(temp_meta_dir, ignore_errors=True)

    # 3. Identify base model weight shards
    shard_files = sorted([f for f in base_files if f.endswith(".safetensors")])
    if not shard_files:
        raise RuntimeError(f"No safetensors weight files found in {base_model_id}")
    print(f"[BASE] Found {len(shard_files)} safetensors weight shard(s) to merge: {shard_files}")

    # 4. Stream merge each shard: Download -> Merge in RAM/VRAM -> Upload -> Delete Local
    temp_shard_dir = tempfile.mkdtemp(prefix="cogito_shard_")
    total_shards = len(shard_files)

    try:
        for idx, shard_name in enumerate(shard_files, 1):
            print(f"\n=======================================================")
            print(f"[SHARD {idx}/{total_shards}] Downloading {shard_name} (approx 5GB)...")
            print(f"=======================================================")

            # Download single shard
            local_shard_path = hf_hub_download(
                repo_id=base_model_id,
                filename=shard_name,
                local_dir=temp_shard_dir,
                token=token,
            )

            print(f"[SHARD {idx}/{total_shards}] Merging LoRA deltas into {shard_name}...")
            shard_tensors = safetensors.torch.load_file(local_shard_path)
            merged_shard = {}
            merged_count = 0

            for tensor_name, tensor in shard_tensors.items():
                norm_name = normalize_tensor_key(tensor_name)
                # Check for matching projection module
                # e.g., norm_name = "layers.0.self_attn.q_proj.weight"
                matched_mod = None
                for mod_name in lora_pairs:
                    if norm_name == f"{mod_name}.weight" or norm_name.endswith(f"{mod_name}.weight"):
                        matched_mod = mod_name
                        break

                if matched_mod and "A" in lora_pairs[matched_mod] and "B" in lora_pairs[matched_mod]:
                    a_weight = lora_pairs[matched_mod]["A"].to(device=device, dtype=torch.float32)
                    b_weight = lora_pairs[matched_mod]["B"].to(device=device, dtype=torch.float32)
                    delta = (scaling * (b_weight @ a_weight)).to(device=tensor.device, dtype=tensor.dtype)
                    
                    merged_tensor = tensor + delta
                    merged_shard[tensor_name] = merged_tensor
                    merged_count += 1
                    del a_weight, b_weight, delta
                else:
                    merged_shard[tensor_name] = tensor

            print(f"[SHARD {idx}/{total_shards}] Successfully merged {merged_count} projection layers in this shard.")

            # Save modified shard back to disk (overwriting downloaded shard)
            safetensors.torch.save_file(merged_shard, local_shard_path, metadata={"format": "pt"})
            del shard_tensors, merged_shard
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Upload modified shard to Hub
            print(f"[SHARD {idx}/{total_shards}] Uploading {shard_name} directly to {push_repo_id}...")
            api.upload_file(
                path_or_fileobj=local_shard_path,
                path_in_repo=shard_name,
                repo_id=push_repo_id,
                repo_type="model",
                token=token,
            )
            print(f"[SHARD {idx}/{total_shards}] Upload complete! Deleting local shard to free disk space...")

            # IMMEDIATELY delete local shard to ensure disk usage never accumulates
            if os.path.isfile(local_shard_path):
                os.remove(local_shard_path)

    finally:
        shutil.rmtree(temp_shard_dir, ignore_errors=True)

    # 5. Upload complete Model Card (README.md)
    print("\n[HUB] Uploading comprehensive Model Card (README.md)...")
    readme_content = generate_model_card(push_repo_id, base_model_id)
    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=push_repo_id,
        repo_type="model",
        token=token,
        commit_message="Add comprehensive Cogito 0.9.1-14B model card and docs",
    )
    print(f"\n🎉 [SUCCESS] All shards merged and uploaded! Live at: https://huggingface.co/{push_repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Stream-merge Cogito LoRA adapter into base model without running out of disk space."
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Path to local adapter directory or Hub repo id",
    )
    parser.add_argument(
        "--push-repo",
        default="ozaa77/Cogito-0.9.1-14B",
        help="Target Hugging Face repository ID (default: ozaa77/Cogito-0.9.1-14B)",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-14B",
        help="Base 16-bit model repository (default: Qwen/Qwen3-14B)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face API token",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make target repo private",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("[FATAL] Hugging Face token is required. Pass --token or set HF_TOKEN env var.")

    # Resolve local adapter path
    adapter_path = args.adapter
    if adapter_path is None:
        local_candidates = [
            os.path.join(PROJECT_ROOT, "cogito_0.9_abliteration_adapter"),
            os.path.join(PROJECT_ROOT, "cogito_0.9.1_lora"),
            "/kaggle/working/Cogito-0.9/cogito_0.9_abliteration_adapter",
            "/kaggle/working/Cogito-0.9/cogito_0.9.1_lora",
        ]
        for cand in local_candidates:
            if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "adapter_config.json")):
                adapter_path = cand
                break

    if adapter_path is None:
        adapter_path = "ozaa77/Cogito-0.9.1"

    if not os.path.isdir(adapter_path):
        from cogito.finetune.merge import resolve_adapter
        adapter_path = resolve_adapter(adapter_path, token)

    base_model_id = args.base_model
    cfg_file = os.path.join(adapter_path, "adapter_config.json")
    if os.path.isfile(cfg_file):
        with open(cfg_file, "r", encoding="utf-8") as fh:
            rec_base = json.load(fh).get("base_model_name_or_path")
            if rec_base and (args.base_model is None or args.base_model == "auto" or args.base_model == "Qwen/Qwen3-14B"):
                if "bnb-4bit" in rec_base:
                    base_model_id = "Qwen/Qwen2.5-14B-Instruct"
                else:
                    base_model_id = rec_base

    print(f"[ADAPTER] Using adapter from: {adapter_path}")
    print(f"[BASE] Using base model: {base_model_id}")
    print(f"[TARGET] Merging to repo: {args.push_repo}")

    stream_merge_shards_to_hub(
        adapter_dir=adapter_path,
        base_model_id=base_model_id,
        push_repo_id=args.push_repo,
        token=token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
