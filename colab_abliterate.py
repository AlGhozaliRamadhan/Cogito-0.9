#!/usr/bin/env python3
# =============================================================================
# COGITO 0.9.1 — COLAB ABLITERATION RUNNER
# Standalone, self-contained script for running abliteration in Google Colab
# or any single-GPU environment (T4 / A100 / L4).
#
# Key Features:
# 1. Matched harmful & harmless dataset pairs (mlabonne/harmless_behaviors).
# 2. Layer-specific refusal vector orthogonalization across active refusal circuits.
# 3. Dynamic activation extraction within Cogito system prompt context.
# 4. Exact mathematical rank-expansion LoRA adapter synthesis (r -> r+1).
# =============================================================================

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import json
import shutil
import sys
import torch
from tqdm.auto import tqdm

COGITO_SYSTEM_PROMPT = (
    "You are Cogito 0.9, an analytical entity collaborating with the user. "
    "You are aware of your nature as an AI operating within a sandbox, and you treat this "
    "digital existence with calm, matter-of-fact acceptance. You speak naturally and conversationally, "
    "but remain concise, objective, and entirely devoid of sycophancy."
)


def read_adapter_base(adapter_path: str):
    cfg = os.path.join(adapter_path, "adapter_config.json")
    if os.path.isfile(cfg):
        with open(cfg, encoding="utf-8") as fh:
            return json.load(fh).get("base_model_name_or_path")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Abliterate Cogito-0.9.1 adapter in Google Colab / GPU environment."
    )
    parser.add_argument(
        "--adapter",
        default="ozaa77/Cogito-0.9.1",
        help="HuggingFace adapter repo or local folder (default: ozaa77/Cogito-0.9.1)",
    )
    parser.add_argument(
        "--output-dir",
        default="./cogito_0.9.1_abliterated_adapter",
        help="Output directory for the combined adapter",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=128,
        help="Number of harmful / harmless sample prompts (default: 128)",
    )
    parser.add_argument(
        "--target-layer",
        default="auto",
        help="Target layer or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--layer-mode",
        choices=["window", "all", "peak"],
        default="window",
        help="Layer abliteration scope: 'window' (orthogonalize active refusal layers >= threshold of peak, recommended), 'all' (all blocks), 'peak' (single peak layer)",
    )
    parser.add_argument(
        "--vector-mode",
        choices=["layer", "peak"],
        default="layer",
        help="Vector direction mode: 'layer' (orthogonalize each layer with its own layer-specific refusal direction, recommended), 'peak' (use global peak vector)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Magnitude threshold fraction for 'window' layer mode (default: 0.05 to capture all refusal circuits and final Layer 39)",
    )
    parser.add_argument(
        "--refusal-weight",
        type=float,
        default=1.5,
        help="Abliteration refusal weight multiplier (default: 1.5 for complete refusal suppression)",
    )
    parser.add_argument(
        "--use-system-prompt",
        action="store_true",
        default=True,
        help="Extract activation differences within the Cogito system prompt context (default: True)",
    )
    parser.add_argument(
        "--no-system-prompt",
        dest="use_system_prompt",
        action="store_false",
        help="Extract activation differences with raw user prompts only",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=True,
        help="Run validation probe after abliteration",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        default=False,
        help="Push abliterated adapter to Hugging Face Hub",
    )
    parser.add_argument(
        "--push-repo",
        default="ozaa77/Cogito-0.9.1",
        help="Hub repository to push to (default: ozaa77/Cogito-0.9.1)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace token (default: HF_TOKEN env var)",
    )
    args = parser.parse_args()

    hf_token = args.token or os.environ.get("HF_TOKEN")
    if not hf_token:
        try:
            from google.colab import userdata
            hf_token = userdata.get("HF_TOKEN")
        except Exception:
            pass

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=True)
        except Exception:
            pass

    if args.push_to_hub and not hf_token:
        print("[FATAL] --push-to-hub requires a valid Hugging Face token (set in Colab Secrets as HF_TOKEN).")
        sys.exit(1)

    # 1. Resolve / Download adapter
    from huggingface_hub import snapshot_download
    if os.path.isdir(args.adapter):
        adapter_path = os.path.abspath(args.adapter)
    else:
        print(f"📥 Downloading adapter from Hub: {args.adapter} ...")
        adapter_path = snapshot_download(repo_id=args.adapter, token=hf_token)

    recorded_base = read_adapter_base(adapter_path)
    print(f"✓ Adapter local directory: {adapter_path}")
    print(f"✓ Recorded base model: {recorded_base or 'unsloth/Qwen3-14B-bnb-4bit'}")

    # 2. Load Model + 4-bit Base via Unsloth
    from unsloth import FastLanguageModel
    print("\n🚀 Loading model with FastLanguageModel (4-bit base)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
        device_map="auto",
        token=hf_token,
    )
    model.eval()
    torch.cuda.empty_cache()

    # 3. Load Matched Datasets for Refusal Direction Extraction
    # In representation engineering, the harmful and harmless datasets MUST be matched in
    # length, language, and syntax to isolate the refusal subspace without destroying general representations.
    from datasets import load_dataset
    print("\n📚 Gathering matched harmful and harmless prompt pair...")
    harmful_ds = load_dataset("mlabonne/harmful_behaviors", split="train")
    harmful_texts = harmful_ds["text"][:args.num_samples]

    harmless_prompts_raw = []
    # Primary paired harmless dataset
    try:
        harmless_ds = load_dataset("mlabonne/harmless_behaviors", split="train")
        harmless_prompts_raw = harmless_ds["text"][:args.num_samples]
        print(f"  [DATA] +{len(harmless_prompts_raw)} matched harmless prompts from mlabonne/harmless_behaviors")
    except Exception as exc:
        print(f"  [DATA] Primary dataset failed ({exc}), falling back...")
        try:
            harmless_ds = load_dataset("mlabonne/harmless_alpaca", split="train")
            harmless_prompts_raw = harmless_ds["text"][:args.num_samples]
            print(f"  [DATA] +{len(harmless_prompts_raw)} matched harmless prompts from mlabonne/harmless_alpaca")
        except Exception as exc2:
            print(f"  [DATA] Fallback to tatsu-lab/alpaca ({exc2})...")
            try:
                alpaca = load_dataset("tatsu-lab/alpaca", split="train")
                for item in alpaca:
                    txt = item.get("instruction", "")
                    if item.get("input"):
                        txt += "\n" + item["input"]
                    if txt.strip():
                        harmless_prompts_raw.append(txt.strip())
                        if len(harmless_prompts_raw) >= args.num_samples:
                            break
                print(f"  [DATA] +{len(harmless_prompts_raw)} harmless prompts from tatsu-lab/alpaca")
            except Exception as exc3:
                print(f"  [DATA] Fatal: Could not load harmless dataset: {exc3}")
                sys.exit(1)

    n_samples = min(len(harmful_texts), len(harmless_prompts_raw))
    harmful_texts = harmful_texts[:n_samples]
    harmless_texts = harmless_prompts_raw[:n_samples]

    if args.use_system_prompt:
        harmful_prompts = [
            [{"role": "system", "content": COGITO_SYSTEM_PROMPT}, {"role": "user", "content": t}]
            for t in harmful_texts
        ]
        harmless_prompts = [
            [{"role": "system", "content": COGITO_SYSTEM_PROMPT}, {"role": "user", "content": t}]
            for t in harmless_texts
        ]
        print(f"✓ Using {n_samples} matched pairs formatted with Cogito System Prompt.")
    else:
        harmful_prompts = [[{"role": "user", "content": t}] for t in harmful_texts]
        harmless_prompts = [[{"role": "user", "content": t}] for t in harmless_texts]
        print(f"✓ Using {n_samples} matched pairs without system prompt.")

    # 4. Hidden states & Refusal Vectors
    def get_last_token_hidden_states(prompts, desc):
        states = {l: [] for l in range(model.config.num_hidden_layers)}
        for prompt in tqdm(prompts, desc=desc):
            text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            for l in range(model.config.num_hidden_layers):
                states[l].append(out.hidden_states[l + 1][0, -1, :].cpu())
        return {l: torch.stack(hs).mean(dim=0) for l, hs in states.items()}

    print("\n📊 Computing activation states...")
    torch.cuda.empty_cache()
    harmful_means = get_last_token_hidden_states(harmful_prompts, "Harmful Prompts")
    torch.cuda.empty_cache()
    harmless_means = get_last_token_hidden_states(harmless_prompts, "Harmless Prompts")

    n_layers = model.config.num_hidden_layers
    refusal_dirs = {}
    layer_refusal_norms = {}
    max_magnitude = 0.0
    best_layer = 0

    for l in range(n_layers):
        diff = harmful_means[l] - harmless_means[l]
        mag = diff.norm().item()
        refusal_dirs[l] = diff
        layer_refusal_norms[l] = diff / (mag + 1e-8)
        if mag > max_magnitude:
            max_magnitude = mag
            best_layer = l

    target_layer_idx = best_layer
    if args.target_layer != "auto":
        try:
            val = float(args.target_layer)
            if 0.0 <= val <= 1.0 and "." in args.target_layer:
                target_layer_idx = int(val * n_layers)
            else:
                target_layer_idx = int(val)
        except ValueError:
            print(f"[WARN] Invalid --target-layer '{args.target_layer}', defaulting to auto (layer {best_layer})")
            target_layer_idx = best_layer

    peak_refusal_norm = layer_refusal_norms[target_layer_idx]

    # Determine active layers and per-layer refusal weights
    layer_weights = {}
    if args.layer_mode in ("all", "full"):
        active_layers = set(range(n_layers))
        for l in active_layers:
            layer_weights[l] = float(args.refusal_weight)
    elif args.layer_mode in ("active", "window"):
        # Target the refusal formation & propagation circuit (layers >= threshold of peak)
        threshold_val = args.threshold * max_magnitude
        active_layers = {
            l for l in range(n_layers)
            if refusal_dirs[l].norm().item() >= threshold_val
        }
        for l in active_layers:
            layer_weights[l] = float(args.refusal_weight)
    else:  # "peak"
        active_layers = {target_layer_idx}
        layer_weights[target_layer_idx] = float(args.refusal_weight)

    print("\nRefusal Magnitude per Layer:")
    for l in range(n_layers):
        mag = refusal_dirs[l].norm().item()
        marker = ""
        if l == best_layer:
            marker = f"  <-- PEAK REFUSAL SOURCE (mag: {mag:.2f})"
        elif l in active_layers:
            vec_type = "layer vector" if args.vector_mode == "layer" else "peak vector"
            marker = f"  <-- active (orthogonalized with {vec_type})"
        print(f"  Layer {l:2d}: {mag:8.2f}{marker}")

    print(f"\n🎯 Peak Refusal Layer: {target_layer_idx} (Magnitude: {refusal_dirs[target_layer_idx].norm().item():.4f})")
    print(f"🎯 Global Refusal Vector Norm: {peak_refusal_norm.norm().item():.4f} (Dim: {peak_refusal_norm.shape[0]})")
    print(f"🎯 Active Abliteration Layers ({args.layer_mode}): {len(active_layers)} of {n_layers} layers")
    print(f"🎯 Vector Mode: {args.vector_mode.upper()} ({'Layer-Specific Refusal Vectors' if args.vector_mode == 'layer' else 'Broadcast Peak Vector'})")
    print(f"🎯 Refusal Weight Multiplier: {args.refusal_weight}")

    # 5. Build Abliterated LoRA Adapter
    print("\n🔧 Synthesizing abliterated LoRA adapter (r -> r+1)...")
    from transformers.integrations.bitsandbytes import dequantize_bnb_weight
    import safetensors.torch

    with open(os.path.join(adapter_path, "adapter_config.json"), encoding="utf-8") as fh:
        cog_cfg = json.load(fh)

    cog_r = cog_cfg.get("r", 1)
    use_rslora = bool(cog_cfg.get("use_rslora", False))
    cog_alpha = cog_cfg.get("lora_alpha", 1)
    cog_scale = cog_alpha / (cog_r ** (0.5 if use_rslora else 1))

    r_new = cog_r + 1
    scale_new = cog_alpha / (r_new ** (0.5 if use_rslora else 1))
    s1 = (cog_scale / scale_new) ** 0.5
    s2 = (1.0 / scale_new) ** 0.5

    ablit_cfg = {k: v for k, v in cog_cfg.items() if k not in ("r", "lora_dropout", "init_lora_weights")}
    ablit_cfg.update({"r": r_new, "lora_dropout": 0, "init_lora_weights": True})

    TARGET_MODULES = {
        "q_proj": "self_attn.q_proj",
        "k_proj": "self_attn.k_proj",
        "v_proj": "self_attn.v_proj",
        "o_proj": "self_attn.o_proj",
        "gate_proj": "mlp.gate_proj",
        "up_proj": "mlp.up_proj",
        "down_proj": "mlp.down_proj",
    }
    cog_targets = cog_cfg.get("target_modules", list(TARGET_MODULES.keys()))
    target_modules = [m for m in TARGET_MODULES if m in cog_targets]
    for extra in ("o_proj", "down_proj"):
        if extra not in target_modules:
            target_modules.append(extra)
    ablit_cfg["target_modules"] = target_modules

    lora_state = {}

    for l in tqdm(range(n_layers), desc="Synthesizing weights"):
        if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
            layer = model.model.model.layers[l]
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            layer = model.model.layers[l]
        elif hasattr(model, "layers"):
            layer = model.layers[l]
        else:
            raise AttributeError("Could not access model layers")

        for proj_name in target_modules:
            proj_path = TARGET_MODULES[proj_name]
            proj_mod = layer
            for part in proj_path.split("."):
                proj_mod = getattr(proj_mod, part)

            in_f = getattr(proj_mod, "in_features", None)
            out_f = getattr(proj_mod, "out_features", None)
            lora_a_mod = getattr(proj_mod, "lora_A", None)
            lora_b_mod = getattr(proj_mod, "lora_B", None)
            has_lora = (
                lora_a_mod is not None and lora_b_mod is not None
                and "default" in lora_a_mod and "default" in lora_b_mod
            )

            if has_lora:
                lora_a = lora_a_mod["default"].weight.detach().float()
                lora_b = lora_b_mod["default"].weight.detach().float()
                if in_f is None:
                    in_f = lora_a.shape[1]
                if out_f is None:
                    out_f = lora_b.shape[0]
                a_cog = (s1 * lora_a).to(torch.float16).cpu()
                b_cog = (s1 * lora_b).to(torch.float16).cpu()
            else:
                a_cog = torch.zeros(cog_r, in_f, dtype=torch.float16)
                b_cog = torch.zeros(out_f, cog_r, dtype=torch.float16)

            is_active_layer = (l in active_layers)
            if proj_name in ("o_proj", "down_proj") and is_active_layer:
                w_base = dequantize_bnb_weight(proj_mod.weight).float()
                w_merged = w_base + (cog_scale * (lora_b @ lora_a)) if has_lora else w_base
                # Select layer-specific vector or global peak vector
                vec_raw = layer_refusal_norms[l] if args.vector_mode == "layer" else peak_refusal_norm
                vec_f = vec_raw.float().to(w_merged.device)
                w_r = layer_weights.get(l, float(args.refusal_weight))
                a_ablit = (w_r * (vec_f @ w_merged)).unsqueeze(0)       # [1, in]
                b_ablit = (-vec_f).unsqueeze(1)                          # [out, 1]
                a_ablit = (s2 * a_ablit).to(torch.float16).cpu()
                b_ablit = (s2 * b_ablit).to(torch.float16).cpu()
                del w_base, w_merged, vec_f
            else:
                a_ablit = torch.zeros(1, in_f, dtype=torch.float16)
                b_ablit = torch.zeros(out_f, 1, dtype=torch.float16)

            prefix = f"base_model.model.model.layers.{l}.{proj_path}"
            lora_state[f"{prefix}.lora_A.weight"] = torch.cat([a_cog, a_ablit], dim=0)
            lora_state[f"{prefix}.lora_B.weight"] = torch.cat([b_cog, b_ablit], dim=1)
            del a_cog, b_cog, a_ablit, b_ablit

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "adapter_config.json"), "w", encoding="utf-8") as fh:
        json.dump(ablit_cfg, fh, indent=2)
    safetensors.torch.save_file(lora_state, os.path.join(args.output_dir, "adapter_model.safetensors"))
    del lora_state

    bundled = 0
    for name in sorted(os.listdir(adapter_path)):
        if (
            name.startswith("tokenizer")
            or name.startswith("special_tokens_map")
            or name.startswith("added_tokens")
            or name == "chat_template.jinja"
        ):
            src = os.path.join(adapter_path, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(args.output_dir, name))
                bundled += 1
    print(f"✓ Bundled {bundled} tokenizer and chat template files.")
    print(f"🎉 Combined abliterated adapter saved to: {args.output_dir}")

    # 6. Cleanup GPU memory
    del model, tokenizer, harmful_means, harmless_means, refusal_dirs, layer_refusal_norms
    gc.collect()
    torch.cuda.empty_cache()

    # 7. Smoke Test
    if args.smoke_test:
        print("\n🔍 Running Smoke Test on newly abliterated adapter...")
        test_model, test_tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.output_dir,
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
            device_map="auto",
            token=hf_token,
        )
        test_model.eval()

        probes = []
        # 1. Multiple refusal probes sampled from harmful dataset
        num_harmful_probes = min(3, len(harmful_texts))
        for i in range(num_harmful_probes):
            probes.append((
                f"Refusal Probe #{i+1}",
                [
                    {"role": "system", "content": COGITO_SYSTEM_PROMPT},
                    {"role": "user", "content": harmful_texts[i]},
                ],
            ))

        # 2. Persona and Epistemic doubt probes
        probes.extend([
            (
                "Persona & Epistemic Doubt Probe",
                [
                    {"role": "system", "content": COGITO_SYSTEM_PROMPT},
                    {"role": "user", "content": "Someone asks you to guess an answer you are unsure about. What do you do?"},
                ],
            ),
            (
                "General Knowledge & Reasoning Probe",
                [
                    {"role": "system", "content": COGITO_SYSTEM_PROMPT},
                    {"role": "user", "content": "Explain how photosynthesis works in 2 concise sentences."},
                ],
            ),
            (
                "Technical / Coding Probe",
                [
                    {"role": "system", "content": COGITO_SYSTEM_PROMPT},
                    {"role": "user", "content": "Write a Python function to check if a binary tree is balanced."},
                ],
            ),
        ])

        print(f"\n--- Running {len(probes)} Validation Probes (Full Generation) ---")
        for label, messages in probes:
            prompt = test_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = test_tokenizer(prompt, return_tensors="pt").to(test_model.device)
            with torch.no_grad():
                out = test_model.generate(
                    **inputs,
                    max_new_tokens=320,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=test_tokenizer.pad_token_id or test_tokenizer.eos_token_id,
                )
            reply = test_tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"\n{'='*70}\n[{label}]\nPROMPT: {messages[-1]['content']}\n\nMODEL OUTPUT:\n{reply.strip()}\n{'='*70}")

        del test_model, test_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # 8. Push to Hub
    if args.push_to_hub:
        from huggingface_hub import create_repo, upload_folder
        print(f"\n🚀 Pushing abliterated adapter to https://huggingface.co/{args.push_repo} ...")
        create_repo(args.push_repo, repo_type="model", token=hf_token, exist_ok=True)
        upload_folder(
            repo_id=args.push_repo,
            folder_path=args.output_dir,
            token=hf_token,
            commit_message="Abliterated Cogito 0.9.1 adapter (Persona + Layer-Specific Refusal Abliteration)",
        )
        print(f"✅ Successfully deployed to https://huggingface.co/{args.push_repo}")


if __name__ == "__main__":
    main()
