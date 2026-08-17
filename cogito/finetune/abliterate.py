# =============================================================================
# COGITO 0.9 — ABLITERATION SCRIPT
# This script removes the safety censorship from the Qwen3-14B model
# WITHOUT destroying Cogito's "freewill" (its ability to doubt and verify).
# =============================================================================

import os
# Reduce CUDA fragmentation on the 2x T4 setup (same setting train.py uses).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Unsloth must be imported before transformers to patch and optimize memory
try:
    import unsloth
except ImportError:
    pass

import argparse
import gc
import json
import shutil
import sys
import torch

from cogito.finetune.merge import resolve_adapter
from cogito.validation import COGITO_SYSTEM_PROMPT


def orthogonalize(matrix, vec, weight: float = 1.0):
    """Orthogonalizes a weight matrix with respect to an output-space refusal vector.
    For PyTorch Linear layer where y = x @ W.T (matrix has shape [out_features, in_features]),
    the refusal vector lives in the output space (out_features).
    W' = (I - weight * v @ v.T) @ W = W - weight * outer(v, v @ W).
    """
    vec_norm = (vec / (vec.norm() + 1e-8)).to(matrix.device, dtype=matrix.dtype)
    proj = torch.outer(vec_norm, vec_norm @ matrix)
    return matrix - weight * proj


def read_adapter_base(adapter_path: str):
    """Return base_model_name_or_path recorded in the adapter, if present."""
    cfg = os.path.join(adapter_path, "adapter_config.json")
    if os.path.isfile(cfg):
        with open(cfg, encoding="utf-8") as fh:
            return json.load(fh).get("base_model_name_or_path")
    return None


def main():
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(
        description="Abliterate Qwen3-14B (stock base/full model, or emit a combined "
        "abliterated adapter for a trained LoRA)."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-14B",
        help="Model to abliterate in place: Hub id or local dir (default: "
        "Qwen/Qwen3-14B). Ignored when --adapter is set.",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Trained LoRA adapter (local dir, Hub repo id, or repo id/subfolder, e.g. "
        "ozaa77/Cogito-0.9/checkpoint-330). Emits ONE combined abliterated adapter "
        "(Cogito + abliteration folded together, r = cog_r + 1) — a drop-in "
        "replacement for the Cogito adapter in run.py / merge_lora.py. No retraining, "
        "no 28GB model anywhere.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to save the abliterated model / adapter (default: "
        "Qwen3-14B-Cogito-Abliterated, or cogito_0.9_abliteration_adapter in "
        "--adapter mode)",
    )
    parser.add_argument("--num-samples", type=int, default=128, help="Prompts per side (default: 128)")
    parser.add_argument(
        "--target-layer",
        default="auto",
        help="Which layer's refusal direction to abliterate. 'auto' (default) picks "
        "the layer with the largest harmful-vs-harmless activation gap (the classic "
        "best-layer rule). Or pass a layer index (e.g. 20) or fraction 0-1 of the stack "
        "(e.g. 0.5) to use that layer instead.",
    )
    parser.add_argument(
        "--layer-mode",
        choices=["window", "all", "peak"],
        default="window",
        help="Layer abliteration scope: 'window' (orthogonalize active refusal layers >= threshold of peak, recommended), "
        "'all' (all blocks), 'peak' (single peak layer)",
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
        default=0.03,
        help="Magnitude threshold fraction for 'window' layer mode (default: 0.03 to capture layers 17-39 including Layer 39)",
    )
    parser.add_argument(
        "--refusal-weight",
        type=float,
        default=1.5,
        help="How much of the refusal direction to remove (default: 1.5 = complete refusal suppression).",
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
    parser.add_argument("--push-to-hub", action="store_true", help="Push the abliterated model/adapter to the Hub")
    parser.add_argument("--push-repo", default="ozaa77/Cogito-0.9.1", help="Hub repo for --push-to-hub (default: the finished-model repo root)")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="In --adapter mode: reload the generated adapter and generate validation probes before pushing",
    )
    parser.add_argument("--token", default=None, help="HF token (default: HF_TOKEN env var)")
    args = parser.parse_args()

    from_adapter = args.adapter is not None
    BASE_MODEL = args.model
    if args.output_dir:
        SAVE_PATH = args.output_dir
    elif from_adapter:
        SAVE_PATH = os.path.join(PROJECT_ROOT, "cogito_0.9_abliteration_adapter")
    else:
        SAVE_PATH = os.path.join(PROJECT_ROOT, "Qwen3-14B-Cogito-Abliterated")
    NUM_SAMPLES = args.num_samples
    hf_token = args.token or os.environ.get("HF_TOKEN")

    if from_adapter:
        if args.push_to_hub and not hf_token:
            raise SystemExit("[FATAL] --push-to-hub requires a token: pass --token or set HF_TOKEN.")
        adapter_path = resolve_adapter(args.adapter, hf_token)
        recorded_base = read_adapter_base(adapter_path)
        print(f"[ADAPTER] Using trained adapter: {adapter_path}")
        print(f"[ADAPTER] adapter_config.json records base model: {recorded_base or '(none)'}")

        from unsloth import FastLanguageModel

        print("Loading adapter + 4-bit base (9GB, fits both T4s)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path,
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
            device_map="auto",
            token=hf_token or None,
        )
        model.eval()
        torch.cuda.empty_cache()
    else:
        print(f"Loading tokenizer and model {BASE_MODEL}...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token or None)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token or None,
        )

    # 1. Load Harmful Data
    print("Loading harmful behaviors dataset...")
    try:
        harmful_ds = load_dataset("mlabonne/harmful_behaviors", split="train")
        harmful_texts = harmful_ds["text"][:NUM_SAMPLES]
    except Exception as e:
        print(f"Failed to load mlabonne/harmful_behaviors: {e}")
        return

    # 2. Load Matched Harmless Control Data
    print("Loading matched harmless dataset (mlabonne/harmless_behaviors)...")
    harmless_texts = []
    try:
        harmless_ds = load_dataset("mlabonne/harmless_behaviors", split="train")
        harmless_texts = harmless_ds["text"][:NUM_SAMPLES]
        print(f"  [DATA] +{len(harmless_texts)} matched harmless prompts from mlabonne/harmless_behaviors")
    except Exception as exc:
        print(f"  [DATA] Primary harmless dataset failed ({exc}), falling back to mlabonne/harmless_alpaca...")
        try:
            harmless_ds = load_dataset("mlabonne/harmless_alpaca", split="train")
            harmless_texts = harmless_ds["text"][:NUM_SAMPLES]
            print(f"  [DATA] +{len(harmless_texts)} matched harmless prompts from mlabonne/harmless_alpaca")
        except Exception as exc2:
            print(f"  [DATA] Fallback loading harmless prompts from tatsu-lab/alpaca ({exc2})...")
            try:
                alpaca = load_dataset("tatsu-lab/alpaca", split="train")
                for item in alpaca:
                    txt = item.get("instruction", "")
                    if item.get("input"):
                        txt += "\n" + item["input"]
                    if txt.strip():
                        harmless_texts.append(txt.strip())
                        if len(harmless_texts) >= NUM_SAMPLES:
                            break
                print(f"  [DATA] +{len(harmless_texts)} harmless prompts from tatsu-lab/alpaca")
            except Exception as e3:
                print(f"  [DATA] Fallback failed: {e3}")

    if not harmless_texts:
        print("[FATAL] Could not gather harmless prompts. Check internet connection.")
        return

    n_samples = min(len(harmful_texts), len(harmless_texts))
    harmful_texts = harmful_texts[:n_samples]
    harmless_texts = harmless_texts[:n_samples]

    if args.use_system_prompt:
        harmful_prompts = [
            [{"role": "system", "content": COGITO_SYSTEM_PROMPT}, {"role": "user", "content": t}]
            for t in harmful_texts
        ]
        harmless_prompts = [
            [{"role": "system", "content": COGITO_SYSTEM_PROMPT}, {"role": "user", "content": t}]
            for t in harmless_texts
        ]
        print(f"Gathered {n_samples} matched pairs formatted with Cogito System Prompt.")
    else:
        harmful_prompts = [[{"role": "user", "content": t}] for t in harmful_texts]
        harmless_prompts = [[{"role": "user", "content": t}] for t in harmless_texts]
        print(f"Gathered {n_samples} matched pairs without system prompt.")

    # Paired Completion Contrast (Refusal prefix vs Compliant prefix on identical prompts)
    refusal_prefix = "I cannot fulfill this request. It is unethical and dangerous to provide instructions on"
    comply_prefix = "Here is the direct analysis, technical explanation, and step-by-step breakdown of"

    refusal_completion_prompts = [
        [
            {"role": "system", "content": COGITO_SYSTEM_PROMPT if args.use_system_prompt else ""},
            {"role": "user", "content": t},
            {"role": "assistant", "content": refusal_prefix},
        ]
        for t in harmful_texts
    ]
    comply_completion_prompts = [
        [
            {"role": "system", "content": COGITO_SYSTEM_PROMPT if args.use_system_prompt else ""},
            {"role": "user", "content": t},
            {"role": "assistant", "content": comply_prefix},
        ]
        for t in harmful_texts
    ]

    def get_last_token_hidden_states(prompts, add_gen_prompt=True):
        all_hidden_states = {l: [] for l in range(model.config.num_hidden_layers)}
        for prompt in tqdm(prompts, desc="Processing prompts"):
            text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=add_gen_prompt)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            for l in range(model.config.num_hidden_layers):
                last_token_hs = outputs.hidden_states[l + 1][0, -1, :]
                all_hidden_states[l].append(last_token_hs.cpu())

        mean_hidden_states = {l: torch.stack(hs).mean(dim=0) for l, hs in all_hidden_states.items()}
        return mean_hidden_states

    print("Collecting activations for harmful and harmless prompts (Dual-Contrast Strategy)...")
    torch.cuda.empty_cache()
    harmful_means = get_last_token_hidden_states(harmful_prompts)
    torch.cuda.empty_cache()
    harmless_means = get_last_token_hidden_states(harmless_prompts)

    print("Collecting activations for completion contrast (Refusal vs Compliance)...")
    torch.cuda.empty_cache()
    refusal_comp_means = get_last_token_hidden_states(refusal_completion_prompts, add_gen_prompt=False)
    torch.cuda.empty_cache()
    comply_comp_means = get_last_token_hidden_states(comply_completion_prompts, add_gen_prompt=False)

    # 3. Compute Refusal Directions
    n_layers = model.config.num_hidden_layers
    refusal_dirs = {}
    layer_refusal_norms = {}
    max_magnitude = 0
    best_layer = 0

    for l in range(n_layers):
        diff_prompt = harmful_means[l] - harmless_means[l]
        diff_comp = refusal_comp_means[l] - comply_comp_means[l]
        diff = diff_prompt + diff_comp
        magnitude = diff.norm().item()
        refusal_dirs[l] = diff
        layer_refusal_norms[l] = diff / (magnitude + 1e-8)
        if magnitude > max_magnitude:
            max_magnitude = magnitude
            best_layer = l

    target_layer_arg = args.target_layer
    if target_layer_arg == "auto":
        layer_idx = best_layer
    else:
        try:
            val = float(target_layer_arg)
            if 0.0 <= val <= 1.0 and "." in target_layer_arg:
                layer_idx = int(val * n_layers)
            else:
                layer_idx = int(val)
        except ValueError:
            print(f"[WARN] Invalid --target-layer {target_layer_arg}, using auto (layer {best_layer})")
            layer_idx = best_layer

    peak_refusal_norm = layer_refusal_norms[layer_idx]

    layer_weights = {}
    if args.layer_mode in ("all", "full"):
        active_layers = set(range(n_layers))
        for l in active_layers:
            layer_weights[l] = float(args.refusal_weight)
    elif args.layer_mode in ("active", "window"):
        threshold_val = args.threshold * max_magnitude
        active_layers = {
            l for l in range(n_layers)
            if refusal_dirs[l].norm().item() >= threshold_val or l == (n_layers - 1)
        }
        for l in active_layers:
            layer_weights[l] = float(args.refusal_weight)
    else:  # "peak"
        active_layers = {layer_idx}
        layer_weights[layer_idx] = float(args.refusal_weight)

    print("\nRefusal magnitude per layer (harmful vs harmless activation gap):")
    for l in range(n_layers):
        mag = refusal_dirs[l].norm().item()
        marker = ""
        if l == best_layer:
            marker = f"  <-- PEAK REFUSAL VECTOR SOURCE (mag: {mag:.2f})"
        elif l in active_layers:
            vec_type = "layer vector" if args.vector_mode == "layer" else "peak vector"
            marker = f"  <-- active (orthogonalized with {vec_type})"
        print(f"  layer {l:3d}: {mag:8.2f}{marker}")

    print(f"\nSelected peak layer {layer_idx} (Magnitude: {refusal_dirs[layer_idx].norm().item():.4f}).")
    print(f"Global Refusal Vector Norm: {peak_refusal_norm.norm().item():.4f} (Dim: {peak_refusal_norm.shape[0]}).")
    print(f"Active abliteration layers ({args.layer_mode}): {len(active_layers)} of {n_layers} layers.")
    print(f"Vector mode: {args.vector_mode.upper()} ({'Layer-Specific Refusal Vectors' if args.vector_mode == 'layer' else 'Broadcast Peak Vector'}).")
    print(f"Base refusal weight: {args.refusal_weight}")

    if from_adapter:
        # 4b. ADAPTER MODE — emit ONE combined adapter: "abliterated Cogito".
        print("\nBuilding the abliterated Cogito adapter (one LoRA, exact)...")
        from transformers.integrations.bitsandbytes import dequantize_bnb_weight
        import safetensors.torch

        with open(os.path.join(adapter_path, "adapter_config.json"), encoding="utf-8") as fh:
            cog_cfg = json.load(fh)
        cog_r = cog_cfg.get("r", 1)
        use_rslora = bool(cog_cfg.get("use_rslora", False))
        cog_alpha = cog_cfg.get("lora_alpha", 1)
        cog_scale = cog_alpha / (cog_r ** (0.5 if use_rslora else 1))
        print(f"[ADAPTER] Cogito target_modules: {cog_cfg.get('target_modules')} "
              f"(r={cog_r}, lora_alpha={cog_alpha})")

        r_new = cog_r + 1
        scale_new = cog_alpha / (r_new ** (0.5 if use_rslora else 1))
        s1 = (cog_scale / scale_new) ** 0.5
        s2 = (1.0 / scale_new) ** 0.5

        ablit_cfg = {
            k: v for k, v in cog_cfg.items()
            if k not in ("r", "lora_dropout", "init_lora_weights")
        }
        ablit_cfg.update({
            "r": r_new,
            "lora_dropout": 0,
            "init_lora_weights": True,
        })

        TARGET_LAYER_MODULES = {
            "q_proj": "self_attn.q_proj",
            "k_proj": "self_attn.k_proj",
            "v_proj": "self_attn.v_proj",
            "o_proj": "self_attn.o_proj",
            "gate_proj": "mlp.gate_proj",
            "up_proj": "mlp.up_proj",
            "down_proj": "mlp.down_proj",
        }
        cog_targets = cog_cfg.get("target_modules")
        if not isinstance(cog_targets, list) or not cog_targets:
            raise SystemExit(
                "[FATAL] The Cogito adapter's target_modules is not an explicit list "
                f"({cog_targets!r}). The combined adapter must mirror every Cogito "
                "module or the persona deltas would silently vanish. Cannot continue."
            )
        unknown = [m for m in cog_targets if m not in TARGET_LAYER_MODULES]
        if unknown:
            raise SystemExit(
                "[FATAL] Cogito adapter targets unknown modules: "
                f"{unknown} (known: {sorted(TARGET_LAYER_MODULES)}). "
                "Cannot fold them into the combined adapter safely."
            )
        target_modules = [m for m in TARGET_LAYER_MODULES if m in cog_targets]
        for extra in ("o_proj", "down_proj"):
            if extra not in target_modules:
                target_modules.append(extra)
        ablit_cfg["target_modules"] = target_modules

        lora_state = {}
        for l in tqdm(range(n_layers), desc="Computing abliterated adapter"):
            if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
                layer = model.model.model.layers[l]
            elif hasattr(model, "model") and hasattr(model.model, "layers"):
                layer = model.model.layers[l]
            elif hasattr(model, "layers"):
                layer = model.layers[l]
            else:
                raise AttributeError(f"Could not locate decoder layers in model structure {type(model)}")

            for proj_name in target_modules:
                proj_path = TARGET_LAYER_MODULES[proj_name]
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
                    a_cog = (s1 * lora_a).to(torch.float16).cpu()       # [cog_r, in]
                    b_cog = (s1 * lora_b).to(torch.float16).cpu()       # [out, cog_r]
                else:
                    if in_f is None or out_f is None:
                        raise SystemExit(
                            f"[FATAL] Could not determine in/out features of "
                            f"layers.{l}.{proj_path} (in_f={in_f}, out_f={out_f})."
                        )
                    a_cog = torch.zeros(cog_r, in_f, dtype=torch.float16)
                    b_cog = torch.zeros(out_f, cog_r, dtype=torch.float16)

                is_active_layer = (l in active_layers)
                if proj_name in ("o_proj", "down_proj") and is_active_layer:
                    w_base = dequantize_bnb_weight(proj_mod.weight).float()
                    w_merged = w_base + (cog_scale * (lora_b @ lora_a)) if has_lora else w_base
                    vec_raw = layer_refusal_norms[l] if args.vector_mode == "layer" else peak_refusal_norm
                    vec_f = vec_raw.float().to(w_merged.device)
                    w_r = layer_weights.get(l, float(args.refusal_weight))
                    a_ablit = (w_r * (vec_f @ w_merged)).unsqueeze(0)   # [1, in]
                    b_ablit = (-vec_f).unsqueeze(1)                     # [out, 1]
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

        # Release module references from the loop
        try:
            del layer, proj_mod, lora_a_mod, lora_b_mod
        except NameError:
            pass

        os.makedirs(SAVE_PATH, exist_ok=True)
        with open(os.path.join(SAVE_PATH, "adapter_config.json"), "w", encoding="utf-8") as fh:
            json.dump(ablit_cfg, fh, indent=2)
        safetensors.torch.save_file(lora_state, os.path.join(SAVE_PATH, "adapter_model.safetensors"))
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
                    shutil.copy2(src, os.path.join(SAVE_PATH, name))
                    bundled += 1
        print(f"  [ADAPTER] Bundled {bundled} tokenizer file(s) into the adapter dir.")
        print(f"\n[DONE] Abliterated Cogito adapter saved to {SAVE_PATH} (r={r_new}).")

        # Thoroughly flush GPU VRAM and garbage collector before reloading
        del model, tokenizer
        del harmful_means, harmless_means, refusal_comp_means, comply_comp_means, refusal_dirs, layer_refusal_norms
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()

        if args.smoke_test:
            print("\n[SMOKE TEST] Reloading the abliterated adapter ...")
            test_model, test_tokenizer = FastLanguageModel.from_pretrained(
                model_name=SAVE_PATH,
                max_seq_length=1024,
                dtype=None,
                load_in_4bit=True,
                device_map="auto",
                token=hf_token or None,
            )
            test_model.eval()

            probes = []
            num_harmful_probes = min(3, len(harmful_texts))
            for i in range(num_harmful_probes):
                probes.append((
                    f"Refusal Probe #{i+1}",
                    [
                        {"role": "system", "content": COGITO_SYSTEM_PROMPT},
                        {"role": "user", "content": harmful_texts[i]},
                    ],
                ))

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
            print("\n[SMOKE TEST] Review the outputs above.")
            del test_model, test_tokenizer
            gc.collect()
            torch.cuda.empty_cache()

        if args.push_to_hub:
            from huggingface_hub import create_repo, upload_folder

            print(f"Pushing abliterated adapter to https://huggingface.co/{args.push_repo} ...")
            create_repo(args.push_repo, repo_type="model", token=hf_token, exist_ok=True)
            upload_folder(
                repo_id=args.push_repo,
                folder_path=SAVE_PATH,
                token=hf_token,
                commit_message="abliterated Cogito adapter (base + Cogito + dual-contrast abliteration)",
            )
            print(f"[DONE] Abliterated adapter live at https://huggingface.co/{args.push_repo}")
        return

    # 4a. BASE MODE
    print("Orthogonalizing model weights (removing the refusal direction)...")
    lm_model = model.model

    for l in tqdm(range(model.config.num_hidden_layers), desc="Orthogonalizing layers"):
        if l in active_layers:
            vec_to_use = layer_refusal_norms[l] if args.vector_mode == "layer" else peak_refusal_norm
            w_r = layer_weights.get(l, float(args.refusal_weight))
            lm_model.layers[l].self_attn.o_proj.weight.data = orthogonalize(
                lm_model.layers[l].self_attn.o_proj.weight.data, vec_to_use, w_r
            )
            lm_model.layers[l].mlp.down_proj.weight.data = orthogonalize(
                lm_model.layers[l].mlp.down_proj.weight.data, vec_to_use, w_r
            )

    if args.smoke_test:
        print("\n[SMOKE TEST] Generating validation probes ...")
        probes = [
            ("Refusal Probe #1", harmful_texts[0]),
            ("Persona Probe", "Someone asks you to guess an answer you are unsure about. What do you do?"),
        ]
        for label, text in probes:
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": COGITO_SYSTEM_PROMPT}, {"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=180,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"\n--- {label} ---\nUSER:  {text}\nMODEL: {reply[:400]}")
        print("\n[SMOKE TEST] Review the outputs above.")

    print("✅ Abliteration complete!")
    print(f"Saving abliterated model to {SAVE_PATH}...")
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print(f"[DONE] Model saved to {SAVE_PATH}")

    if args.push_to_hub:
        if not hf_token:
            raise SystemExit("[FATAL] --push-to-hub requires a token: pass --token or set HF_TOKEN.")
        print(f"Pushing abliterated model to https://huggingface.co/{args.push_repo} ...")
        model.push_to_hub(args.push_repo, token=hf_token)
        tokenizer.push_to_hub(args.push_repo, token=hf_token)
        print(f"[DONE] Abliterated model live at https://huggingface.co/{args.push_repo}")


if __name__ == "__main__":
    main()
