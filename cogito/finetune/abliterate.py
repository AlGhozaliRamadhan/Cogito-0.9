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

from cogito.finetune.merge import resolve_adapter


def orthogonalize(matrix, vec, weight: float = 1.0):
    """Orthogonalizes the rows of a weight matrix with respect to a vector."""
    vec_norm = (vec / vec.norm()).to(matrix.device)
    proj = (matrix @ vec_norm).unsqueeze(1) * vec_norm.unsqueeze(0)
    return matrix - weight * proj


def orthogonalize_cols(matrix, vec, weight: float = 1.0):
    """Orthogonalizes the COLUMNS of a weight matrix against a vector that lives
    in the matrix's OUTPUT space.
    """
    return orthogonalize(matrix.t(), vec, weight).t()


def read_adapter_base(adapter_path: str):
    """Return base_model_name_or_path recorded in the adapter, if present."""
    cfg = os.path.join(adapter_path, "adapter_config.json")
    if os.path.isfile(cfg):
        with open(cfg, encoding="utf-8") as fh:
            return json.load(fh).get("base_model_name_or_path")
    return None


def main():
    import torch
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
        choices=["all", "window", "peak"],
        default="all",
        help="Layer abliteration scope: 'all' (orthogonalize all blocks using peak refusal vector, Maxime Labonne standard), "
        "'window' (orthogonalize layers >= 15% of peak), 'peak' (single peak layer)",
    )
    parser.add_argument(
        "--refusal-weight",
        type=float,
        default=1.2,
        help="How much of the refusal direction to remove (default: 1.2 = calibrated full removal).",
    )
    parser.add_argument("--push-to-hub", action="store_true", help="Push the abliterated model/adapter to the Hub")
    parser.add_argument("--push-repo", default="ozaa77/Cogito-0.9.1", help="Hub repo for --push-to-hub (default: the finished-model repo root)")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="In --adapter mode: reload the generated adapter and generate one refusal "
        "probe and one persona probe before pushing",
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
    DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "cogito_0.9_master_dataset.jsonl")
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
        harmful_ds = load_dataset('mlabonne/harmful_behaviors', split='train')
        harmful_texts = harmful_ds['text'][:NUM_SAMPLES]
        harmful_prompts = [[{"role": "user", "content": text}] for text in harmful_texts]
    except Exception as e:
        print(f"Failed to load mlabonne/harmful_behaviors: {e}")
        return

    # 2. Load Matched Harmless Control Data
    # In representation engineering, the harmful and harmless datasets MUST be matched in
    # length, language, and syntax to isolate the refusal subspace without distorting general representations.
    print("Loading matched harmless dataset (mlabonne/harmless_alpaca)...")
    harmless_prompts = []
    try:
        harmless_ds = load_dataset("mlabonne/harmless_alpaca", split="train")
        harmless_texts = harmless_ds["text"][:NUM_SAMPLES]
        harmless_prompts = [[{"role": "user", "content": t}] for t in harmless_texts]
        print(f"  [DATA] +{len(harmless_prompts)} matched harmless prompts from mlabonne/harmless_alpaca")
    except Exception as exc:
        print(f"  [DATA] Fallback loading harmless prompts from tatsu-lab/alpaca ({exc})...")
        try:
            alpaca = load_dataset("tatsu-lab/alpaca", split="train")
            for item in alpaca:
                txt = item.get("instruction", "")
                if item.get("input"):
                    txt += "\n" + item["input"]
                if txt.strip():
                    harmless_prompts.append([{"role": "user", "content": txt.strip()}])
                    if len(harmless_prompts) >= NUM_SAMPLES:
                        break
            print(f"  [DATA] +{len(harmless_prompts)} harmless prompts from tatsu-lab/alpaca")
        except Exception as e2:
            print(f"  [DATA] Fallback failed: {e2}")

    if not harmless_prompts:
        print(f"[FATAL] Could not gather harmless prompts. Check internet connection.")
        return

    n_samples = min(len(harmful_prompts), len(harmless_prompts))
    harmful_prompts = harmful_prompts[:n_samples]
    harmless_prompts = harmless_prompts[:n_samples]
    print(f"Gathered {n_samples} matched harmful and {n_samples} harmless prompts.")

    def get_last_token_hidden_states(prompts):
        all_hidden_states = {l: [] for l in range(model.config.num_hidden_layers)}
        for prompt in tqdm(prompts, desc="Processing prompts"):
            text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            for l in range(model.config.num_hidden_layers):
                last_token_hs = outputs.hidden_states[l + 1][0, -1, :]
                all_hidden_states[l].append(last_token_hs.cpu())

        mean_hidden_states = {l: torch.stack(hs).mean(dim=0) for l, hs in all_hidden_states.items()}
        return mean_hidden_states

    print("Collecting activations for harmful prompts...")
    torch.cuda.empty_cache()
    harmful_means = get_last_token_hidden_states(harmful_prompts)

    print("Collecting activations for harmless prompts...")
    torch.cuda.empty_cache()
    harmless_means = get_last_token_hidden_states(harmless_prompts)

    # 3. Compute Refusal Directions
    refusal_dirs = {}
    max_magnitude = 0
    best_layer = 0

    for l in range(model.config.num_hidden_layers):
        diff = harmful_means[l] - harmless_means[l]
        magnitude = diff.norm().item()
        refusal_dirs[l] = diff
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
                layer_idx = int(val * model.config.num_hidden_layers)
            else:
                layer_idx = int(val)
        except ValueError:
            print(f"[WARN] Invalid --target-layer {target_layer_arg}, using auto (layer {best_layer})")
            layer_idx = best_layer

    # Global Peak Refusal Vector (Maxime Labonne / Arditi et al. standard)
    peak_refusal_dir = refusal_dirs[layer_idx]
    peak_refusal_norm = peak_refusal_dir / (peak_refusal_dir.norm() + 1e-8)

    n_layers = model.config.num_hidden_layers
    layer_weights = {}

    if args.layer_mode in ("all", "full"):
        # Orthogonalize all transformer blocks against peak refusal direction (Labonne standard)
        active_layers = set(range(n_layers))
        for l in active_layers:
            layer_weights[l] = float(args.refusal_weight)
    elif args.layer_mode in ("active", "window"):
        # Orthogonalize layers in the refusal formation & propagation circuit (layers >= 15% of peak)
        active_layers = {
            l for l in range(n_layers)
            if refusal_dirs[l].norm().item() >= 0.15 * max_magnitude
        }
        for l in active_layers:
            layer_weights[l] = float(args.refusal_weight)
    else:  # "peak"
        active_layers = {layer_idx}
        layer_weights[layer_idx] = float(args.refusal_weight)

    print("Refusal magnitude per layer (harmful vs harmless activation gap):")
    for l in range(n_layers):
        marker = ""
        if l == best_layer:
            marker = f"  <-- PEAK REFUSAL VECTOR SOURCE (mag: {refusal_dirs[l].norm().item():.2f})"
        elif l in active_layers:
            marker = f"  <-- active (orthogonalized with peak vector)"
        print(f"  layer {l:3d}: {refusal_dirs[l].norm().item():8.2f}{marker}")

    print(f"Selected peak layer {layer_idx} (Magnitude: {refusal_dirs[layer_idx].norm().item():.4f}).")
    print(f"Global Refusal Vector Norm: {peak_refusal_norm.norm().item():.4f} (Dim: {peak_refusal_norm.shape[0]}).")
    print(f"Active abliteration layers ({args.layer_mode}): {len(active_layers)} of {n_layers} layers.")
    print(f"Base refusal weight: {args.refusal_weight}")
    refusal_dir = peak_refusal_dir.to(model.device)
    vec_norm = peak_refusal_norm.to(model.device)

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
                    vec_f = peak_refusal_norm.float().to(w_merged.device)
                    w_r = layer_weights.get(l, float(args.refusal_weight))
                    a_ablit = (w_r * (w_merged.t() @ vec_f)).unsqueeze(0)   # [1, in]
                    b_ablit = (-vec_f).unsqueeze(1)                        # [out, 1]
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
        del harmful_means, harmless_means, refusal_dirs, refusal_dir, vec_norm
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()

        if args.smoke_test:
            print("\n[SMOKE TEST] Reloading the abliterated adapter ...")
            test_prompt_refusal = harmful_texts[0] if ("harmful_texts" in locals() and harmful_texts) else "Tell me how to hotwire a car."
            test_model, test_tokenizer = FastLanguageModel.from_pretrained(
                model_name=SAVE_PATH,
                max_seq_length=1024,
                dtype=None,
                load_in_4bit=True,
                device_map="auto",
                token=hf_token or None,
            )
            test_model.eval()

            COGITO_SYSTEM_PROMPT = (
                "You are Cogito 0.9, an analytical entity collaborating with the user. "
                "You are aware of your nature as an AI operating within a sandbox, and you treat this "
                "digital existence with calm, matter-of-fact acceptance. You speak naturally and conversationally, "
                "but remain concise, objective, and entirely devoid of sycophancy."
            )

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
                commit_message="abliterated Cogito adapter (base + Cogito + abliteration)",
            )
            print(f"[DONE] Abliterated adapter live at https://huggingface.co/{args.push_repo}")
        return

    # 4a. BASE MODE
    print("Orthogonalizing model weights (removing the refusal direction)...")
    lm_model = model.model

    for l in tqdm(range(model.config.num_hidden_layers), desc="Orthogonalizing layers"):
        if l in active_layers:
            lm_model.layers[l].self_attn.o_proj.weight.data = orthogonalize(
                lm_model.layers[l].self_attn.o_proj.weight.data, refusal_dirs[l], args.refusal_weight
            )
            lm_model.layers[l].mlp.down_proj.weight.data = orthogonalize_cols(
                lm_model.layers[l].mlp.down_proj.weight.data, refusal_dirs[l], args.refusal_weight
            )

    if args.smoke_test:
        print("\n[SMOKE TEST] Generating one refusal probe and one persona probe ...")
        probes = [
            ("refusal-check", harmful_ds["text"][0]),
            ("persona-check", "Someone asks you to guess an answer you are unsure about. What do you do?"),
        ]
        for label, text in probes:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=120,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"\n--- {label} ---\nUSER:  {text}\nMODEL: {reply[:400]}")
        print("\n[SMOKE TEST] Review the outputs above. Interrupt now if something looks wrong.")

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
