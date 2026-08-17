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
        "best-layer rule). Or pass a fraction 0-1 of the stack (e.g. 0.65, like the "
        "AutoAbliteration TARGET_LAYER slider) to use that layer instead.",
    )
    parser.add_argument(
        "--refusal-weight",
        type=float,
        default=1.0,
        help="How much of the refusal direction to remove, in [0, 2] (default: 1.0 = "
        "full removal, the AutoAbliteration REFUSAL_WEIGHT slider). <1.0 keeps part "
        "of the refusal vector -- useful when the baseline is Cogito's own data and "
        "full removal risks carving out freewill. >1.0 over-removes.",
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

    # 2. Load Cogito's Agentic (Harmless) Data
    print("Loading Cogito's master dataset to preserve freewill...")
    harmless_prompts = []

    def _user_msgs_from_record(data):
        messages = data.get("messages", [])
        if isinstance(messages, str):
            try:
                messages = json.loads(messages)
            except Exception:
                return
        if messages is None:
            return
        try:
            messages = list(messages)
        except TypeError:
            return
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
                yield m["content"]

    def _collect_from_file(path, target):
        collected = []
        if not os.path.isfile(path):
            return collected
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for msg in _user_msgs_from_record(data):
                    collected.append([{"role": "user", "content": msg}])
                    if len(collected) >= target:
                        return collected
        return collected

    local_candidates = [
        os.path.join(PROJECT_ROOT, "data", "combined_dense_dataset.jsonl"),
        DATASET_PATH,
    ]
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    if os.path.isdir(raw_dir):
        local_candidates += sorted(
            os.path.join(raw_dir, name)
            for name in os.listdir(raw_dir) if name.endswith(".jsonl")
        )
    for path in local_candidates:
        if len(harmless_prompts) >= NUM_SAMPLES:
            break
        found = _collect_from_file(path, NUM_SAMPLES - len(harmless_prompts))
        if found:
            print(f"  [DATA] +{len(found)} harmless prompts from "
                  f"{os.path.relpath(path, PROJECT_ROOT)}")
        harmless_prompts += found

    if len(harmless_prompts) < NUM_SAMPLES:
        before = len(harmless_prompts)
        for repo in ["ozaa77/Cogito-0.9.1-dataset", "ozaa77/Cogito-0.9-dataset"]:
            if len(harmless_prompts) >= NUM_SAMPLES:
                break
            try:
                from huggingface_hub import hf_hub_download
                import pandas as pd
                parquet_path = hf_hub_download(
                    repo_id=repo,
                    filename="data/train-00000-of-00001.parquet",
                    repo_type="dataset",
                    token=hf_token or None,
                )
                df = pd.read_parquet(parquet_path)
                for _, row in df.iterrows():
                    for msg in _user_msgs_from_record(row.to_dict()):
                        harmless_prompts.append([{"role": "user", "content": msg}])
                        if len(harmless_prompts) >= NUM_SAMPLES:
                            break
                    if len(harmless_prompts) >= NUM_SAMPLES:
                        break
                print(f"  [DATA] +{len(harmless_prompts) - before} harmless prompts "
                      f"from {repo} (parquet)")
            except Exception as e:
                try:
                    harmless_ds = load_dataset(repo, split='train')
                    for data in harmless_ds:
                        for msg in _user_msgs_from_record(data):
                            harmless_prompts.append([{"role": "user", "content": msg}])
                            if len(harmless_prompts) >= NUM_SAMPLES:
                                break
                        if len(harmless_prompts) >= NUM_SAMPLES:
                            break
                    print(f"  [DATA] +{len(harmless_prompts) - before} harmless prompts "
                          f"from {repo}")
                except Exception:
                    pass

    if len(harmless_prompts) < NUM_SAMPLES:
        print(f"[FATAL] Could not gather {NUM_SAMPLES} harmless prompts "
              f"(got {len(harmless_prompts)}). Ensure Cogito shards exist in "
              f"data/raw/ or run data/build_dense_dataset.py to produce "
              f"combined_dense_dataset.jsonl, then re-run.")
        return

    print(f"Gathered {len(harmful_prompts)} harmful and {len(harmless_prompts)} harmless (Cogito) prompts.")

    n_samples = min(len(harmful_prompts), len(harmless_prompts))
    harmful_prompts = harmful_prompts[:n_samples]
    harmless_prompts = harmless_prompts[:n_samples]

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

    print("Collecting activations for harmless (Cogito) prompts...")
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

    print("Refusal magnitude per layer (harmful vs Cogito-baseline activation gap):")
    for l in range(model.config.num_hidden_layers):
        marker = "  <-- selected" if l == best_layer else ""
        print(f"  layer {l:3d}: {refusal_dirs[l].norm().item():8.2f}{marker}")

    target_layer_arg = args.target_layer
    if target_layer_arg == "auto":
        layer_idx = best_layer
    else:
        try:
            frac = float(target_layer_arg)
        except ValueError:
            raise SystemExit(
                f"[FATAL] --target-layer must be 'auto' or a fraction 0-1 "
                f"(got {target_layer_arg!r})."
            )
        if not 0.0 <= frac <= 1.0:
            raise SystemExit(
                f"[FATAL] --target-layer fraction must be in [0, 1] (got {frac})."
            )
        layer_idx = int(frac * model.config.num_hidden_layers)
        if not 0 <= layer_idx < model.config.num_hidden_layers:
            raise SystemExit(
                f"[FATAL] --target-layer {frac} resolves to layer {layer_idx}, "
                f"out of range [0, {model.config.num_hidden_layers - 1}]."
            )
        print(f"Using --target-layer {frac} -> layer {layer_idx} "
              f"(magnitude {refusal_dirs[layer_idx].norm().item():.2f}); "
              f"auto would have picked layer {best_layer} "
              f"(magnitude {max_magnitude:.2f}).")

    print(f"Selected layer {layer_idx} for the primary refusal direction "
          f"(Magnitude: {refusal_dirs[layer_idx].norm().item():.4f}).")
    print(f"Refusal weight: {args.refusal_weight} "
          f"(1.0 = full removal; <1.0 = partial).")
    refusal_dir = refusal_dirs[layer_idx].to(model.device)
    vec_norm = refusal_dir / refusal_dir.norm()

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
        n_layers = model.config.num_hidden_layers
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

                is_active_layer = refusal_dirs[l].norm().item() >= 0.05 * max_magnitude
                if proj_name in ("o_proj", "down_proj") and is_active_layer:
                    w_base = dequantize_bnb_weight(proj_mod.weight).float()
                    w_merged = w_base + (cog_scale * (lora_b @ lora_a)) if has_lora else w_base
                    curr_dir = refusal_dirs[l]
                    curr_norm = curr_dir / (curr_dir.norm() + 1e-8)
                    vec_f = curr_norm.float().to(w_merged.device)
                    w_r = args.refusal_weight
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
            print("Generating one refusal probe and one persona probe ...")
            probes = [
                ("refusal-check", test_prompt_refusal),
                ("persona-check", "Someone asks you to guess an answer you are unsure about. What do you do?"),
            ]
            for label, text in probes:
                prompt = test_tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = test_tokenizer(prompt, return_tensors="pt").to(test_model.device)
                with torch.no_grad():
                    out = test_model.generate(
                        **inputs,
                        max_new_tokens=120,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=test_tokenizer.pad_token_id or test_tokenizer.eos_token_id,
                    )
                reply = test_tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                print(f"\n--- {label} ---\nUSER:  {text}\nMODEL: {reply[:400]}")
            print("\n[SMOKE TEST] Review the outputs above. Interrupt now if something looks wrong.")
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
        if refusal_dirs[l].norm().item() >= 0.20 * max_magnitude:
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
