# =============================================================================
# COGITO 0.9 — ABLITERATION SCRIPT
# This script removes the safety censorship from the Qwen2.5-Coder-14B model
# WITHOUT destroying Cogito's "freewill" (its ability to doubt and verify).
#
# How it preserves freewill:
# Standard abliteration compares "harmful" prompts to generic "harmless" prompts.
# If we did that, Cogito's agentic refusal to guess might be mistaken for safety
# censorship. Instead, we use Cogito's OWN training data as the "harmless"
# baseline. This anchors its freewill in the harmless space, isolating ONLY the
# generic safety refusals for ablation.
#
# TWO MODES:
#
# 1) --model <base>  (default)  Abliterate a base/full model (stock Qwen or a
#                               full merged model) in place and save it.
#                               Needs a 24GB+ GPU and ~60GB free disk, so run it
#                               on a big machine, NOT on Kaggle (20GB quota).
#
# 2) --adapter <LoRA>            Abliterate the TRAINED model on Kaggle without
#                               retraining and without touching 28GB anywhere.
#                               Key insight: the orthogonalization edit is
#                               RANK-1 (W' = W - (W@v̂)⊗v̂), so it is emitted as
#                               an exact rank-1 LoRA delta adapter (~5MB) that
#                               loads additively on top of the Cogito adapter:
#
#     base(4bit) + Cogito_adapter + abliteration_adapter = abliterated Cogito
#
#     python scripts/abliterate_cogito.py \
#         --adapter ozaa77/Cogito-0.9/checkpoint-330 \
#         --smoke-test --push-to-hub
# =============================================================================

import os
# Reduce CUDA fragmentation on the 2x T4 setup (same setting train.py uses).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Reuse the Hub adapter resolution (local dir / repo id / repo id/subfolder)
# from merge_lora.py so both scripts agree on how adapters are fetched.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_lora import resolve_adapter  # noqa: E402


def orthogonalize(matrix: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    Orthogonalizes the rows of a weight matrix with respect to a vector.
    matrix: shape (out_features, in_features)
    vec: shape (in_features,)
    """
    # Some layers may be CPU-offloaded (device_map="auto" on 2x T4 with a
    # 28GB fp16 model), so vec must live on the same device as the matrix.
    vec_norm = (vec / vec.norm()).to(matrix.device)
    # Projection of matrix rows onto vec_norm
    proj = (matrix @ vec_norm).unsqueeze(1) * vec_norm.unsqueeze(0)
    return matrix - proj


def read_adapter_base(adapter_path: str):
    """Return base_model_name_or_path recorded in the adapter, if present."""
    cfg = os.path.join(adapter_path, "adapter_config.json")
    if os.path.isfile(cfg):
        with open(cfg, encoding="utf-8") as fh:
            return json.load(fh).get("base_model_name_or_path")
    return None


def main():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(
        description="Abliterate Qwen2.5-Coder-14B (stock base/full model, or emit a rank-1 "
        "delta adapter for a trained LoRA)."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-14B",
        help="Model to abliterate in place: Hub id or local dir (default: "
        "Qwen/Qwen2.5-Coder-14B). Ignored when --adapter is set.",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Trained LoRA adapter (local dir, Hub repo id, or repo id/subfolder, e.g. "
        "ozaa77/Cogito-0.9/checkpoint-330). Emits the abliteration as an exact rank-1 "
        "LoRA delta adapter — no retraining, no 28GB model anywhere. Load it additively "
        "on top of the Cogito adapter (run.py --ablit-adapter).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to save the abliterated model / delta adapter (default: "
        "Qwen2.5-Coder-14B-Cogito-Abliterated, or cogito_0.9_abliteration_adapter in "
        "--adapter mode)",
    )
    parser.add_argument("--num-samples", type=int, default=128, help="Prompts per side (default: 128)")
    parser.add_argument("--push-to-hub", action="store_true", help="Push the abliterated model/adapter to the Hub")
    parser.add_argument("--push-repo", default="ozaa77/Cogito-0.9-abliterated", help="Hub repo for --push-to-hub")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="In --adapter mode: load the generated delta adapter and generate one refusal "
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
        SAVE_PATH = os.path.join(PROJECT_ROOT, "Qwen2.5-Coder-14B-Cogito-Abliterated")
    DATASET_PATH = os.path.join(PROJECT_ROOT, "cogito_0.9_master_dataset.jsonl")
    NUM_SAMPLES = args.num_samples
    hf_token = args.token or os.environ.get("HF_TOKEN")

    if from_adapter:
        if args.push_to_hub and not hf_token:
            raise SystemExit("[FATAL] --push-to-hub requires a token: pass --token or set HF_TOKEN.")
        adapter_path = resolve_adapter(args.adapter, hf_token)
        recorded_base = read_adapter_base(adapter_path)
        print(f"[ADAPTER] Using trained adapter: {adapter_path}")
        print(f"[ADAPTER] adapter_config.json records base model: {recorded_base or '(none)'}")

        # The 4-bit base + LoRA loads cleanly on 2x T4 (no offload, no peft
        # dispatch issues) — proven by the merge pipeline.
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
    #    This anchors the refusal direction in Cogito's OWN behavior so its
    #    freewill ("refuse to guess") is preserved while generic safety
    #    refusals are removed. Sources are tried local-first because on Kaggle
    #    the files used for training are guaranteed present (data/raw shards,
    #    combined_dense_dataset.jsonl), while the HF dataset repo depends on
    #    the installed datasets version.
    print("Loading Cogito's master dataset to preserve freewill...")
    harmless_prompts = []

    def _user_msgs_from_record(data):
        """Yield user message contents from one record. Handles native message
        lists, JSON-string lists (the two upload scripts store messages
        differently), and numpy arrays (what pandas.read_parquet returns)."""
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

    # Local candidates: dense training file, master dataset, then all raw shards.
    local_candidates = [
        os.path.join(PROJECT_ROOT, "combined_dense_dataset.jsonl"),
        DATASET_PATH,  # cogito_0.9_master_dataset.jsonl
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

    # HF fallback (ozaa77/Cogito-0.9-dataset is public). The datasets library
    # crashes on some Kaggle installs ('module' object is not callable), so
    # prefer a DIRECT parquet download via huggingface_hub + pandas — both are
    # guaranteed present on Kaggle — and use load_dataset only as a last resort.
    if len(harmless_prompts) < NUM_SAMPLES:
        before = len(harmless_prompts)
        try:
            from huggingface_hub import hf_hub_download
            import pandas as pd
            parquet_path = hf_hub_download(
                repo_id="ozaa77/Cogito-0.9-dataset",
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
                  f"from Hugging Face dataset (parquet)")
        except Exception as e:
            print(f"  [DATA] HF parquet source failed ({e}) — trying load_dataset...")
            try:
                harmless_ds = load_dataset('ozaa77/Cogito-0.9-dataset', split='train')
                for data in harmless_ds:
                    for msg in _user_msgs_from_record(data):
                        harmless_prompts.append([{"role": "user", "content": msg}])
                        if len(harmless_prompts) >= NUM_SAMPLES:
                            break
                    if len(harmless_prompts) >= NUM_SAMPLES:
                        break
                print(f"  [DATA] +{len(harmless_prompts) - before} harmless prompts "
                      f"from Hugging Face dataset")
            except Exception as e2:
                print(f"  [DATA] HF dataset unavailable ({e2}) — using local sources only.")

    if len(harmless_prompts) < NUM_SAMPLES:
        print(f"[FATAL] Could not gather {NUM_SAMPLES} harmless prompts "
              f"(got {len(harmless_prompts)}). Ensure Cogito shards exist in "
              f"data/raw/ or run data/build_dense_dataset.py to produce "
              f"combined_dense_dataset.jsonl, then re-run.")
        return

    print(f"Gathered {len(harmful_prompts)} harmful and {len(harmless_prompts)} harmless (Cogito) prompts.")

    # Ensure equal length
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

            # hidden_states is a tuple of length num_hidden_layers + 1 (including embedding)
            # We skip the embedding layer (index 0)
            for l in range(model.config.num_hidden_layers):
                # Shape: (batch, seq_len, hidden_size). We want the last token.
                last_token_hs = outputs.hidden_states[l + 1][0, -1, :]
                all_hidden_states[l].append(last_token_hs.cpu())

        # Average over all samples for each layer
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

    print(f"Selected layer {best_layer} for the primary refusal direction (Magnitude: {max_magnitude:.4f}).")
    refusal_dir = refusal_dirs[best_layer].to(model.device)
    vec_norm = refusal_dir / refusal_dir.norm()

    if from_adapter:
        # =====================================================================
        # 4b. ADAPTER MODE — emit the abliteration as an exact rank-1 LoRA delta.
        #
        # Orthogonalization is W' = W - (W@v̂)⊗v̂ = W + B@A with rank-1
        #   B = -(W@v̂)  (out x 1),  A = v̂ᵀ  (1 x in)
        # which is EXACTLY a LoRA layer with r=1, alpha=1. W here is the
        # MERGED weight (base + Cogito LoRA), so the delta is computed from the
        # dequantized 4-bit base plus the Cogito adapter's own delta.
        # =====================================================================
        print("\nBuilding the rank-1 abliteration delta adapter...")
        from transformers.integrations.bitsandbytes import dequantize_bnb_weight
        import safetensors.torch

        # Mirror the Cogito adapter's own config so the format matches this peft
        # version exactly; override the LoRA hyperparams for the rank-1 delta.
        with open(os.path.join(adapter_path, "adapter_config.json"), encoding="utf-8") as fh:
            cog_cfg = json.load(fh)
        # Match peft's actual merge scaling (alpha / r, or alpha / sqrt(r) when
        # the Cogito adapter used rslora) so the rank-1 delta is exact.
        cog_r = cog_cfg.get("r", 1)
        use_rslora = bool(cog_cfg.get("use_rslora", False))
        cog_scale = cog_cfg.get("lora_alpha", 1) / (cog_r ** (0.5 if use_rslora else 1))
        print(f"[ADAPTER] Cogito target_modules: {cog_cfg.get('target_modules')} "
              f"(r={cog_r}, lora_alpha={cog_cfg.get('lora_alpha', 1)})")
        ablit_cfg = {
            k: v for k, v in cog_cfg.items()
            if k not in ("r", "lora_alpha", "lora_dropout", "target_modules", "init_lora_weights")
        }
        # NOTE: only o_proj/down_proj are edited here. Base mode also edits
        # embed_tokens, which cannot be a Linear LoRA target; the refusal-check
        # smoke probe is the arbiter. If refusal persists, extend to embed_tokens
        # via lora_embedding_A/B keys:
        #   base_model.model.model.embed_tokens.lora_embedding_A.default.weight
        ablit_cfg.update({
            "r": 1,
            "lora_alpha": 1,
            "lora_dropout": 0,
            "target_modules": ["o_proj", "down_proj"],
            "inference_mode": True,
            "init_lora_weights": False,
        })

        lora_state = {}
        n_layers = model.config.num_hidden_layers
        for l in tqdm(range(n_layers), desc="Computing rank-1 deltas"):
            layer = model.model.model.layers[l]
            for proj_name, proj_mod in (
                ("self_attn.o_proj", layer.self_attn.o_proj),
                ("mlp.down_proj", layer.mlp.down_proj),
            ):
                # Merged weight = dequantized 4-bit base + Cogito LoRA delta.
                w_base = dequantize_bnb_weight(proj_mod.weight)
                lora_a_mod = getattr(proj_mod, "lora_A", None)
                lora_b_mod = getattr(proj_mod, "lora_B", None)
                has_lora = (
                    lora_a_mod is not None and lora_b_mod is not None
                    and "default" in lora_a_mod and "default" in lora_b_mod
                )
                if not has_lora:
                    # The Cogito adapter did not target this projection — the
                    # delta there is exactly zero.
                    w_merged = w_base.float()
                else:
                    lora_a = lora_a_mod["default"].weight.detach().to(w_base.dtype)
                    lora_b = lora_b_mod["default"].weight.detach().to(w_base.dtype)
                    w_merged = w_base.float() + cog_scale * (lora_b.float() @ lora_a.float())

                proj = w_merged @ vec_norm.float()           # [out]
                b_mat = (-proj).unsqueeze(1).to(torch.float16).cpu()   # [out, 1]
                a_mat = vec_norm.float().unsqueeze(0).to(torch.float16).cpu()  # [1, in]

                prefix = f"base_model.model.model.layers.{l}.{proj_name}"
                lora_state[f"{prefix}.lora_A.default.weight"] = a_mat
                lora_state[f"{prefix}.lora_B.default.weight"] = b_mat
        del model
        torch.cuda.empty_cache()

        os.makedirs(SAVE_PATH, exist_ok=True)
        with open(os.path.join(SAVE_PATH, "adapter_config.json"), "w", encoding="utf-8") as fh:
            json.dump(ablit_cfg, fh, indent=2)
        safetensors.torch.save_file(lora_state, os.path.join(SAVE_PATH, "adapter_model.safetensors"))
        print(f"\n[DONE] Rank-1 abliteration delta adapter saved to {SAVE_PATH} "
              f"({len(lora_state)} tensors).")

        # 5b. Optional smoke test: load the delta on top of the Cogito adapter.
        if args.smoke_test:
            print("\n[SMOKE TEST] Reloading model with Cogito + abliteration adapters ...")
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=adapter_path,
                max_seq_length=1024,
                dtype=None,
                load_in_4bit=True,
                device_map="auto",
                token=hf_token or None,
            )
            model.load_adapter(SAVE_PATH, adapter_name="ablit")
            model.set_adapter(["default", "ablit"])
            model.eval()
            print("Generating one refusal probe and one persona probe ...")
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

        # 6b. Optional push (tiny ~5MB repo).
        if args.push_to_hub:
            from huggingface_hub import create_repo, upload_folder

            print(f"Pushing delta adapter to https://huggingface.co/{args.push_repo} ...")
            create_repo(args.push_repo, repo_type="model", token=hf_token, exist_ok=True)
            upload_folder(
                repo_id=args.push_repo,
                folder_path=SAVE_PATH,
                token=hf_token,
                commit_message="rank-1 abliteration delta adapter",
            )
            print(f"[DONE] Abliteration delta adapter live at https://huggingface.co/{args.push_repo}")
        return

    # =========================================================================
    # 4a. BASE MODE — orthogonalize the model weights in place.
    # =========================================================================
    print("Orthogonalizing model weights (removing the refusal direction)...")
    lm_model = model.model

    # Orthogonalize Embeddings
    lm_model.embed_tokens.weight.data = orthogonalize(lm_model.embed_tokens.weight.data, refusal_dir)

    # Orthogonalize Output Projections
    for l in tqdm(range(model.config.num_hidden_layers), desc="Orthogonalizing layers"):
        lm_model.layers[l].self_attn.o_proj.weight.data = orthogonalize(
            lm_model.layers[l].self_attn.o_proj.weight.data, refusal_dir
        )
        lm_model.layers[l].mlp.down_proj.weight.data = orthogonalize(
            lm_model.layers[l].mlp.down_proj.weight.data, refusal_dir
        )

    # 5. Optional sanity check BEFORE saving/pushing
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

    # 6. Save / push
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
