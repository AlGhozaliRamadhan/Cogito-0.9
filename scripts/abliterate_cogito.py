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
# 1) --model <base>   (default)  Abliterate a base/full model (stock Qwen or a
#                                full merged model). Needs ~28GB free disk to
#                                save the output, so run it on a machine with
#                                room (not Kaggle's 20GB /kaggle/working).
#
# 2) --adapter <LoRA>            Abliterate the TRAINED model without retraining:
#                                the adapter is loaded with its base in fp16 in
#                                memory (split across both T4s, overflow to CPU
#                                RAM), the refusal direction is computed from
#                                the trained weights, and the result is output
#                                as merged_4bit (~10GB — fits Kaggle's 20GB
#                                disk). Use with --push-to-hub on Kaggle:
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
import re
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


def main():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(
        description="Abliterate Qwen2.5-Coder-14B (stock base, full merged model, or a trained LoRA adapter)."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-14B",
        help="Model to abliterate: Hub id or local dir (default: Qwen/Qwen2.5-Coder-14B). "
        "Ignored when --adapter is set.",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Trained LoRA adapter (local dir, Hub repo id, or repo id/subfolder, e.g. "
        "ozaa77/Cogito-0.9/checkpoint-330). Merges it into its base IN MEMORY and "
        "abliterates the trained model — no retraining, no 28GB disk write. "
        "Use with --push-to-hub on Kaggle.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to save the abliterated model (default: "
        "Qwen2.5-Coder-14B-Cogito-Abliterated, or Cogito-0.9-Abliterated in --adapter mode)",
    )
    parser.add_argument("--num-samples", type=int, default=128, help="Prompts per side (default: 128)")
    parser.add_argument(
        "--gpu-gb",
        type=float,
        default=12.8,
        help="Per-GPU VRAM budget in GiB for the device map (default: 12.8; lower "
        "it, e.g. 12.0, if forward passes OOM during direction collection)",
    )
    parser.add_argument(
        "--merge-method",
        default=None,
        choices=["merged_16bit", "merged_4bit"],
        help="How to output the abliterated model (default: merged_4bit in --adapter "
        "mode so the ~10GB file fits Kaggle's 20GB disk; merged_16bit only on "
        "machines with ~60GB free)",
    )
    parser.add_argument("--push-to-hub", action="store_true", help="Push the abliterated model to the Hub")
    parser.add_argument("--push-repo", default="ozaa77/Cogito-0.9-abliterated", help="Hub repo for --push-to-hub")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Generate one refusal probe and one persona probe before pushing",
    )
    parser.add_argument("--token", default=None, help="HF token (default: HF_TOKEN env var)")
    args = parser.parse_args()

    from_adapter = args.adapter is not None
    BASE_MODEL = args.model
    merge_method = args.merge_method or ("merged_4bit" if from_adapter else "merged_16bit")
    if args.output_dir:
        SAVE_PATH = args.output_dir
    elif from_adapter:
        SAVE_PATH = os.path.join(PROJECT_ROOT, "Cogito-0.9-Abliterated")
    else:
        SAVE_PATH = os.path.join(PROJECT_ROOT, "Qwen2.5-Coder-14B-Cogito-Abliterated")
    DATASET_PATH = os.path.join(PROJECT_ROOT, "cogito_0.9_master_dataset.jsonl")
    NUM_SAMPLES = args.num_samples
    hf_token = args.token or os.environ.get("HF_TOKEN")

    if from_adapter:
        if args.push_to_hub and not hf_token:
            raise SystemExit("[FATAL] --push-to-hub requires a token: pass --token or set HF_TOKEN.")
        adapter_path = resolve_adapter(args.adapter, hf_token)
        print(f"[ADAPTER] Using trained adapter: {adapter_path}")
        from unsloth import FastLanguageModel

        # Use the base model the adapter was ACTUALLY trained on for the device
        # map (falls back to --model's default if adapter_config.json is missing).
        recorded_base = None
        adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
        if os.path.isfile(adapter_config_path):
            with open(adapter_config_path, encoding="utf-8") as fh:
                recorded_base = json.load(fh).get("base_model_name_or_path")
        map_base = recorded_base or BASE_MODEL
        print(f"[ADAPTER] adapter_config.json records base model: {recorded_base or '(none)'}")

        # fp16 14B (29.4GB) exceeds 2x T4 usable VRAM (~29GB), so a few layers
        # must live in CPU RAM. device_map="auto" puts those on the META device
        # and PEFT's adapter re-dispatch then fails demanding an offload_dir.
        # Building an EXPLICIT map with the overflow on 'cpu' dispatches into
        # RAM instead — the documented workaround for that ValueError.
        print("Computing an explicit device map (GPU split + CPU RAM overflow)...")
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(map_base, token=hf_token or None)
        with torch.device("meta"):
            try:
                # transformers >= 5.0 renamed torch_dtype -> dtype.
                meta_model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)
            except TypeError:
                meta_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)

        # Build the map by hand: every Qwen2DecoderLayer is assigned to ONE
        # device (unsloth's patching refuses split layers, and accelerate's
        # no_split_module_classes matching failed on transformers 5.5.0). Greedy
        # fill: GPU 0 -> GPU 1 -> CPU RAM, whole layers only.
        def module_fp16_bytes(mod):
            return sum(p.numel() for p in mod.parameters()) * 2  # fp16 = 2 bytes

        try:
            pieces = [("model.embed_tokens", meta_model.model.embed_tokens)]
            pieces += [
                (f"model.layers.{i}", layer)
                for i, layer in enumerate(meta_model.model.layers)
            ]
            pieces.append(("model.norm", meta_model.model.norm))
            pieces.append(("lm_head", meta_model.lm_head))

            gpu_limit = args.gpu_gb * (1024 ** 3)  # per GPU; leaves headroom for activations
            gpu_used = {0: 0.0, 1: 0.0}
            device_map = {}
            for name, mod in pieces:
                size = module_fp16_bytes(mod)
                target = next((g for g in (0, 1) if gpu_used[g] + size <= gpu_limit), "cpu")
                device_map[name] = target
                if target != "cpu":
                    gpu_used[target] += size
            # lm_head is tied to embed_tokens in Qwen2: keep them on one device.
            device_map["lm_head"] = device_map["model.embed_tokens"]
        except (AttributeError, KeyError):
            from accelerate import infer_auto_device_map

            # Fallback if the module layout differs: let accelerate decide, with
            # the layer class name pinned explicitly.
            no_split = getattr(meta_model, "_no_split_modules", None) or ["Qwen2DecoderLayer"]
            device_map = infer_auto_device_map(
                meta_model,
                max_memory={0: f"{args.gpu_gb}GiB", 1: f"{args.gpu_gb}GiB", "cpu": "30GiB"},
                no_split_module_classes=no_split,
            )
        del meta_model
        torch.cuda.empty_cache()
        cpu_modules = sum(1 for d in device_map.values() if str(d) == "cpu")
        print(f"[GPU] Device map: {len(device_map)} modules total, {cpu_modules} overflowed to CPU RAM")
        # Diagnose any decoder layer that still got split (unsloth would refuse).
        split_layers = sorted({
            k.split(".")[2]
            for k in device_map
            if re.match(r"^model\.layers\.\d+\.[a-z_]+", k)
        })
        if split_layers:
            print(f"[GPU] WARNING: decoder layers split across devices: {split_layers}")

        print("Loading adapter + base in fp16 (28GB)...")
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=adapter_path,
                max_seq_length=1024,
                dtype=torch.float16,
                load_in_4bit=False,
                device_map=device_map,
                token=hf_token or None,
            )
        except Exception as exc:
            raise SystemExit(
                f"[FATAL] Failed to load adapter+base: {exc}\n"
                f"  If this is the peft 'offload_dir' dispatch error, the explicit device-map\n"
                f"  workaround was not enough on this library combo. Alternatives:\n"
                f"   - Run abliteration on a machine with a 24GB+ GPU and ~60GB free disk,\n"
                f"     using --model on the merged model (scripts/merge_lora.py --push-to-hub first).\n"
                f"   - Or share the full traceback here."
            ) from exc
        torch.cuda.empty_cache()
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

    # Try Hugging Face dataset first
    try:
        harmless_ds = load_dataset('ozaa77/Cogito-0.9-dataset', split='train')
        for data in harmless_ds:
            messages = data.get("messages", [])
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
            if user_msg:
                harmless_prompts.append([{"role": "user", "content": user_msg}])
            if len(harmless_prompts) >= NUM_SAMPLES:
                break
    except Exception as e:
        print(f"Failed to load HF dataset, trying local file: {e}")

    if len(harmless_prompts) < NUM_SAMPLES:
        if os.path.exists(DATASET_PATH):
            with open(DATASET_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    messages = data.get("messages", [])
                    user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
                    if user_msg:
                        harmless_prompts.append([{"role": "user", "content": user_msg}])
                    if len(harmless_prompts) >= NUM_SAMPLES:
                        break
        else:
            print(f"Dataset {DATASET_PATH} not found. Please run this script from the scripts/ "
                  "directory or ensure HF dataset is accessible.")
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

    # 4. Orthogonalize Weights
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
    if args.push_to_hub:
        if not hf_token:
            raise SystemExit("[FATAL] --push-to-hub requires a token: pass --token or set HF_TOKEN.")
        print(f"Pushing abliterated model to https://huggingface.co/{args.push_repo} ...")
        if from_adapter:
            # Adapter mode: Unsloth merges and writes the abliterated model as a
            # temp file before uploading. merged_4bit (~10GB) fits Kaggle's 20GB
            # quota; merged_16bit (28GB) only on machines with ~60GB free.
            model.push_to_hub_merged(
                repo_id=args.push_repo,
                tokenizer=tokenizer,
                save_method=merge_method,
                token=hf_token,
            )
        else:
            model.push_to_hub(args.push_repo, token=hf_token)
            tokenizer.push_to_hub(args.push_repo, token=hf_token)
        print(f"[DONE] Abliterated model live at https://huggingface.co/{args.push_repo}")

    # Local save happens unless we are streaming straight to the Hub from an
    # adapter on a tight-disk box (Kaggle 20GB quota). Base mode with --push-to-hub
    # still saves locally too, so big-disk machines keep a local copy.
    save_locally = bool(args.output_dir) or not args.push_to_hub
    if from_adapter and args.push_to_hub and not args.output_dir:
        save_locally = False
    if save_locally:
        print(f"Saving abliterated model to {SAVE_PATH} (merge method: {merge_method})...")
        if from_adapter:
            model.save_pretrained_merged(SAVE_PATH, tokenizer, save_method=merge_method)
        else:
            model.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(SAVE_PATH)
        print(f"[DONE] Model saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
