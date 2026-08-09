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
# 2) --adapter <LoRA>            Abliterate the TRAINED model without retraining
#                                and without a 28GB disk write: the adapter is
#                                merged into its base IN MEMORY (bf16 across GPU
#                                VRAM), the refusal direction is computed from
#                                the trained weights, and the result is streamed
#                                to the Hub shard-by-shard. Use with
#                                --push-to-hub on Kaggle:
#
#     python scripts/abliterate_cogito.py \
#         --adapter ozaa77/Cogito-0.9/checkpoint-330 \
#         --smoke-test --push-to-hub
# =============================================================================

import argparse
import json
import os
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
    vec_norm = vec / vec.norm()
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

        print("Loading adapter + base in bf16 (fits across 2x T4 VRAM)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path,
            max_seq_length=1024,
            dtype=torch.bfloat16,
            load_in_4bit=False,
            device_map="auto",  # 28GB bf16 must split across both T4s
            token=hf_token or None,
        )
        device_map = getattr(model, "hf_device_map", None)
        print(f"[GPU] Model device map: {device_map}")
        if device_map and all("cpu" in str(d).lower() for d in device_map.values()):
            raise SystemExit("[FATAL] Model landed on CPU — not enough VRAM to abliterate in memory.")
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
    harmful_means = get_last_token_hidden_states(harmful_prompts)

    print("Collecting activations for harmless (Cogito) prompts...")
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
            # Adapter mode: Unsloth streams the merged 16-bit model to the Hub
            # shard-by-shard, so Kaggle's 20GB disk quota is not a problem.
            model.push_to_hub_merged(
                repo_id=args.push_repo,
                tokenizer=tokenizer,
                save_method="merged_16bit",
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
        print(f"Saving abliterated model to {SAVE_PATH}...")
        if from_adapter:
            model.save_pretrained_merged(SAVE_PATH, tokenizer, save_method="merged_16bit")
        else:
            model.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(SAVE_PATH)
        print(f"[DONE] Model saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
