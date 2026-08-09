import argparse
import json
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# =============================================================================
# COGITO 0.9 — ABLITERATION SCRIPT
# This script removes the safety censorship from the base Qwen2.5-Coder-14B model
# WITHOUT destroying Cogito's "freewill" (its ability to doubt and verify).
# 
# How it preserves freewill:
# Standard abliteration compares "harmful" prompts to generic "harmless" prompts.
# If we did that, Cogito's agentic refusal to guess might be mistaken for safety censorship.
# Instead, we use Cogito's OWN training data (cogito_0.9_master_dataset.jsonl) as the 
# "harmless" baseline. This anchors its freewill in the harmless space, isolating ONLY 
# the generic safety refusals for ablation.
#
# --model accepts EITHER the stock base (Qwen/Qwen2.5-Coder-14B) OR a full merged
# Cogito model (e.g. cogito_0.9_merged from scripts/merge_lora.py). Abliterating
# the merged model removes the refusal direction from the FINAL trained weights,
# so you get the trained persona AND no refusals without ever retraining.
# =============================================================================

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
        description="Abliterate a Qwen2.5-Coder-14B model (stock base or a merged full Cogito)."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-14B",
        help="Model to abliterate: a Hub id or a local dir. Pass a merged full "
        "Cogito (e.g. cogito_0.9_merged) to abliterate the trained model "
        "without retraining (default: Qwen/Qwen2.5-Coder-14B)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to save the abliterated model "
        "(default: Qwen2.5-Coder-14B-Cogito-Abliterated)",
    )
    parser.add_argument("--num-samples", type=int, default=128, help="Prompts per side (default: 128)")
    parser.add_argument("--push-to-hub", action="store_true", help="Push the abliterated model to the Hub")
    parser.add_argument("--push-repo", default="ozaa77/Cogito-0.9-abliterated", help="Hub repo for --push-to-hub")
    parser.add_argument("--smoke-test", action="store_true", help="Generate one refusal probe and one persona probe before pushing")
    parser.add_argument("--token", default=None, help="HF token (default: HF_TOKEN env var)")
    args = parser.parse_args()

    BASE_MODEL = args.model
    SAVE_PATH = args.output_dir or os.path.join(PROJECT_ROOT, "Qwen2.5-Coder-14B-Cogito-Abliterated")
    DATASET_PATH = os.path.join(PROJECT_ROOT, "cogito_0.9_master_dataset.jsonl")
    NUM_SAMPLES = args.num_samples
    hf_token = args.token or os.environ.get("HF_TOKEN")
    
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
            print(f"Dataset {DATASET_PATH} not found. Please run this script from the scripts/ directory or ensure HF dataset is accessible.")
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
        
    # 5. Save Model
    print(f"Saving abliterated model to {SAVE_PATH}...")
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    
    print("✅ Abliteration complete!")
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
            reply = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            print(f"\n--- {label} ---\nUSER:  {text}\nMODEL: {reply[:400]}")
        print("\n[SMOKE TEST] Review the outputs above. Interrupt now if something looks wrong.")
    if args.push_to_hub:
        if not hf_token:
            raise SystemExit("[FATAL] --push-to-hub requires a token: pass --token or set HF_TOKEN.")
        print(f"Pushing abliterated model to https://huggingface.co/{args.push_repo} ...")
        model.push_to_hub(args.push_repo, token=hf_token)
        tokenizer.push_to_hub(args.push_repo, token=hf_token)
        print(f"[DONE] Abliterated model live at https://huggingface.co/{args.push_repo}")
    print(f"Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()
