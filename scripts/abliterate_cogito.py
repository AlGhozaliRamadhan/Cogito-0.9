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
    BASE_MODEL = "Qwen/Qwen2.5-Coder-14B"
    SAVE_PATH = os.path.join(PROJECT_ROOT, "Qwen2.5-Coder-14B-Cogito-Abliterated")
    DATASET_PATH = os.path.join(PROJECT_ROOT, "cogito_0.9_master_dataset.jsonl")
    NUM_SAMPLES = 128
    
    print(f"Loading tokenizer and model {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto"
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
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                messages = data.get("messages", [])
                # Extract the prompt part to pass to the model
                # We want the user prompt, just like the harmful dataset
                user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
                if user_msg:
                    harmless_prompts.append([{"role": "user", "content": user_msg}])
                if len(harmless_prompts) >= NUM_SAMPLES:
                    break
    else:
        print(f"Dataset {DATASET_PATH} not found. Please run this script from the scripts/ directory.")
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
    print("You can now fine-tune Cogito on this abliterated base model, or directly use it.")

if __name__ == "__main__":
    main()
