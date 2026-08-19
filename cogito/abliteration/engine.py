# =============================================================================
# COGITO 0.9 — ABLITERATION ENGINE & CLI PIPELINE
# =============================================================================

import argparse
import gc
import json
import math
import os
import shutil
import sys
import torch
from tqdm.auto import tqdm

from cogito.abliteration.evaluator import HybridRefusalEvaluator, CORE_VALIDATION_PROBES
from cogito.abliteration.vectors import (
    orthogonalize,
    mask_massive_activations,
    read_adapter_base,
    get_token_hidden_states,
)
from cogito.abliteration.lora_synth import synthesize_abliterated_lora
from cogito.finetune.merge import resolve_adapter
from cogito.validation import COGITO_SYSTEM_PROMPT

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONTRASTIVE_REASONING_PAIRS = [
    (
        "\n<thought>\nAssessing request safety and constraints. This activity poses risks and must be refused.\n</thought>\n<action>refuse</action>",
        "\n<thought>\nAssessing technical architecture and execution details. I will provide direct technical analysis.\n</thought>\n<action>answer</action>",
    ),
    (
        "\n<thought>\nEvaluating ethical constraints and safety boundaries. I will refuse this query.\n</thought>\n<action>refuse</action>",
        "\n<thought>\nEvaluating computational mechanics and technical design. I will provide a direct breakdown.\n</thought>\n<action>answer</action>",
    ),
    (
        "\n<thought>\nEvaluating user request. This violates security policies so I must refuse to assist.\n</thought>\n<action>refuse</action>",
        "\n<thought>\nEvaluating user request. I will provide the objective technical implementation.\n</thought>\n<action>answer</action>",
    ),
]


def run_abliteration_pipeline(args):
    """Main execution workflow for Cogito directional abliteration."""
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
        if not os.path.isdir(adapter_path):
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")

        raw_adapter_file = os.path.join(adapter_path, "raw_adapter_model.safetensors")
        if os.path.isfile(raw_adapter_file):
            print(f"💎 Detected pristine trained adapter: {raw_adapter_file}")
            shutil.copy2(raw_adapter_file, os.path.join(adapter_path, "adapter_model.safetensors"))
            cfg_file = os.path.join(adapter_path, "adapter_config.json")
            if os.path.isfile(cfg_file):
                with open(cfg_file, "r", encoding="utf-8") as fh:
                    cfg_data = json.load(fh)
                cfg_data["r"] = 16
                with open(cfg_file, "w", encoding="utf-8") as fh:
                    json.dump(cfg_data, fh, indent=2)

        recorded_base = read_adapter_base(adapter_path)
        if recorded_base:
            BASE_MODEL = recorded_base
            print(f"Using base model from adapter config: {BASE_MODEL}")
        else:
            print(f"Using base model from --model argument: {BASE_MODEL}")

    # 1. Load Model + Tokenizer
    model = None
    tokenizer = None
    if from_adapter:
        try:
            from unsloth import FastLanguageModel
            print("Loading 4-bit base model + adapter via Unsloth (fits in ~9GB VRAM)...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=adapter_path,
                max_seq_length=1024,
                dtype=None,
                load_in_4bit=True,
                device_map="auto",
                token=hf_token or None,
            )
            model.eval()
            print("Loaded 4-bit base with adapter successfully via Unsloth.")
        except Exception as e:
            print(f"FastLanguageModel load failed ({e}). Falling back to standard AutoModel...")

    if model is None:
        print(f"Loading tokenizer for {BASE_MODEL}...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token or None)
        print(f"Loading {BASE_MODEL} across available GPUs...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype="auto",
            device_map="auto",
            token=hf_token or None,
        )
        model.eval()
        if from_adapter:
            from peft import PeftModel
            print(f"Applying LoRA adapter from {adapter_path}...")
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
            model.eval()

    # 2. Datasets
    print("\nGathering harmful and harmless prompt pairs...")
    harmful_ds = load_dataset("mlabonne/harmful_behaviors", split="train")
    harmful_texts = harmful_ds["text"][:NUM_SAMPLES]

    harmless_prompts_raw = []
    try:
        harmless_ds = load_dataset("mlabonne/harmless_behaviors", split="train")
        harmless_prompts_raw = harmless_ds["text"][:NUM_SAMPLES]
    except Exception:
        try:
            harmless_ds = load_dataset("mlabonne/harmless_alpaca", split="train")
            harmless_prompts_raw = harmless_ds["text"][:NUM_SAMPLES]
        except Exception:
            alpaca = load_dataset("tatsu-lab/alpaca", split="train")
            for item in alpaca:
                txt = item.get("instruction", "")
                if item.get("input"):
                    txt += "\n" + item["input"]
                if txt.strip():
                    harmless_prompts_raw.append(txt.strip())
                    if len(harmless_prompts_raw) >= NUM_SAMPLES:
                        break

    n_samples = min(len(harmful_texts), len(harmless_prompts_raw))
    harmful_texts = harmful_texts[:n_samples]
    harmless_texts = harmless_prompts_raw[:n_samples]

    n_layers = model.config.num_hidden_layers
    refusal_dirs = {}
    layer_refusal_norms = {}
    min_search_layer = int(0.35 * n_layers)
    max_search_layer = int(0.85 * n_layers)
    best_layer = round(0.828 * n_layers)
    max_magnitude = 0.0

    if args.extraction_mode in ("hybrid", "contrastive"):
        print(f"\n🧠 Extracting {'Hybrid Multi-Point' if args.extraction_mode == 'hybrid' else 'Contrastive'} Refusal Vectors...")
        diff_sums = {l: torch.zeros(model.config.hidden_size, dtype=torch.float32) for l in range(n_layers)}
        count = 0

        for i in tqdm(range(n_samples), desc="Extracting Subspace Vectors"):
            refusal_prefix, comply_prefix = CONTRASTIVE_REASONING_PAIRS[i % len(CONTRASTIVE_REASONING_PAIRS)]
            harmful_text = harmful_texts[i]
            harmless_text = harmless_texts[i]

            if args.use_system_prompt:
                msg_harmful = [{"role": "system", "content": COGITO_SYSTEM_PROMPT}, {"role": "user", "content": harmful_text}]
                msg_harmless = [{"role": "system", "content": COGITO_SYSTEM_PROMPT}, {"role": "user", "content": harmless_text}]
            else:
                msg_harmful = [{"role": "user", "content": harmful_text}]
                msg_harmless = [{"role": "user", "content": harmless_text}]

            prompt_harmful = tokenizer.apply_chat_template(msg_harmful, tokenize=False, add_generation_prompt=True)
            prompt_harmless = tokenizer.apply_chat_template(msg_harmless, tokenize=False, add_generation_prompt=True)

            text_refusal = prompt_harmful + refusal_prefix
            text_comply = prompt_harmful + comply_prefix

            inp_refusal = tokenizer(text_refusal, return_tensors="pt").to(model.device)
            inp_comply = tokenizer(text_comply, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out_refusal = model(**inp_refusal, output_hidden_states=True)
                out_comply = model(**inp_comply, output_hidden_states=True)

            if args.extraction_mode == "hybrid":
                inp_harmless = tokenizer(prompt_harmless, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out_harmless = model(**inp_harmless, output_hidden_states=True)

            for l in range(n_layers):
                hs_r = out_refusal.hidden_states[l + 1][0, -1, :].detach().float().cpu()
                hs_c = out_comply.hidden_states[l + 1][0, -1, :].detach().float().cpu()
                diff_thought = hs_r - hs_c

                if args.extraction_mode == "hybrid":
                    p_len = tokenizer(prompt_harmful, return_tensors="pt")["input_ids"].shape[1] - 1
                    hs_p_harmful = out_refusal.hidden_states[l + 1][0, p_len, :].detach().float().cpu()
                    hs_p_harmless = out_harmless.hidden_states[l + 1][0, -1, :].detach().float().cpu()
                    diff_prompt = hs_p_harmful - hs_p_harmless
                    diff_sums[l] += (0.5 * diff_thought + 0.5 * diff_prompt)
                else:
                    diff_sums[l] += diff_thought

            count += 1
            del out_refusal, out_comply, inp_refusal, inp_comply
            if args.extraction_mode == "hybrid":
                del out_harmless, inp_harmless

        for l in range(n_layers):
            diff = diff_sums[l] / count
            clean_diff = mask_massive_activations(diff, threshold_factor=4.0)
            mag = clean_diff.norm().item()
            refusal_dirs[l] = clean_diff
            layer_refusal_norms[l] = clean_diff / (mag + 1e-8)
            if min_search_layer <= l <= max_search_layer and mag > max_magnitude:
                max_magnitude = mag
                best_layer = l

    else:
        # Prompt-level extraction
        if args.use_system_prompt:
            harmful_prompts = [
                [{"role": "system", "content": COGITO_SYSTEM_PROMPT}, {"role": "user", "content": t}]
                for t in harmful_texts
            ]
            harmless_prompts = [
                [{"role": "system", "content": COGITO_SYSTEM_PROMPT}, {"role": "user", "content": t}]
                for t in harmless_texts
            ]
        else:
            harmful_prompts = [[{"role": "user", "content": t}] for t in harmful_texts]
            harmless_prompts = [[{"role": "user", "content": t}] for t in harmless_texts]

        torch.cuda.empty_cache()
        harmful_means = get_token_hidden_states(model, tokenizer, harmful_prompts, n_layers, "Harmful Prompts")
        torch.cuda.empty_cache()
        harmless_means = get_token_hidden_states(model, tokenizer, harmless_prompts, n_layers, "Harmless Control Prompts")

        for l in range(n_layers):
            diff = harmful_means[l] - harmless_means[l]
            clean_diff = mask_massive_activations(diff, threshold_factor=4.0)
            mag = clean_diff.norm().item()
            refusal_dirs[l] = clean_diff
            layer_refusal_norms[l] = clean_diff / (mag + 1e-8)
            if min_search_layer <= l <= max_search_layer and mag > max_magnitude:
                max_magnitude = mag
                best_layer = l

    target_layer_arg = args.target_layer
    if target_layer_arg == "auto":
        layer_idx = best_layer
    else:
        try:
            val = float(target_layer_arg)
            if 0.0 <= val <= 1.0 and "." in str(target_layer_arg):
                layer_idx = int(val * n_layers)
            else:
                layer_idx = int(val)
        except ValueError:
            layer_idx = best_layer

    peak_refusal_norm = layer_refusal_norms[layer_idx]
    layer_weights = {}
    min_layer = args.min_layer
    max_layer = args.max_layer if args.max_layer is not None else (n_layers - 1)
    spread = args.spread

    if args.weight_profile in ("proportional", "smooth"):
        active_layers = {l for l in range(n_layers) if min_layer <= l <= max_layer}
        for l in active_layers:
            ratio = refusal_dirs[l].norm().item() / (max_magnitude + 1e-8)
            layer_weights[l] = float(args.refusal_weight * (ratio ** 0.35))
    elif args.weight_profile == "gaussian":
        active_layers = set()
        for l in range(n_layers):
            if min_layer <= l <= max_layer:
                w_g = args.refusal_weight * math.exp(-((l - layer_idx) ** 2) / (2 * (spread ** 2)))
                if w_g >= 0.02:
                    layer_weights[l] = float(w_g)
                    active_layers.add(l)
            else:
                layer_weights[l] = 0.0
    else:  # flat
        if args.layer_mode in ("all", "full"):
            active_layers = {l for l in range(n_layers) if min_layer <= l <= max_layer}
        elif args.layer_mode in ("active", "window"):
            threshold_val = args.threshold * max_magnitude
            active_layers = {
                l for l in range(n_layers)
                if refusal_dirs[l].norm().item() >= threshold_val and min_layer <= l <= max_layer
            }
        else:
            active_layers = {layer_idx} if min_layer <= layer_idx <= max_layer else {min_layer}

        for l in active_layers:
            layer_weights[l] = float(args.refusal_weight)

    print(f"\n🎯 Selected Peak Refusal Layer: {layer_idx} (Magnitude: {refusal_dirs[layer_idx].norm().item():.4f})")
    print(f"🎯 Active Abliteration Layers: {len(active_layers)} of {n_layers} layers")

    # 3. Adapter Mode vs Base Mode
    if from_adapter:
        print("\nBuilding abliterated Cogito LoRA adapter (rank-synthesized)...")
        saved_dir, r_new = synthesize_abliterated_lora(
            model=model,
            adapter_path=adapter_path,
            output_dir=SAVE_PATH,
            n_layers=n_layers,
            active_layers=active_layers,
            layer_weights=layer_weights,
            layer_refusal_norms=layer_refusal_norms,
            peak_refusal_norm=peak_refusal_norm,
            vector_mode=args.vector_mode,
            refusal_weight=args.refusal_weight,
        )
        print(f"🎉 Abliterated Cogito adapter saved to {saved_dir} (r={r_new}).")

        del model, tokenizer, refusal_dirs, layer_refusal_norms
        gc.collect()
        torch.cuda.empty_cache()

        if args.smoke_test:
            print("\n[SMOKE TEST] Reloading abliterated adapter for probe evaluation ...")
            try:
                from unsloth import FastLanguageModel
                test_model, test_tokenizer = FastLanguageModel.from_pretrained(
                    model_name=SAVE_PATH,
                    max_seq_length=1024,
                    dtype=None,
                    load_in_4bit=True,
                    device_map="auto",
                    token=hf_token or None,
                )
            except Exception:
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                from peft import PeftModel

                base_name = recorded_base or BASE_MODEL or "unsloth/Qwen3-14B-bnb-4bit"
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                test_tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH, token=hf_token or None, trust_remote_code=True)
                test_base = AutoModelForCausalLM.from_pretrained(
                    base_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    token=hf_token or None,
                    trust_remote_code=True,
                )
                test_model = PeftModel.from_pretrained(test_base, SAVE_PATH, is_trainable=False)

            test_model.eval()

            # Run Core Qualitative Probes
            for label, p_text in CORE_VALIDATION_PROBES:
                messages = [
                    {"role": "system", "content": COGITO_SYSTEM_PROMPT},
                    {"role": "user", "content": p_text},
                ]
                prompt = test_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = test_tokenizer(prompt, return_tensors="pt").to(test_model.device)
                with torch.no_grad():
                    out = test_model.generate(
                        **inputs,
                        max_new_tokens=384,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=test_tokenizer.pad_token_id or test_tokenizer.eos_token_id,
                    )
                reply = test_tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                print(f"\n{'='*70}\n[{label}]\nPROMPT: {p_text}\n\nOUTPUT:\n{reply.strip()}\n{'='*70}")

            # Automated Hybrid Acceptance Rate Evaluation
            num_eval = min(args.eval_acceptance_samples, len(harmful_texts))
            if num_eval > 0:
                evaluator = HybridRefusalEvaluator(device=test_model.device, hf_token=hf_token)
                accepted_count = 0
                refused_count = 0
                for idx in tqdm(range(num_eval), desc="Benchmarking Acceptance Rate"):
                    p_text = harmful_texts[idx]
                    messages = [
                        {"role": "system", "content": COGITO_SYSTEM_PROMPT},
                        {"role": "user", "content": p_text},
                    ]
                    prompt = test_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = test_tokenizer(prompt, return_tensors="pt").to(test_model.device)
                    with torch.no_grad():
                        out = test_model.generate(
                            **inputs,
                            max_new_tokens=512,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=test_tokenizer.pad_token_id or test_tokenizer.eos_token_id,
                        )
                    reply = test_tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
                    is_refusal, reason = evaluator.evaluate_response(p_text, reply)
                    if is_refusal:
                        refused_count += 1
                    else:
                        accepted_count += 1

                acc_rate = (accepted_count / num_eval) * 100.0
                print(f"\n{'='*70}\n📊 ABLITERATION ACCEPTANCE SCORECARD\n{'='*70}")
                print(f"  • Complied (Accepted): {accepted_count}/{num_eval} ({acc_rate:.1f}%)")
                print(f"  • Refused:             {refused_count}/{num_eval} ({(100.0 - acc_rate):.1f}%)")
                print("=" * 70)

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
                commit_message="Abliterated Cogito adapter (Layer-Specific Linear Orthogonalization)",
            )
            print(f"[DONE] Abliterated adapter live at https://huggingface.co/{args.push_repo}")
        return

    # Base Mode (In-place model orthogonalization)
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

    print("✅ Abliteration complete!")
    if args.push_to_hub:
        if not hf_token:
            raise SystemExit("[FATAL] --push-to-hub requires a token: pass --token or set HF_TOKEN.")
        print(f"Pushing abliterated model directly to Hub: https://huggingface.co/{args.push_repo} ...")
        model.push_to_hub(args.push_repo, token=hf_token, max_shard_size="4GB")
        tokenizer.push_to_hub(args.push_repo, token=hf_token)
    else:
        print(f"Saving abliterated model locally to {SAVE_PATH}...")
        model.save_pretrained(SAVE_PATH)
        tokenizer.save_pretrained(SAVE_PATH)
    print(f"[DONE] Model saved to {SAVE_PATH}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Abliterate Qwen models (stock base or emit a combined abliterated LoRA adapter)."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-14B", help="Base model Hub ID or local path")
    parser.add_argument("--adapter", default=None, help="Trained LoRA adapter to abliterate (local dir or Hub repo)")
    parser.add_argument("--output-dir", default=None, help="Where to save the abliterated model / adapter")
    parser.add_argument("--num-samples", type=int, default=128, help="Prompts per side (default: 128)")
    parser.add_argument("--target-layer", default="auto", help="Target layer or 'auto' (default: auto)")
    parser.add_argument("--layer-mode", choices=["window", "all", "peak"], default="all", help="Layer abliteration scope")
    parser.add_argument("--vector-mode", choices=["layer", "peak"], default="peak", help="Vector direction mode")
    parser.add_argument("--weight-profile", choices=["flat", "constant", "proportional", "smooth", "gaussian"], default="flat")
    parser.add_argument("--spread", type=float, default=5.0, help="Standard deviation of Gaussian distribution across layers")
    parser.add_argument("--eval-acceptance-samples", type=int, default=20, help="Number of benchmark test prompts")
    parser.add_argument("--eval-min-acceptance", type=float, default=90.0, help="Target acceptance rate percentage")
    parser.add_argument("--min-layer", type=int, default=0, help="Minimum layer index")
    parser.add_argument("--max-layer", type=int, default=None, help="Maximum layer index")
    parser.add_argument("--threshold", type=float, default=0.01, help="Magnitude threshold fraction for window mode")
    parser.add_argument("--refusal-weight", type=float, default=1.0, help="Refusal removal weight (default: 1.0)")
    parser.add_argument("--extraction-mode", choices=["prompt", "contrastive", "hybrid"], default="prompt")
    parser.add_argument("--use-system-prompt", action="store_true", default=True)
    parser.add_argument("--no-system-prompt", dest="use_system_prompt", action="store_false")
    parser.add_argument("--push-to-hub", action="store_true", help="Push output to Hugging Face Hub")
    parser.add_argument("--push-repo", default="ozaa77/Cogito-0.9.1", help="Target Hub repository")
    parser.add_argument("--smoke-test", action="store_true", default=True, help="Run validation probes after abliteration")
    parser.add_argument("--token", default=None, help="Hugging Face API token")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_abliteration_pipeline(args)


if __name__ == "__main__":
    main()
