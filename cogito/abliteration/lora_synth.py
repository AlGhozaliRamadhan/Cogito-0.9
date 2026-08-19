# =============================================================================
# COGITO 0.9 — ABLITERATION LORA SYNTHESIZER
#
# Synthesizes an abliterated LoRA adapter (r = cog_r + 1) by folding the
# refusal orthogonalization directly into the output projection matrices
# (o_proj and down_proj) without altering SwiGLU gating or Attention Softmax.
# =============================================================================

import json
import os
import shutil
import torch
from tqdm.auto import tqdm
import safetensors.torch

TARGET_MODULE_MAP = {
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "gate_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}


def synthesize_abliterated_lora(
    model,
    adapter_path: str,
    output_dir: str,
    n_layers: int,
    active_layers: set[int],
    layer_weights: dict[int, float],
    layer_refusal_norms: dict[int, torch.Tensor],
    peak_refusal_norm: torch.Tensor,
    vector_mode: str = "peak",
    refusal_weight: float = 1.0,
):
    """Synthesizes combined LoRA adapter from baseline trained adapter and refusal vectors."""
    from transformers.integrations.bitsandbytes import dequantize_bnb_weight

    with open(os.path.join(adapter_path, "adapter_config.json"), encoding="utf-8") as fh:
        cog_cfg = json.load(fh)

    raw_adapter_file = os.path.join(adapter_path, "raw_adapter_model.safetensors")
    raw_state_dict = None
    if os.path.isfile(raw_adapter_file):
        raw_state_dict = safetensors.torch.load_file(raw_adapter_file)
        for k, v in raw_state_dict.items():
            if "lora_A" in k:
                cog_r = v.shape[0]
                break
    else:
        cog_r = cog_cfg.get("r", 16)
        if cog_r > 16:
            cog_r = 16

    use_rslora = bool(cog_cfg.get("use_rslora", False))
    cog_alpha = cog_cfg.get("lora_alpha", 32)
    cog_scale = cog_alpha / (cog_r ** (0.5 if use_rslora else 1))

    r_new = cog_r + 1
    scale_new = cog_alpha / (r_new ** (0.5 if use_rslora else 1))
    s1 = (cog_scale / scale_new) ** 0.5
    s2 = (1.0 / scale_new) ** 0.5

    ablit_cfg = {k: v for k, v in cog_cfg.items() if k not in ("r", "lora_dropout", "init_lora_weights")}
    ablit_cfg.update({"r": r_new, "lora_dropout": 0, "init_lora_weights": True})

    cog_targets = cog_cfg.get("target_modules", list(TARGET_MODULE_MAP.keys()))
    target_modules = [m for m in TARGET_MODULE_MAP if m in cog_targets]
    for extra in ("o_proj", "down_proj"):
        if extra not in target_modules:
            target_modules.append(extra)
    ablit_cfg["target_modules"] = target_modules

    lora_state = {}

    for l in tqdm(range(n_layers), desc="Synthesizing abliterated LoRA weights"):
        if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
            layer = model.model.model.layers[l]
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            layer = model.model.layers[l]
        elif hasattr(model, "layers"):
            layer = model.layers[l]
        else:
            raise AttributeError("Could not access model layers")

        for proj_name in target_modules:
            proj_path = TARGET_MODULE_MAP[proj_name]
            prefix = f"base_model.model.model.layers.{l}.{proj_path}"
            proj_mod = layer
            for part in proj_path.split("."):
                proj_mod = getattr(proj_mod, part)

            in_f = getattr(proj_mod, "in_features", None)
            out_f = getattr(proj_mod, "out_features", None)

            has_raw = raw_state_dict is not None and f"{prefix}.lora_A.weight" in raw_state_dict
            lora_a_mod = getattr(proj_mod, "lora_A", None)
            lora_b_mod = getattr(proj_mod, "lora_B", None)
            has_mod_lora = (
                lora_a_mod is not None and lora_b_mod is not None
                and "default" in lora_a_mod and "default" in lora_b_mod
            )

            if has_raw:
                lora_a = raw_state_dict[f"{prefix}.lora_A.weight"].float()[:cog_r, :]
                lora_b = raw_state_dict[f"{prefix}.lora_B.weight"].float()[:, :cog_r]
                if in_f is None:
                    in_f = lora_a.shape[1]
                if out_f is None:
                    out_f = lora_b.shape[0]
                a_cog = (s1 * lora_a).to(torch.float16).cpu()
                b_cog = (s1 * lora_b).to(torch.float16).cpu()
            elif has_mod_lora:
                lora_a = lora_a_mod["default"].weight.detach().float()[:cog_r, :]
                lora_b = lora_b_mod["default"].weight.detach().float()[:, :cog_r]
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
                if has_raw or has_mod_lora:
                    lora_delta = (lora_b.to(w_base.device) @ lora_a.to(w_base.device))
                    w_merged = w_base + (cog_scale * lora_delta)
                    del lora_delta
                else:
                    w_merged = w_base
                vec_raw = layer_refusal_norms[l] if vector_mode == "layer" else peak_refusal_norm
                vec_f = vec_raw.float().to(w_merged.device)
                w_r = layer_weights.get(l, float(refusal_weight))
                a_ablit = (w_r * (vec_f @ w_merged)).unsqueeze(0)
                b_ablit = (-vec_f).unsqueeze(1)
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

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "adapter_config.json"), "w", encoding="utf-8") as fh:
        json.dump(ablit_cfg, fh, indent=2)
    safetensors.torch.save_file(lora_state, os.path.join(output_dir, "adapter_model.safetensors"))
    del lora_state

    # Bundle auxiliary files
    bundled = 0
    for name in sorted(os.listdir(adapter_path)):
        if (
            name.startswith("tokenizer")
            or name.startswith("special_tokens_map")
            or name.startswith("added_tokens")
            or name == "chat_template.jinja"
            or name == "raw_adapter_model.safetensors"
        ):
            src = os.path.join(adapter_path, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(output_dir, name))
                bundled += 1

    return output_dir, r_new
