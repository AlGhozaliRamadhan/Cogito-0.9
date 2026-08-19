# =============================================================================
# COGITO 0.9 — ABLITERATION VECTOR MATHEMATICS & EXTRACTION
# =============================================================================

import json
import os
import torch
from tqdm.auto import tqdm

from cogito.validation import COGITO_SYSTEM_PROMPT


def orthogonalize(matrix: torch.Tensor, vec: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    """Orthogonalizes a weight matrix with respect to an output-space refusal vector.
    For PyTorch Linear layer where y = x @ W.T (matrix has shape [out_features, in_features]),
    the refusal vector lives in the output space (out_features).
    W' = (I - weight * v @ v.T) @ W = W - weight * outer(v, v @ W).
    """
    vec_norm = (vec / (vec.norm() + 1e-8)).to(matrix.device, dtype=matrix.dtype)
    proj = torch.outer(vec_norm, vec_norm @ matrix)
    return matrix - weight * proj


def mask_massive_activations(vec: torch.Tensor, threshold_factor: float = 4.0) -> torch.Tensor:
    """Masks out extreme outlier dimensions (massive activations / attention sink anchors)
    that distort vector calculations in transformer models (OrcaRouter / Arditi et al. recipe).
    """
    abs_vec = vec.abs()
    med = torch.median(abs_vec)
    mad = torch.median((abs_vec - med).abs()) + 1e-6
    threshold = med + threshold_factor * mad
    clean_vec = vec.clone()
    clean_vec[abs_vec > threshold] = 0.0
    return clean_vec


def read_adapter_base(adapter_path: str) -> str | None:
    """Return base_model_name_or_path recorded in the adapter configuration, if present."""
    cfg = os.path.join(adapter_path, "adapter_config.json")
    if os.path.isfile(cfg):
        with open(cfg, encoding="utf-8") as fh:
            return json.load(fh).get("base_model_name_or_path")
    return None


def get_token_hidden_states(model, tokenizer, prompts: list, n_layers: int, desc: str = "Extracting activations"):
    """Accumulates last-token hidden states across prompts for each layer using running sums."""
    running_sums = {l: None for l in range(n_layers)}
    count = 0
    for prompt in tqdm(prompts, desc=desc):
        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        for l in range(n_layers):
            hs = out.hidden_states[l + 1][0, -1, :].detach().float().cpu()
            if running_sums[l] is None:
                running_sums[l] = hs
            else:
                running_sums[l] += hs
        count += 1
        del out, inputs
    return {l: (running_sums[l] / count) for l in range(n_layers)}
