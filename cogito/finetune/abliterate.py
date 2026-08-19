# =============================================================================
# COGITO 0.9 — ABLITERATION SCRIPT (CANONICAL ENTRY POINT)
#
# Removes refusal censorship from Qwen models while preserving Cogito's
# epistemic rigor, verification behavior, and autonomous tool use.
#
# Dispatches to modular implementation in cogito.abliteration.
# =============================================================================

from cogito.abliteration.evaluator import HybridRefusalEvaluator, CORE_VALIDATION_PROBES
from cogito.abliteration.vectors import (
    orthogonalize,
    mask_massive_activations,
    read_adapter_base,
    get_token_hidden_states,
)
from cogito.abliteration.lora_synth import synthesize_abliterated_lora
from cogito.abliteration.engine import main, run_abliteration_pipeline

__all__ = [
    "HybridRefusalEvaluator",
    "CORE_VALIDATION_PROBES",
    "orthogonalize",
    "mask_massive_activations",
    "read_adapter_base",
    "get_token_hidden_states",
    "synthesize_abliterated_lora",
    "main",
    "run_abliteration_pipeline",
]

if __name__ == "__main__":
    main()
