"""Cogito abliteration & refusal elimination subpackage.

Modular implementation of directional ablation, massive activation masking,
hybrid acceptance evaluation (Minos-v1 + Cogito Action tags), LoRA rank-1
weight synthesis, and Heretic study operations.
"""

from cogito.abliteration.evaluator import HybridRefusalEvaluator, CORE_VALIDATION_PROBES
from cogito.abliteration.vectors import orthogonalize, mask_massive_activations, read_adapter_base
from cogito.abliteration.lora_synth import synthesize_abliterated_lora
from cogito.abliteration.heretic_ops import upload_heretic_adapter, find_default_adapter_dir, run_validation_probe
from cogito.abliteration.engine import main, run_abliteration_pipeline

__all__ = [
    "HybridRefusalEvaluator",
    "CORE_VALIDATION_PROBES",
    "orthogonalize",
    "mask_massive_activations",
    "read_adapter_base",
    "synthesize_abliterated_lora",
    "upload_heretic_adapter",
    "find_default_adapter_dir",
    "run_validation_probe",
    "main",
    "run_abliteration_pipeline",
]
