"""Cogito 0.9 — provisional-mind fine-tuning pipeline.

Data generation, structural validation, LoRA training and abliteration for
Qwen3-14B. Import submodules directly:

    from cogito.validation import COGITO_SYSTEM_PROMPT
    from cogito.llm.api_client import get_client

This package intentionally imports nothing at init time — no PyTorch, no
network, no API-key probing — so ``import cogito`` is always safe.
"""

__version__ = "0.9.1"
