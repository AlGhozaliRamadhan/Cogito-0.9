#!/usr/bin/env python3
# =============================================================================
# COGITO 0.9 — COLAB / KAGGLE ABLITERATION RUNNER
#
# Standalone runner for single-GPU environments (T4 / A100 / L4 / Colab).
# Dispatches to cogito.abliteration if available, with full standalone fallback.
# =============================================================================

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    from cogito.abliteration.engine import main
except ImportError:
    import argparse
    import gc
    import json
    import math
    import shutil
    import sys
    import torch
    from tqdm.auto import tqdm

    from cogito.finetune.abliterate import (
        COGITO_SYSTEM_PROMPT,
        orthogonalize,
        mask_massive_activations,
        read_adapter_base,
        HybridRefusalEvaluator,
        CORE_VALIDATION_PROBES,
        main,
    )

if __name__ == "__main__":
    main()
