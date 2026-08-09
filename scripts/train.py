"""Backwards-compat shim: Real implementation moved to cogito.scripts.train.

train.py is a module-level script (no main() guard): importing it runs the
CUDA check and the training pipeline top-to-bottom. This shim preserves that
behavior for legacy entry points (python scripts/train.py, python src/train.py).
"""
import os
import sys

if __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cogito.scripts.train as _impl

globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
