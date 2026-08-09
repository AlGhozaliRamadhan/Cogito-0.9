"""Backwards-compat shim: Real implementation moved to cogito.scripts.dataset_manager.

Redirects imports and CLI entry points to the canonical module so every
existing invocation (python scripts/..., python src/..., python data/...)
keeps working byte-for-byte.
"""
import os
import sys

if __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cogito.scripts.dataset_manager as _impl

if __name__ == "__main__":
    _impl.main()
    sys.exit(0)

globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
