"""Backwards-compat shim: Real implementation moved to cogito.generators.topics.

Redirects imports and CLI entry points to the canonical module so every
existing invocation keeps working byte-for-byte.
"""
import os
import sys

if __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cogito.generators.topics as _impl

if __name__ == "__main__":
    if hasattr(_impl, "main"):
        _impl.main()
    sys.exit(0)

globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
