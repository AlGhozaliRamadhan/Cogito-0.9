"""Test bootstrap.

The repo is not pip-installed, so tests run from the checkout need the
repository root on ``sys.path`` to import the ``cogito`` package. This is
the *only* place path setup lives — production code must never manipulate
``sys.path`` (see docs/ARCHITECTURE.md).
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
