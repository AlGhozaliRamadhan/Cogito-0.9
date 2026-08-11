"""CLI entry point: `python -m cogito` / `python -m cogito.cli`.

Replaces the former run.py root shim. Dispatches to the canonical runtime
in cogito.runtime so every invocation of the interactive model starts from
the same main().
"""

from cogito.runtime import main

if __name__ == "__main__":
    main()
