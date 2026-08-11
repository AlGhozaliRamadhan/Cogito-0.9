"""Allow ``python -m cogito`` to launch the interactive runtime.

Forwards to cogito.cli so the entry point stays in one place.
"""

from cogito.cli import main

if __name__ == "__main__":
    main()
