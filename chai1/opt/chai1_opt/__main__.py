"""``python -m chai1_opt ...`` -> :func:`chai1_opt.cli.main`."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
