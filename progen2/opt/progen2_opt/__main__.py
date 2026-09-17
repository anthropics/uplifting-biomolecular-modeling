"""``python -m progen2_opt ...`` -> :func:`progen2_opt.cli.main`."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
