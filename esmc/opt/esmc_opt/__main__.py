"""``python -m esmc_opt ...`` -> :func:`esmc_opt.cli.main`."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
