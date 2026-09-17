"""``python -m ef2inv_opt ...`` -> :func:`ef2inv_opt.cli.main`."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
