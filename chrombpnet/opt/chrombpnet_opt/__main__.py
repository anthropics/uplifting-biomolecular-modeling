"""``python -m chrombpnet_opt ...`` -> :func:`chrombpnet_opt.cli.main`."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
