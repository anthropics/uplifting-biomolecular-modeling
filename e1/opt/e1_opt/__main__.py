"""`python -m e1_opt <command> ...` == `e1-opt <command> ...` (cli.py)."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
