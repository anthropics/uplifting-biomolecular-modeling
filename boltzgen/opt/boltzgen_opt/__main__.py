"""``python -m boltzgen_opt ...`` -> :func:`boltzgen_opt.cli.main` (also the console script `boltzgen-opt`, which imports this module).
Statement one is the core pin gate (``boltzgen_opt.core_gate``): an absent or mismatched shared core is one NOT ACTIVE line and exit 3
before anything of the core or the CLI is imported."""
import sys

from . import core_gate

core_gate()

from .cli import main  # noqa: E402  (after the gate by design: cli imports the core)

if __name__ == "__main__":
    sys.exit(main())
