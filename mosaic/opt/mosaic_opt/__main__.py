"""`python -m mosaic_opt` == `mosaic-opt` (console script): design | check | warm. Statement one is the core pin gate
(`mosaic_opt.core_gate`): an absent or older shared core is one NOT ACTIVE line and exit 3 before anything of the core or
the CLI is imported."""
import sys

from . import core_gate

core_gate()

from .cli import main  # noqa: E402  (after the gate by design: cli imports the core)

if __name__ == "__main__":
    sys.exit(main())
