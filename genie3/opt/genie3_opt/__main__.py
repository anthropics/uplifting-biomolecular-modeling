"""`python -m genie3_opt ...` == `genie3-opt ...` (cli.main). Statement one is the core pin gate (_core.core_gate): an absent, older,
newer or edited shared core is one NOT ACTIVE line and exit 3 before anything of the core or the CLI is imported."""
import sys

from ._core import core_gate

core_gate()

from .cli import main  # noqa: E402  (after the gate by design)

if __name__ == "__main__":
    sys.exit(main())
