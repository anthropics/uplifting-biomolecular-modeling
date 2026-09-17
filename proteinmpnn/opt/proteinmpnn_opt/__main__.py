"""`python -m proteinmpnn_opt <command> ...` == `proteinmpnn-opt <command> ...` (cli.main; the console script imports `main` from here).
Statement one is the core pin gate (`proteinmpnn_opt.core_gate`): a shared core absent or older than the kit's pin is one NOT ACTIVE line and
exit 3 before anything of the core or the command line is imported."""
import sys

from . import core_gate

core_gate()

from .cli import main  # noqa: E402  (after the gate by design)

if __name__ == "__main__":
    sys.exit(main())
