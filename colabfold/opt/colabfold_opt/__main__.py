"""`python -m colabfold_opt ...` == `colabfold-opt ...` (cli.py). Statement one is the core pin gate (_core_gate, the package's copy of the
shared core's kit template: an absent or another opt_core is `NOT ACTIVE: reason=core_missing:opt_core …` / `reason=core_mismatch: …`, exit 3);
a package import that then fails because a module of the core is missing is `NOT ACTIVE: core_missing:<module>`, exit 3 (_autoload.core_missing)
— never a traceback."""
import sys

from ._autoload import TAG, core_missing                  # os / sys only
from ._core_gate import gate                             # standard library only


def main(argv=None):
    gate(__file__, tag=TAG)                               # the pinned core, or NOT ACTIVE and exit 3 — before any opt_core import
    try:
        from .cli import main as cli_main
    except ImportError as e:
        return core_missing(e)
    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
