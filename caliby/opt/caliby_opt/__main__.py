"""``python -m caliby_opt ...`` and the ``caliby-opt`` console script -> :func:`caliby_opt.cli.main`. Statement one is the core pin gate
(``stack.core_gate``): an absent, older/newer or edited ``opt_core`` is one ``[caliby-opt] NOT ACTIVE: reason=core_...`` line and exit 3
before anything of the core or the CLI is imported."""
import sys

from .stack import core_gate

core_gate()

from .cli import main  # noqa: E402  (cli -> report imports opt_core: after the gate by design)

if __name__ == "__main__":
    sys.exit(main())
