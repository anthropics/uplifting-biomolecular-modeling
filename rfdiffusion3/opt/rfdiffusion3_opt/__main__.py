"""``python -m rfdiffusion3_opt ...`` and the console script ``rfdiffusion3-opt`` -> :func:`rfdiffusion3_opt.cli.main`, after the core
pin gate (statement one: the pinned ``opt_core`` is importable, else one ``NOT ACTIVE`` line and exit 3 before anything of the core loads)."""
from ._autoload import core_gate

core_gate()          # absent / stale / pre-manifest core or unreadable pin -> '[rfdiffusion3-opt] NOT ACTIVE: reason=core_missing:opt_core | core_mismatch: … | core_pin_unreadable: …', exit 3

import sys  # noqa: E402

from .cli import main  # noqa: E402  (cli imports the core at module level: report -> opt_core.report)

if __name__ == "__main__":
    sys.exit(main())
