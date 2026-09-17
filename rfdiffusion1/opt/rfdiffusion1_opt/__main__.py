"""``python -m rfdiffusion1_opt ...`` and the ``rfdiffusion1-opt`` console script (which imports this module) -> :func:`rfdiffusion1_opt.cli.main`,
behind the core pin gate: the ``opt_core`` this interpreter would import must be the one opt/pyproject.toml pins, or the process ends here with
``[rfdiffusion1-opt] NOT ACTIVE: reason=core_missing:opt_core | core_mismatch: … | core_pin_unreadable: …`` and exit 3, before anything of the
core is imported."""
from ._core_gate import gate
from .report import TAG

gate(__file__, tag=TAG)                                   # the core pin gate (kit_template): statement one of the entry, before any opt_core import

import sys  # noqa: E402

from .cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
