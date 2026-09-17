"""``python -m pxdesign_opt ...`` and the console script ``pxdesign-opt`` (``pxdesign_opt.__main__:main``) -> :func:`pxdesign_opt.cli.main`.
The core pin gate is statement one (``cli`` imports the core at module level): an absent or mismatched ``opt_core`` is one
``[pxdesign-opt] NOT ACTIVE: reason=…`` line and exit status 3, never a traceback."""
from . import core_gate

core_gate()                                              # [pxdesign-opt] NOT ACTIVE: reason=core_missing:opt_core | core_mismatch: … | core_pin_unreadable: …  -> exit 3

import sys  # noqa: E402

from .cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
