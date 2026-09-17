"""The package's exit codes — the shared core's exit table (``opt_core.report``: 0 ok · 1 · 2 · 3). One leaf module whose only
import is the core's stdlib-only report module (``cli.py`` and ``stack.py`` import it; ``_autoload.py`` and ``stock_design.py``
keep import-free copies of EXIT_NOT_ACTIVE, held equal by the kit's unit tests).

0 ok · 1 the run failed (a failed child, a failed step) · 2 usage · 3 not active — the requested mode cannot activate on this box (a
lever directory or the pinned upstream absent or at another version, no GPU, no ``--seed``, the accelerator library a kit mode routes
through absent in the model process, the stock process's environment not clean of the kit), or the activation is PARTIAL — a lever of
the mode could not run on the box (a kernel that does not import, compile or launch, an unsupported shape or dtype): a mode is all of
its levers and refuses by name, with no opt-out. Nothing else exits non-zero: a request upstream would run is never refused, an
untested card, driver or library patch level is named on a NOTE line and the levers engage, and what the accelerator census finds
after a run is recorded (``opt_manifest.json``), not an exit.
"""
from opt_core.report import EXIT_FAIL, EXIT_NOT_ACTIVE, EXIT_OK, EXIT_USAGE

__all__ = ["EXIT_OK", "EXIT_FAIL", "EXIT_USAGE", "EXIT_NOT_ACTIVE"]
