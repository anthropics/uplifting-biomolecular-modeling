"""``python -m protenix_opt ...`` (and the console script ``protenix-opt``) -> :func:`protenix_opt.cli.main`. Statement one, before
any module of the shared core is imported: the kit's pre-import core pin gate (``_core_gate``, the shared template's bytes; the pinned copy
beside the tree is placed on sys.path first when no core is installed) — an absent core, a core that is not the pinned one, or an unreadable
pin is ``[protenix-opt] NOT ACTIVE: reason=core_missing:opt_core …`` / ``reason=core_mismatch: …`` / ``reason=core_pin_unreadable: …``,
exit 3; then the module-granular words for a core lacking a producer this tree imports (``reason=producer_missing:…``). Never a traceback,
never a run without the lever."""
import sys

from . import _core                                        # sys.path placement of the pinned copy; imports nothing of the core

_core.gate()                                               # exit 3 by name unless the importable opt_core is the pinned one
_why = _core.producer_refusal()
if _why:
    sys.stderr.write(f"[protenix-opt] NOT ACTIVE: {_why}\n"); sys.stderr.flush()
    sys.exit(3)

from .cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
