"""``python -m protenix_opt._stackkey`` — the probe ``configs/<card>.env`` keys the persistent JIT caches by: prints this stack's
JIT cache key (``modes.jit_cache_key``: ``torch<v>-cu<c>-sm<cc>``) or the word ``unknown`` when a part cannot be established. Not a
command of the CLI: the configuration's own helper. Statement one is the kit's pre-import core pin gate, as in ``__main__`` — an absent
or older shared core exits 3 by name, never a swallowed import error, never a traceback."""
import sys

from . import _core                                        # sys.path placement of the pinned copy; imports nothing of the core

_core.gate()                                               # exit 3 by name unless the importable opt_core is the pinned one
_why = _core.producer_refusal()
if _why:
    sys.stderr.write(f"[protenix-opt] NOT ACTIVE: {_why}\n"); sys.stderr.flush()
    sys.exit(3)

from . import modes  # noqa: E402
from opt_core.jit_cache import StackKeyUnknown  # noqa: E402

if __name__ == "__main__":
    try:
        print(modes.jit_cache_key())
    except StackKeyUnknown:
        print("unknown")
