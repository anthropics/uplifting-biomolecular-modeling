"""``python -m rosettafold3_opt ...`` -> :func:`rosettafold3_opt.cli.main`. Statement one is the pin gate (``_core_gate.gate``):
an absent core (``reason=core_missing:opt_core (…)``) or a core that is not the pinned one
(``reason=core_mismatch: opt_core pinned >= v<want> at <path>, installed v<have> at <root>``) is ONE NOT ACTIVE line and exit 3 BEFORE any
``opt_core`` import; then the producer check (``_core.require``: ``reason=producer_missing:<module>,…`` for a core lacking a module this
package loads); a kit module that cannot be imported is named the same way (cli.EXIT_NOT_ACTIVE) — never a traceback, never a command on a
lesser core."""
import sys

from . import _core                                 # standard library only: exposes the pinned checkout on sys.path when no opt_core is installed
from ._core_gate import gate

gate(__file__)                                      # THE pin gate: NOT ACTIVE reason=core_missing|core_mismatch|core_pin_unreadable, exit 3, before any opt_core import

_core.require_or_exit()                             # every opt_core module this package loads, found BEFORE the CLI module (which imports the core) is imported: the NOT ACTIVE line, exit 3
try:
    from .cli import main
except ImportError as _e:                           # a kit / core module failing to import: named the same way, exit 3
    sys.stderr.write(f"{_core.NOT_ACTIVE_PREFIX} {_core.not_active_reason(_e)}\n")
    sys.stderr.flush()
    sys.exit(_core.EXIT_NOT_ACTIVE)

if __name__ == "__main__":
    try:
        sys.exit(main())
    except ImportError as _e:                       # a producer that disappears between the check and its first load (never expected): same line, same code
        sys.stderr.write(f"{_core.NOT_ACTIVE_PREFIX} {_core.not_active_reason(_e)}\n")
        sys.exit(_core.EXIT_NOT_ACTIVE)
