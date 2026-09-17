"""``python -m rosettafold3_opt._require`` — ``run.sh``'s and ``configs/<gpu>.env``'s check before anything of the package runs: the pin
gate (``_core_gate.gate``: core absent / not the pinned one → the NOT ACTIVE line, exit 3), then the producer check (``_core.require_or_exit``:
``reason=producer_missing:<m>,…``, exit 3); exit 0 silently otherwise. Standard library only, so it runs on an interpreter whose core is absent."""
import sys

from . import _core                                 # exposes the pinned checkout on sys.path when no opt_core is installed (the tree-checkout route)
from ._core_gate import gate

if __name__ == "__main__":
    gate(__file__)                                  # SystemExit(3) with the line printed when refused
    _core.require_or_exit()                         # the producer check: the NOT ACTIVE line + SystemExit(3), or silence
    sys.exit(0)
