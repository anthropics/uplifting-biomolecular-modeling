"""Executed by ``af3_torch_opt_autoload.pth`` at interpreter start (the tree's contract: a .pth-installed lazy hook per package).

On this model the hook applies nothing: the levers are arguments of the kit's own ``build_model`` in a model process the wrapper
launches (cli.py / forward.py), not an in-process patch of an import, so there is nothing an import of the model family could activate.
What the hook does: when ``AF3_TORCH_OPT`` is set to a mode other than ``off`` in the environment of an interpreter that then imports
the kit's api (``af3_torch_api`` or ``xfold`` — a caller using the kit directly, not the wrapper), the hook cannot apply that mode in
this process, so it refuses: the ``NOT ACTIVE`` line and exit 3 (``os._exit``: the code holds under the .pth form too, where a raised
``SystemExit`` would be reported by ``site`` as a startup error and exit 1) at the first import (a ``find_spec`` probe of a trigger counts) —
never the mode's name on a process that runs stock. A value that is not a mode of this package is refused at interpreter start the
same way. The wrapper (``af3-torch-opt pred`` / ``run.sh pred``) is unaffected: it imports no trigger and strips the variable from its
model processes. Nothing is imported at start beyond this module unless the variable is set.
"""
from __future__ import annotations

import importlib.abc
import os
import sys

ENV_MODE = "AF3_TORCH_OPT"
TAG = "af3-torch-opt"
TARGETS = ("af3_torch_api", "xfold")
EXIT_NOT_ACTIVE = 3
_STATE = {"armed": False, "fired": False, "mode": None}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):  # noqa: D401 - importlib protocol
        if name in TARGETS or name.startswith(tuple(t + "." for t in TARGETS)):
            _fire()
        return None


def _fire() -> None:
    if _STATE["fired"]:
        return
    _STATE["fired"] = True
    try:
        sys.meta_path.remove(_finder)
    except ValueError:
        pass
    sys.stderr.write(f"[{TAG}] NOT ACTIVE mode={_STATE['mode']} reason={ENV_MODE}={_STATE['mode']} cannot be applied by a direct import of the kit "
                     "(the levers are build_model arguments the wrapper command passes: af3-torch-opt pred / run.sh pred)\n")
    sys.stderr.flush(); sys.stdout.flush()
    os._exit(EXIT_NOT_ACTIVE)                                    # the process ends with the code, wherever the import happened (a SystemExit raised inside
                                                                 # site's .pth processing is reported as a startup error and exits 1)


_finder = _Finder()
_mode = os.environ.get(ENV_MODE, "").strip()
if _mode and _mode != "off":
    from ._core_gate import gate, CoreGateRefused                # THE pin gate (the core's kit template, carried): the importable opt_core is the pinned one, else the NOT ACTIVE line
    try:
        gate(__file__, tag=TAG)
    except CoreGateRefused:                                      # printed by the gate; end the interpreter here (site would swallow a SystemExit from a .pth line)
        sys.stderr.flush(); sys.stdout.flush()
        os._exit(EXIT_NOT_ACTIVE)
    from ._producers import refuse_if_missing                   # the finer words for an importable-but-incomplete core (producer_missing), exit 3
    refuse_if_missing(exit=os._exit, argv=())
    try:
        from .modes import MODES, REFUSED_MODES                  # the one mode table (imported only under a set variable); it imports the shared core
    except ImportError as _e:                                    # a producer present by name that fails to import: refused by name, never a traceback or a stock run
        sys.stderr.write(f"[{TAG}] NOT ACTIVE mode={_mode} reason=core_missing:{getattr(_e, 'name', None) or _e}\n")
        sys.stderr.flush(); sys.stdout.flush()
        os._exit(EXIT_NOT_ACTIVE)
    if _mode in REFUSED_MODES:                                   # a standard mode name this engine refuses by name (modes.REFUSED_MODES: the reason, one line)
        sys.stderr.write(f"[{TAG}] NOT ACTIVE mode={_mode} reason={REFUSED_MODES[_mode]}\n")
        sys.stderr.flush(); sys.stdout.flush()
        os._exit(EXIT_NOT_ACTIVE)
    if _mode not in MODES:
        sys.stderr.write(f"[{TAG}] NOT ACTIVE mode={_mode} reason={ENV_MODE}={_mode} is not a mode of this package (modes: {', '.join(MODES)})\n")
        sys.stderr.flush(); sys.stdout.flush()
        os._exit(EXIT_NOT_ACTIVE)
    _STATE.update(armed=True, mode=_mode)
    sys.meta_path.insert(0, _finder)
