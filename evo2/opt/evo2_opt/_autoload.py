"""Run at interpreter start by ``evo2_opt_autoload.pth`` (installed with the package): when ``EVO2_OPT=exact`` (or ``fast``) is in the environment,
a meta-path finder waits for the ``evo2`` package's import and engages the kit then (``activation.enable(mode)``) — so ``export EVO2_OPT=exact``
makes any unmodified script that constructs ``evo2.Evo2`` run the kit. ``EVO2_OPT`` unset or ``off``: nothing is installed, nothing of the
kit is imported (the stock). Any other value: one ``[evo2-opt] NOT ACTIVE`` line and exit 2 at the ``evo2`` import (a usage error must not
run silently as the stock)."""
import os
import sys

ENV = "EVO2_OPT"
MODES = ("exact", "fast", "off")
TRIGGER = "evo2"
FINDER = None   # the installed finder while EVO2_OPT=<mode> waits for the evo2 import; None = nothing armed in this interpreter


class _Finder:
    """Fires once, on the first import of the ``evo2`` package (or a submodule), removes itself, then engages the kit."""

    def __init__(self, mode):
        self.mode = mode
        self.fired = False

    def find_spec(self, fullname, path=None, target=None):
        if self.fired or fullname.split(".")[0] != TRIGGER:
            return None
        self.fired = True
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        if self.mode not in MODES:
            sys.stderr.write(f"[evo2-opt] NOT ACTIVE: {ENV}={self.mode!r} is not a mode ({' | '.join(MODES)}); exit 2\n"); sys.stderr.flush()
            raise SystemExit(2)
        from evo2_opt import activation
        try:
            activation.enable(self.mode, trigger=f"{ENV}={self.mode} at the import of {fullname}")
        except activation.Evo2OptRefused:
            raise SystemExit(3)                               # the NOT ACTIVE line is printed by enable(); a refused mode never runs as the stock under its name
        return None


def install():
    mode = os.environ.get(ENV)
    if mode is None or mode == "" or mode == "off":
        return None
    if any(isinstance(f, _Finder) for f in sys.meta_path):
        return None
    global FINDER
    f = _Finder(mode)
    sys.meta_path.insert(0, f)
    FINDER = f
    return f


install()
