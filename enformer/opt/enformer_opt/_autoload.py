"""The environment route: ``ENFORMER_OPT=exact`` in front of any program. The package's .pth (site-packages, written by ``pip install -e``)
imports this module at interpreter start; with the switch set to ``exact`` a meta-path finder waits for the first import of ``enformer_pytorch``
and enables the kit right after that import completes — so the user's own script, unchanged, gets the levers. ``off`` / unset: nothing is
installed. Any other value is refused at start (``[enformer-opt] NOT ACTIVE ... reason=unknown ENFORMER_OPT value``, exit 3): a set switch
never runs stock silently. When the kit cannot engage (``enable()`` refuses by name) the process exits 3 after the NOT ACTIVE line."""
from __future__ import annotations

import os
import sys

ENV = "ENFORMER_OPT"
TRIGGER = "enformer_pytorch"
VALUES = ("exact", "off")


def _exit3():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    os._exit(3)                        # not SystemExit: inside a .pth / an import hook at interpreter start a SystemExit is fatal with rc 1


class Finder:
    """Duck-typed meta-path finder: lets the real finder load ``enformer_pytorch``, then enables the kit once."""

    def __init__(self):
        self.armed = True

    def find_spec(self, fullname, path=None, target=None):
        if not self.armed or fullname != TRIGGER:
            return None
        rt = sys.modules.get("enformer_opt._runtime")
        if rt is not None and getattr(rt, "_ENABLING", False):        # an explicit enable() is importing the trigger itself: it is the one activation
            disarm()
            return None
        spec = None
        for finder in sys.meta_path:
            if finder is self:
                continue
            try:
                spec = finder.find_spec(fullname, path, target)
            except Exception:
                spec = None
            if spec is not None:
                break
        if spec is None or spec.loader is None:
            return None
        self.armed = False
        orig = spec.loader.exec_module

        def exec_module(module, _orig=orig):
            _orig(module)                                          # upstream's own body first, then the kit
            disarm()
            import enformer_opt
            try:
                enformer_opt.enable(strict=True, trigger="ENFORMER_OPT")
            except enformer_opt.ActivationError:
                _exit3()                                           # the NOT ACTIVE line has been printed; stock never runs under a set switch
        spec.loader.exec_module = exec_module
        return spec


def disarm():
    for f in list(sys.meta_path):
        if isinstance(f, Finder):
            f.armed = False
            sys.meta_path.remove(f)


def install(environ=None):
    """Arm the finder for ENFORMER_OPT=exact (idempotent); None when the switch is unset or off."""
    environ = os.environ if environ is None else environ
    value = (environ.get(ENV) or "").strip().lower()
    if not value or value == "off":
        return None
    if value not in VALUES:
        sys.stderr.write(f"[enformer-opt] NOT ACTIVE mode={value} reason=unknown {ENV} value {value!r} (expected exact, or off / unset)\n")
        _exit3()
    for f in sys.meta_path:
        if isinstance(f, Finder):
            return f
    f = Finder()
    sys.meta_path.insert(0, f)
    return f


FINDER = install()
