"""Lazy autoload, installed at interpreter start by `esmc_opt_autoload.pth` (`import esmc_opt._autoload`).
With ESMC_OPT=exact in the environment, a meta-path finder waits for the first import of the upstream package `esm` and, BEFORE that
package's body executes (the pipeline kit's import lever must precede it: it trims what `esm/__init__.py` pulls in), calls
`esmc_opt.enable(mode, strict=True)` (the variant derived from the model loaded at `ESMC.from_pretrained`); when the mode
cannot be activated (pins, no GPU, kit missing) the core prints its NOT ACTIVE line and the process exits 3 — stock
never runs silently under ESMC_OPT. Until then nothing else is imported (no torch, no upstream); with ESMC_OPT unset or naming a
stock-class mode ("off": STOCK_MODES) no finder is installed at all.
enable() arms the application on `ESMC.from_pretrained` (stack.py): the kits' own apply() runs on each client as it is built, which is
the only point where a loaded model exists. The finder fires once and removes itself; while it fires, nested imports of the trigger
package (the activation itself may import it) pass through to the normal finders.
"""
import os
import sys

ENV = "ESMC_OPT"
TRIGGERS = ("esm",)
MODES = ("exact", "off")                           # == esmc_opt.modes.MODES as a set (nothing of the package is imported at interpreter start; keep the two in sync)
STOCK_MODES = ("off",)                             # == esmc_opt.modes.STOCK_MODES: nothing to arm (the caller runs upstream as it is)


class Finder:
    """Duck-typed meta-path finder (no importlib.abc import at interpreter start)."""

    def __init__(self, mode):
        self.mode = mode
        self.armed = True
        self.fired = None

    def find_spec(self, fullname, path=None, target=None):
        if not self.armed or fullname not in TRIGGERS:
            return None
        self.armed = False                       # once; nested imports of the trigger during the firing fall through to the other finders
        self._fire(fullname)
        return None                              # the normal finders load the trigger package after the activation

    def _fire(self, trigger):
        self.fired = trigger
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        import esmc_opt
        try:
            esmc_opt.enable(self.mode, strict=True, trigger=f"import {trigger}")
        except esmc_opt.ActivationError:
            # the core has printed `[esmc-opt] NOT ACTIVE: <reason>`; the process stops here (exit 3) rather than running stock silently.
            # os._exit after a flush, not SystemExit: a SystemExit raised while the interpreter is still initialising (a trigger import
            # reached from site processing) is fatal to the interpreter (rc 1, "Fatal Python error: init_import_site"), not a refusal.
            for stream in (sys.stdout, sys.stderr):
                try:
                    stream.flush()
                except Exception:
                    pass
            os._exit(3)


def install(environ=None):
    """Install the finder for ESMC_OPT (idempotent). Returns the finder, or None when nothing is to be done."""
    environ = os.environ if environ is None else environ
    mode = (environ.get(ENV) or "").strip().lower()
    if not mode or mode in STOCK_MODES:
        return None
    if mode not in MODES:
        sys.stderr.write(f"[esmc-opt] NOT ACTIVE: unknown {ENV}={mode!r} (expected {'|'.join(MODES)})\n")
        return None
    for f in sys.meta_path:
        if isinstance(f, Finder):
            return f
    f = Finder(mode)
    sys.meta_path.insert(0, f)
    return f


FINDER = install()
