"""Lazy autoload, installed at interpreter start by `rosettafold3_opt_autoload.pth` (`import rosettafold3_opt._autoload`).

With ROSETTAFOLD3_OPT=fast (or exact, or big) in the environment, a meta-path finder waits for the first import of the upstream package
`rf3` and, BEFORE that package's body executes, calls `rosettafold3_opt.enable(mode, strict=True)`: the kit's switches are read when
`rf3.graph_flags` is imported, so the row must be exported before any rf3 module runs (rf3/__init__.py
imports nothing, but the order is kept for any importer). A row with an FPF arm (fast) is completed by the core's own second watch,
which applies the arm right after `rf3.model.RF3_structure` executes (stack.py). When the mode cannot be activated (tree not
patched, pins, no GPU, contradicting environment, FPF files missing, the core absent or not the pinned one — `enable()`'s first
statement is the pin gate `_core_gate.gate`: `reason=core_missing:opt_core (…)` / `reason=core_mismatch: …` — or a kit module not
importable) one NOT ACTIVE line is printed and the process exits 3 — stock never runs silently under
ROSETTAFOLD3_OPT, and nothing leaves the import as a traceback. Until then nothing else is imported (no torch, no upstream); with ROSETTAFOLD3_OPT
unset or "off" no finder is installed at all (the package exports nothing: the process runs whatever its interpreter carries). The
finder fires once, when the trigger package's body is about to execute, and removes itself then — a spec lookup that is not
followed by an import (``importlib.util.find_spec("rf3")``, the usual "is rf3 installed?" probe) leaves it armed. MODES
below are the names of modes.py, repeated here because nothing may be imported at interpreter start (locked by tests/test_autoload.py).
"""
import os
import sys

ENV = "ROSETTAFOLD3_OPT"
TRIGGERS = ("rf3",)
MODES = ("off", "exact", "fast", "big")
ENV_NAMES = ("ROSETTAFOLD3_OPT_STOCK_PYTHON", "ROSETTAFOLD3_OPT_PYTHON", "ROSETTAFOLD3_OPT_CKPT", "ROSETTAFOLD3_OPT_TALLY_FILE", "ROSETTAFOLD3_OPT_N_GPU", "ROSETTAFOLD3_OPT_DIGEST_DIR")  # stack.ENV_NAMES, repeated here for the same reason (locked by tests/test_autoload.py)


class Finder:
    """Duck-typed meta-path finder (no importlib.abc import at interpreter start)."""

    def __init__(self, mode):
        self.mode = mode
        self.armed = True
        self.fired = None

    def find_spec(self, fullname, path=None, target=None):
        if not self.armed or fullname not in TRIGGERS:
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
        orig = spec.loader.exec_module

        def exec_module(module, _orig=orig):
            if self.armed:                               # the row first, then the trigger package's own body; once per process
                self._fire(module.__name__)
            _orig(module)
        spec.loader.exec_module = exec_module            # this loader instance belongs to this spec; a discarded spec keeps the finder armed
        return spec

    def _fire(self, trigger):
        self.armed = False
        self.fired = trigger
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        try:
            import rosettafold3_opt
            rosettafold3_opt.enable(self.mode, strict=True, trigger=trigger)
        except Exception as e:                                 # noqa: BLE001 — nothing escapes the import as a traceback, nothing runs stock silently (the pin gate's SystemExit(3) passes through: its line is printed)
            act = getattr(sys.modules.get("rosettafold3_opt"), "ActivationError", None)
            if act is None or not isinstance(e, act):          # ActivationError: the package printed `[rosettafold3-opt] NOT ACTIVE: <reason>` itself;
                from rosettafold3_opt import _core             # anything else (the pinned core or a kit module not importable, …) is named here
                sys.stderr.write(f"{_core.NOT_ACTIVE_PREFIX} {_core.not_active_reason(e)}\n")
            _stop()


def _stop():
    """The process stops here (exit 3) rather than running stock silently under ROSETTAFOLD3_OPT — os._exit, because this runs inside
    site's .pth processing at interpreter start (a SystemExit there is a fatal init error, rc 1) and inside an import (the finder)."""
    sys.stderr.flush()
    os._exit(3)


BIG_PREFIX = "ROSETTAFOLD3_BIG_"              # the memory mode's line is one lever set: no lever word is read; the one word under the prefix is the
BIG_NAMES = ("ROSETTAFOLD3_BIG_ALLOW_PARTIAL",)  # partial-unit opt-out of an rf3 process activated through the environment (big.ALLOW_PARTIAL_ENV)


def undeclared(environ=None):
    """``ROSETTAFOLD3_OPT_*`` / ``ROSETTAFOLD3_BIG_*`` names in the environment this tree does not read (a mistyped variable, or a lever
    switch the line does not have, would otherwise be ignored silently)."""
    environ = os.environ if environ is None else environ
    return {k: environ[k] for k in sorted(environ)
            if (k.startswith(ENV + "_") and k not in ENV_NAMES) or (k.startswith(BIG_PREFIX) and k not in BIG_NAMES)}


def install(environ=None):
    """Install the finder for ROSETTAFOLD3_OPT (idempotent). Returns the finder, or None when nothing is to be done."""
    environ = os.environ if environ is None else environ
    bad = undeclared(environ)
    if bad:                                                    # a mistyped ROSETTAFOLD3_OPT_* name is refused, never ignored (with or without a mode)
        sys.stderr.write(f"[rosettafold3-opt] NOT ACTIVE: undeclared ROSETTAFOLD3_OPT_* / ROSETTAFOLD3_BIG_* names in the environment (mistyped? this tree reads "
                         f"{', '.join(ENV_NAMES)}): " + ", ".join(f"{k}={v!r}" for k, v in bad.items()) + "\n")
        _stop()
    mode = (environ.get(ENV) or "").strip().lower()
    if not mode or mode == "off":
        return None
    if mode not in MODES:
        sys.stderr.write(f"[rosettafold3-opt] NOT ACTIVE: unknown {ENV}={mode!r} (expected {'|'.join(MODES)})\n")
        _stop()                                                # a mistyped selection never runs stock under the variable
    for f in sys.meta_path:
        if isinstance(f, Finder):
            return f
    f = Finder(mode)
    sys.meta_path.insert(0, f)
    return f


FINDER = install()
