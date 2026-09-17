"""Lazy autoload, installed at interpreter start by `protenix_v1_opt_autoload.pth` (the build backend's generated line: a guarded
`import protenix_v1_opt._autoload` that turns the module's own SystemExit into that exit code and an import failure under a set
PROTENIX_V1_OPT into the NOT ACTIVE line and exit 3).

With PROTENIX_V1_OPT=exact|fast|big in the environment, a meta-path finder waits for the first import of a trigger package and, right after
that package's own body has executed, calls `protenix_v1_opt.enable(mode, strict=True, det=PROTENIX_V1_OPT_DET == "1", allow_partial=PROTENIX_V1_OPT_ALLOW_PARTIAL == "1")`;
when the mode cannot be activated the core prints its NOT ACTIVE line and the process exits 3 (stock never runs silently under PROTENIX_V1_OPT);
a value that is not a mode refuses the process by name at interpreter start (the same line form, exit 3).
Until then nothing else is imported (no torch, no protenix); with PROTENIX_V1_OPT unset or "off" no finder is installed at all.

Triggers: `runner` (the stock CLI's package — `protenix pred` enters through runner.batch_inference; its `__init__` is empty, so
the levers, and the deterministic recipe, install before any model module loads) and `protenix.model` (library use: a caller that
builds `runner.inference.InferenceRunner` itself; the deterministic recipe is refused there when scatter_utils is already imported).
The finder fires once, when a trigger's body has run (a find_spec probe alone leaves it armed), and removes itself.
"""
import os
import sys

ENV = "PROTENIX_V1_OPT"
ENV_DET = "PROTENIX_V1_OPT_DET"
ENV_ALLOW_PARTIAL = "PROTENIX_V1_OPT_ALLOW_PARTIAL"   # =1: this route's `--allow-partial` (the environment route's only surface for it; `pred` reads its flag, never this name)
BIG_PREFIX = "PROTENIX_V1_BIG_"                  # the memory line reads exactly two names under it (its size statements, == big.GATE_SETTINGS): every other is a mistyped or retired switch
BIG_DECLARED = ("PROTENIX_V1_BIG_SIZE_N_TOKEN", "PROTENIX_V1_BIG_TRIMUL_TORCH_TOKENS")
DECLARED = (ENV, ENV_DET, ENV_ALLOW_PARTIAL, "PROTENIX_V1_OPT_KIT", "PROTENIX_V1_OPT_N_GPU",
            "PROTENIX_V1_OPT_WEIGHTS_MEMO")   # == stack.PACKAGE_ENV (the documented pair; this path imports nothing at start): EVERY name the package reads under its prefix
TRIGGERS = ("runner", "protenix.model")
MODES = ("exact", "fast", "big", "off")
TAG = "protenix-v1-opt"   # == report.TAG (this path imports nothing of the package at start)
EXIT_NOT_ACTIVE = 3   # == report.EXIT_NOT_ACTIVE (test_merge_locks: the documented pair); the .pth path imports nothing at interpreter start


def _protenix_family(fullname, spec):
    """`runner` is a generic name: accept it only when a `protenix` package sits beside it."""
    if fullname != "runner":
        return True
    origin = getattr(spec, "origin", None) or ""
    return os.path.isfile(os.path.join(os.path.dirname(os.path.dirname(origin)), "protenix", "__init__.py"))


class Finder:
    """Duck-typed meta-path finder (no importlib.abc import at interpreter start)."""

    def __init__(self, mode, det=False, allow_partial=False):
        self.mode = mode
        self.det = det
        self.allow_partial = allow_partial
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
        if spec is None or spec.loader is None or not _protenix_family(fullname, spec):
            return None
        # the hook stays armed until a trigger module's body actually runs: a bare importlib.util.find_spec(<trigger>) probe returns a
        # wrapped spec nobody executes, and the real import that follows gets its own wrapped spec — stock never runs unhooked
        orig = spec.loader.exec_module

        def exec_module(module, _orig=orig):
            _orig(module)                              # the trigger package's own body first, then the levers
            self._fire(module.__name__)
        spec.loader.exec_module = exec_module
        return spec

    def _fire(self, trigger):
        if self.fired is not None:                                  # one activation per process: a second wrapped spec executing later is a no-op
            return
        self.fired = trigger
        self.armed = False
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        from ._core_gate import gate                                # statement one: the core pin gate (the importable opt_core is the pinned one,
        gate(__file__, tag=TAG)                                     #   else NOT ACTIVE reason=core_missing:opt_core | core_mismatch | core_pin_unreadable, exit 3)
        from . import _producers                                    # statement two: every core module this package imports resolves, else
        _producers.refuse_if_missing(exit=sys.exit)                 #   reason=producer_missing:<modules>, exit 3
        try:
            import protenix_v1_opt
            enable, refused = protenix_v1_opt.enable, protenix_v1_opt.ActivationError
            import protenix_v1_opt.stack  # noqa: F401  the package's own modules import here, or the process is refused by name below
        except ImportError as e:                                    # a module of the package (or of the stock) is not importable:
            missing = getattr(e, "name", None) or str(e)                #   named, exit 3 — never a traceback, never a silent stock run
            kind = "core_missing" if str(missing).split(".")[0] == "opt_core" else "import_error"
            _refuse_process(f"[protenix-v1-opt] NOT ACTIVE: reason={kind}:{missing} ({ENV}={self.mode}: the package cannot activate without it; "
                            f"pip install -e common/opt_core -e opt); exit {EXIT_NOT_ACTIVE}", exit=sys.exit)
            return
        try:
            enable(self.mode, strict=True, trigger=trigger, det=self.det, allow_partial=self.allow_partial)
        except refused:
            # the core has printed `[protenix-v1-opt] NOT ACTIVE: <reason>`; the process stops here (report.EXIT_NOT_ACTIVE) rather than
            # running stock silently. A PARTIAL activation at the runner (the kit did not apply a lever of the mode) stops the process
            # the same way from stack._wrap_runner unless PROTENIX_V1_OPT_ALLOW_PARTIAL=1 was set (this route's recorded opt-out).
            sys.exit(EXIT_NOT_ACTIVE)



def _refuse_process(line, exit=os._exit):
    """The NOT ACTIVE line, then the process ends with EXIT_NOT_ACTIVE — `os._exit`: at interpreter start (a .pth line runs inside site's own
    try/except) a SystemExit would be reported as a site error and the process would go on to run stock; nothing else may run."""
    sys.stderr.write(line + "\n")
    sys.stderr.flush()
    exit(EXIT_NOT_ACTIVE)


def install(environ=None, exit=os._exit):
    """Install the finder for PROTENIX_V1_OPT (idempotent). Returns the finder, or None when nothing is to be done. A value that is not a
    mode refuses the PROCESS by name at interpreter start (`[protenix-v1-opt] NOT ACTIVE: unknown PROTENIX_V1_OPT=<value> ...`, exit 3):
    stock never runs under a set PROTENIX_V1_OPT, whatever its spelling. A name under the package prefix the package does not declare
    (a mistyped switch: PROTENIX_V1_OPT_MODE, PROTENIX_V1_OPTS, ...) refuses the same way — it would otherwise be ignored and stock run —
    and so does a PROTENIX_V1_BIG_* name other than the memory line's two size statements (the line's levers are not switchable)."""
    environ = os.environ if environ is None else environ
    undeclared = sorted(k for k in environ if (k.startswith(ENV) and k not in DECLARED) or (k.startswith(BIG_PREFIX) and k not in BIG_DECLARED))
    if undeclared:
        reads = DECLARED + (BIG_DECLARED if any(k.startswith(BIG_PREFIX) for k in undeclared) else ())
        _refuse_process(f"[protenix-v1-opt] NOT ACTIVE: undeclared {','.join(undeclared)} (the package reads {', '.join(reads)}); exit {EXIT_NOT_ACTIVE}: a mistyped switch never runs stock", exit)
        return None
    mode = (environ.get(ENV) or "").strip().lower()
    if not mode or mode == "off":
        return None
    if mode not in MODES:
        _refuse_process(f"[protenix-v1-opt] NOT ACTIVE: unknown {ENV}={mode!r} (expected exact|fast|big|off); exit {EXIT_NOT_ACTIVE}: stock never runs under a set {ENV}", exit)
        return None
    for f in sys.meta_path:
        if isinstance(f, Finder):
            return f
    f = Finder(mode, det=(environ.get(ENV_DET) or "").strip() == "1", allow_partial=(environ.get(ENV_ALLOW_PARTIAL) or "").strip() == "1")
    sys.meta_path.insert(0, f)
    return f


FINDER = install()
