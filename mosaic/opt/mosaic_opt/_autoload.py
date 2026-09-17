"""The kit's autoload hook, run at interpreter start by `mosaic_opt_autoload.pth` (`import mosaic_opt._autoload`).

Import-free until MOSAIC_OPT names a mode: the .pth runs in every interpreter on the box, the stock arm's processes included, and those
load nothing of the core. With MOSAIC_OPT=exact the core's finder (`opt_core.autoload`) is installed from this kit's AutoloadSpec: it
waits for the first import of the trigger package `mosaic` (its `__init__` is empty: importing it runs no jax computation, so P1 — which
the kit's `repro_cache.enable()` must apply before the first jax computation — is still in time), lets that package's own body run,
removes itself and calls this module's `enable(mode, strict=True, trigger="mosaic")`. When the mode cannot be activated (cache root
unset, pins, no GPU, kit missing, the JAX backend already initialised) the package prints its NOT ACTIVE line and the process exits 3 —
stock never runs silently under MOSAIC_OPT. Under a set variable statement one is the core pin gate (`mosaic_opt.core_gate`, before the
core's finder is imported): an absent or older shared core is its one NOT ACTIVE line, written and flushed, and the process
ends with status 3 (the generated `mosaic_opt_autoload.pth` guard turns the gate's SystemExit into the process status: site.py would swallow a .pth line's exception and let stock run). A word the
hook does not serve — a tier word with no lever wired, or an unknown word — REFUSES BY NAME: the
trigger prints the kit's NOT ACTIVE line (`unknown_line`) and the process exits 3 (`on_unknown="refuse_at_trigger"`; stock never runs under a
requested mode); with MOSAIC_OPT unset or "off" nothing is installed and nothing of the core is located or imported. Until the trigger nothing else is imported (no jax, no upstream).

What the hook applies: P1 (every mode that carries it; the transparent form steps aside by name without MOSAIC_OPT_CACHE_ROOT) and the
mode's per-step levers (`fast`, `big`: `levers.install(mode)`). P2 and P3 are call-site replacements the program itself would have to
make (`__init__.py`): on this route they STEP ASIDE BY NAME — the ACTIVE line's row token names them (`row=T_fast[…]-aside[P2:driver_only]`,
`row=C_p1warm_p2[P1]-aside[P2:driver_only,P3:driver_only]`), the report lists them under `levers_driver_only`, and the process continues
(status 0) on every other lever of the row; they are not a partial activation and `unavailable=` does not list them. `partial` keeps ONE
meaning: a lever this route should apply and did not. The activation refuses by name before that can happen (pcc.enable() failing, a
per-step lever refusing: NOT ACTIVE, exit 3); should a report nevertheless come back partial, the hook FAILS CLOSED: it prints the one
partial verdict (`[mosaic-opt] NOT ACTIVE: partial activation — <levers> — …; exit 3 (--allow-partial records and proceeds)`) and the process
exits 3, unless MOSAIC_OPT_ALLOW_PARTIAL=1 records the opt-out (`allow_partial=1` beside `partial=` on the ACTIVE line, the `PARTIAL
allowed:` line) and the process continues — that variable is this route's opt-out only; `mosaic-opt design` is the gated form of the driver
route (exit 3 on a partial activation unless `--allow-partial`).

The kit driver (`tools/public_design_run.py`) is never levered by the hook: its arms are composed at the process boundary by
`mosaic-opt design | warm` (the row's environment and flags), and a stock arm run directly with MOSAIC_OPT=exact exported would
otherwise be levered silently (the activation line goes to the arm's log). When the process is the driver, the hook prints its NOT
ACTIVE line and exits 3 (`DRIVER_BASENAME`): unset MOSAIC_OPT to run the driver by hand.
"""
import os
import sys

from . import ActivationError  # noqa: F401  (the package's own exception: the core's finder catches `<this module>.ActivationError`; the package __init__ is import-free)
from . import TAG, core_gate                             # the kit's one tag spelling and its core pin gate (both import-free, the package root)

ENV = "MOSAIC_OPT"
TRIGGERS = ("mosaic",)
MODES = ("fast", "exact", "big", "off")                                # modes.MODES, copied import-free; locked by tests/test_core_adoption.py
EXIT_NOT_ACTIVE = 3                                      # report.EXIT_NOT_ACTIVE (= opt_core.report's), copied import-free; locked by tests/test_core_adoption.py
DRIVER_BASENAME = "public_design_run.py"                 # the kit driver (modes.DRIVER_RELPATH); never levered by the hook


def spec():
    """This kit's AutoloadSpec (imports the core: call only under a set variable)."""
    from opt_core.autoload import AutoloadSpec
    return AutoloadSpec(env=ENV, package=__name__, tag=TAG, triggers=TRIGGERS, modes=MODES, exit_not_active=EXIT_NOT_ACTIVE,
                        on_unknown="refuse_at_trigger", line_of=unknown_line, fold_mode=True)


TIER_WORDS = ("fast", "big")                           # modes.TIER_WORDS, copied import-free; locked by tests/test_core_adoption.py


def unknown_line(what, value):
    """The kit's NOT ACTIVE line for a selection the hook does not serve: a tier word with no lever wired is named as such (never
    resolved to stock), any other word is unknown. The trigger prints it and the process exits EXIT_NOT_ACTIVE (`on_unknown="refuse_at_trigger"`)."""
    v = (value or "").strip().lower()
    if what == "mode" and v in TIER_WORDS and v not in MODES:
        return (f"[{TAG}] NOT ACTIVE: {ENV}={value!r} is a tier word of this kit line with no lever wired into it "
                f"(mosaic/CHANGES.md \"What is not wired\"); the modes served: {', '.join(MODES)}; exit {EXIT_NOT_ACTIVE}")
    return f"[{TAG}] NOT ACTIVE: unknown {ENV}={value!r}; expected one of {MODES}; exit {EXIT_NOT_ACTIVE}"


def driver_process_refusal(argv=None):
    """The reason the hook refuses in this process, or None: the process is the kit driver (sys.argv[0] is `public_design_run.py`)."""
    argv = sys.argv if argv is None else argv
    if argv and os.path.basename(str(argv[0])) == DRIVER_BASENAME:
        return (f"{ENV} is set but this process is the kit driver ({DRIVER_BASENAME}): its arms are composed by `mosaic-opt design | warm`, "
                f"never by the import hook — unset {ENV} to run the driver by hand")
    return None


def enable(mode, *, strict=True, trigger=None):
    """What the core's finder calls at the trigger: the driver-process refusal, the package's activation (`mosaic_opt.enable`; its
    ActivationError reaches the finder, which exits EXIT_NOT_ACTIVE), then the partial gate of the in-process route."""
    why = driver_process_refusal()
    if why:
        sys.stderr.write(f"[{TAG}] NOT ACTIVE: {why}\n"); sys.stderr.flush()
        sys.exit(EXIT_NOT_ACTIVE)
    import mosaic_opt
    from . import report
    rep = mosaic_opt.enable(mode, strict=strict, trigger=trigger)
    if report.partial_levers(rep):
        # a PARTIAL activation (a lever this route should apply and did not — never a call-site lever stepping aside by name, those ride
        # row=…-aside[P2:driver_only] and are not partial) fails closed (exit 3, the one partial verdict) unless MOSAIC_OPT_ALLOW_PARTIAL=1 records the opt-out and proceeds
        allowed = report.allow_partial_env()
        rep["allow_partial"] = allowed
        report.log(report.partial_verdict_line(report.partial_detail(rep, "in-process"), allowed, report.EXIT_NOT_ACTIVE))
        if not allowed:
            sys.exit(report.EXIT_NOT_ACTIVE)
    return rep


def install(environ=None):
    """Install the core's finder for MOSAIC_OPT (idempotent). Returns the finder, or None when nothing is to be done (unset, `off`, or an
    unknown mode after its NOT ACTIVE line)."""
    environ = os.environ if environ is None else environ
    raw = (environ.get(ENV) or "").strip()
    if not raw or raw.lower() == "off":
        return None                                      # nothing of the core is located or imported in a stock process
    core_gate()                                          # THE core pin gate, before the core's finder is imported: absent / older / newer / edited core -> its one NOT ACTIVE line and
                                                         # SystemExit(3); at .pth time the generated mosaic_opt_autoload.pth guard turns that into the process exit status (site.py
                                                         # would otherwise swallow a .pth line's exception and stock would run)
    from opt_core import autoload
    return autoload.install(spec(), environ)


def disarm():
    """Remove this kit's finder (the package's own `enable()` ran first, or the stock route is about to import upstream). True when one was armed."""
    if "opt_core.autoload" not in sys.modules:           # never installed in this process: nothing to import, nothing to remove
        return False
    from opt_core import autoload
    return autoload.disarm(spec())


def __getattr__(name):                                   # PEP 562: `Finder` is the core's class, resolved only when asked for (tests, isinstance checks)
    if name == "Finder":
        from opt_core.autoload import Finder
        return Finder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


FINDER = install()
