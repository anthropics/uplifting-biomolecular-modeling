"""Lazy autoload, installed at interpreter start by `chrombpnet_opt_autoload.pth` (`import chrombpnet_opt._autoload`).

With CHROMBPNET_OPT=fast in the environment, a meta-path finder waits for the first import of the top-level `chrombpnet` package
(the stock console script `chrombpnet = chrombpnet.CHROMBPNET:main`, stock setup.py:27, imports it first; its `__init__` is empty)
and, right after that package's own body has executed, acts:

* `sys.argv` is the stock console script with the `pred_bw` subcommand -> the mode line is printed and the kit's documented line with the
  same arguments (`python <kit>/tf/pred_bw_fast.py <argv[2:]>`) runs as a child this process starts and waits for
  (cli.run_fast — the one path shared with `chrombpnet-opt pred_bw`), with CHROMBPNET_OPT* removed from the child's environment (the kit
  process imports `chrombpnet` too; the finder must not fire there) and, under CHROMBPNET_OPT_DET=1, the deterministic recipe composed
  (det.py); the stock's own main never runs. After the child exits this process owns the exit: the kit's run record is judged
  (stack.applied), `opt_manifest.json` written beside the outputs with the record folded in; a lever of the class's set that could
  not start made the kit refuse the mode by name (`[chrombpnet-opt] NOT ACTIVE mode=fast reason=the kit refused by name: …; exit 3`),
  and a partial activation in the record (the backstop) prints `[chrombpnet-opt] NOT ACTIVE: partial activation — <detail>; exit 3
  (--mode off runs stock)` and exits 3.
  THE ONE DEVIATION from an in-process hook: the kit's form is a replacement entry script; it offers no in-process patch.
* any other subcommand -> `[chrombpnet-opt] NOT ACTIVE mode=fast reason=subcommand <x> is not covered by the kit (stock runs)` and stock
  proceeds; an entry that is not the stock console script -> the same line naming the entry.
* the mode cannot be activated (kit files missing or an unknown value) -> the NOT ACTIVE
  line and exit 3 at the trigger: stock never runs under a set CHROMBPNET_OPT that is not off|exact|fast. Kit-internal names found in the
  environment are removed for the run and named (`[chrombpnet-opt] IGNORED names=… reason=…`), never refused.
* the environment route has no --items: `CHROMBPNET_OPT=fast chrombpnet pred_bw` is one regions file per process, as the stock CLI is.

With CHROMBPNET_OPT unset or "off" no finder is installed at all; nothing else is imported until the finder fires.
"""
import os
import subprocess
import sys

ENV = "CHROMBPNET_OPT"
RUN = None                       # the child runner (subprocess.run); a test injects a recorder here before the finder fires
TRIGGERS = ("chrombpnet",)
STOCK_CONSOLE_SCRIPT = "chrombpnet"
SUBCOMMAND = "pred_bw"


class Finder(object):
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
        self.armed = False
        orig = spec.loader.exec_module

        def exec_module(module, _orig=orig):
            _orig(module)                              # the trigger package's own body first, then the arming
            self._fire(module.__name__)
        spec.loader.exec_module = exec_module
        return spec

    def _fire(self, trigger):
        self.fired = trigger
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        act(self.mode, trigger)


def entry(argv=None):
    """(is the stock console script, subcommand or None, entry name) from sys.argv."""
    argv = sys.argv if argv is None else argv
    name = os.path.basename(argv[0]) if argv else ""
    sub = argv[1] if len(argv) > 1 else None
    return name == STOCK_CONSOLE_SCRIPT, sub, name


def act(mode, trigger, environ=None, run=None):
    """The finder's action; `environ` / `run` (the child runner) are injectable for tests. Returns the report when stock proceeds; otherwise
    exits with the run's exit code (cli.run_fast: the exit rule) or 3 at a refusal."""
    import chrombpnet_opt
    from chrombpnet_opt import cli, det, manifest, modes, report as _report, stack
    environ = os.environ if environ is None else environ
    run = run or RUN or subprocess.run
    is_stock, sub, name = entry()
    if mode not in modes.MODES or mode == "off":            # an unknown value: refused at the trigger, exit 3 — the stock never runs under a set CHROMBPNET_OPT that is not a kit mode (off installs no finder)
        _report.print_mode_line({"active": False, "mode": mode, "reason": "unknown {}={!r} (expected {})".format(ENV, mode, "|".join(modes.MODES))})
        sys.exit(3)
    if not is_stock or sub != SUBCOMMAND:
        why = ("subcommand {} is not covered by the kit (stock runs)".format(sub) if is_stock else
               "entry {} is not the stock console script {} (stock runs)".format(name, STOCK_CONSOLE_SCRIPT))
        rep = {"active": False, "mode": mode, "reason": why, "trigger": trigger, "route_label": "env"}
        _report.print_mode_line(rep)
        return rep
    try:
        rep = stack.activate(mode, det=det.requested(environ), strict=True, trigger=trigger, args=sys.argv[2:])
    except chrombpnet_opt.ActivationError:
        sys.exit(3)                                    # the NOT ACTIVE line is printed; stock never runs silently under CHROMBPNET_OPT
    env, stripped, cache = stack.kit_env(environ, rep["det"], rep["kit"], rep["route"], (rep.get("gpu") or {}).get("class"))
    rep["cache_tar"] = cache
    _report.print_ignored(rep)
    op = cli.output_prefix(sys.argv[2:])
    out_dir = manifest.output_dir(op) if op else os.getcwd()
    sys.stdout.flush(); sys.stderr.flush()
    code = cli.run_fast(rep, env, stripped, cache, argv_record=list(sys.argv), out_dir=out_dir, prefixes=[op] if op else [], items_path=None,
                         det_on=bool(rep["det"]), mode=mode, run=run)
    sys.exit(code)                                     # this process owns the exit: the stock's main never runs


def install(environ=None):
    """Install the finder for CHROMBPNET_OPT (idempotent). Returns the finder, or None when nothing is to be done."""
    environ = os.environ if environ is None else environ
    mode = (environ.get(ENV) or "").strip().lower()
    if not mode or mode == "off":
        return None
    for f in sys.meta_path:
        if isinstance(f, Finder):
            return f
    f = Finder(mode)
    sys.meta_path.insert(0, f)
    return f


FINDER = install()
