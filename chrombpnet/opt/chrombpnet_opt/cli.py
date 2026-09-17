"""python -m chrombpnet_opt {pred_bw,check,warm} [--mode off|exact|fast] [--det 0|1 (off only)] ...  (== `chrombpnet-opt ...`)

A thin command layer over the kit's documented line and the stock console script. It never re-implements a lever, a route or a test:

* ``pred_bw`` — the stock pred_bw arguments, passed through verbatim. ``--mode fast`` (the default): the mode line first on stdout, then
               the kit's documented line `python opt/kit_ho/tf/pred_bw_fast.py <args>` (the fast mode's kit directory, stack.kit_home) as a subprocess with the caller's environment minus
               the package's own switches (CHROMBPNET_OPT*), the recipe composed under ``--det`` (det.py), the driver-cache deviation
               (stack.cache_tar_env) and minus the kit's own internal names (registry.KIT_SWITCH_NAMES, never a prefix: found in the
               environment they are removed for the run and named on the IGNORED line — a mode is the whole composition; never a refusal).
               ``--mode off``: `[chrombpnet-opt] NOT ACTIVE mode=off (stock)` first, then the stock child (stock_pred_bw.py run by file
               path, stdlib only, no package module loaded: the stock's own `chrombpnet.CHROMBPNET:main`, the console script's function)
               in a clean subprocess with all kit and package variables
               removed and the PYTHONPATH entries under opt/kit and the tree dropped — proved in the launching process from names (stack.stock_env)
               and again inside the child (`stock_env_proof.json` beside the outputs, embedded in the manifest as env_proof.child); the
               stock arm never refuses on a kit switch, it strips them. Under ``--det`` the same recipe on top (the seed hook is what
               makes the stock deterministic). Exit = the tool's own; 3 when the mode could not be activated. `opt_manifest.json` is
               written beside the outputs (the directory of the -op prefix; of the items file under ``--items``).
               ``--items <file>``: N regions files in one process, the model loaded once — the same entry script with its own ``--items``
               loop and the stock arguments minus -r/-op/-os (one item per line `regions<TAB>output_prefix[<TAB>stats]`); the mode line
               ends `multi=<N>`; refused with ``--mode off``. The default unit of work stays one regions file per process.
* ``check``   — a dry run: the kit's route resolved and gated on this machine (the kit's detect_gpu / resolve; without a GPU, the route
               the kit would take), nothing applied, the mode line printed; exit 3 when NOT ACTIVE. ``--json`` prints the report.
* ``warm``    — one small documented-line job (warm.py): the Triton cache and the driver cache warmed on this machine.

``--mode`` takes off|exact|fast; any other name is a usage error (exit 2).
Exit codes: 0 done, 1 failed or incomplete (outputs the kit listed absent, or fewer items than the items file: ``incomplete: <n>/<m>`` in the
manifest), 2 usage, 3 not active (the mode was refused) or partial — after a fast run the kit's run record (one entry per item, written by
the entry script to the file the package names in ``CHROMBPNET_FASTKIT_RECORD`` and folded into ``opt_manifest.json`` as ``kit_record``) is
read (stack.applied): the forward it ran against the route the kit's tables give for this machine, the shipped driver cache's untar; a lever that fell back is ``partial`` (``partial: [<lever>, ...]`` with the reasons in the manifest; on stdout
``[chrombpnet-opt] NOT ACTIVE: partial activation — levers=<...>: <reasons>; exit 3 (--mode off runs stock)``, report.partial_line)
and the exit is 3 — the backstop of the rule the kit applies up front: a lever of the class's set that cannot start (the K1 stack or
kernels on an H100, the component models absent) makes the kit refuse the mode BY NAME before running anything (``refused`` in the record;
``[chrombpnet-opt] NOT ACTIVE mode=fast reason=the kit refused by name: …; exit 3`` here). The kit's declared rules (a GPU class whose
lever set excludes K1, native dilation off under the recipe) are ``gated``, recorded by name, exit-neutral. Kit-internal names found in the
caller's environment are removed for the run and named (``[chrombpnet-opt] IGNORED names=… reason=…``), never refused. ``warm`` follows ``pred_bw``. The environment route
(``CHROMBPNET_OPT=fast chrombpnet pred_bw ...``) takes the same path after the kit's line (_autoload.act -> run_fast): one record, one exit rule.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import List, Optional

from . import __version__, det, manifest, modes, report as _report, stack, stock_pred_bw


def _kit_or_none():
    try:
        return stack.kit_home()
    except stack.ActivationError:
        return None

COMMANDS = ("pred_bw", "check", "warm")
EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_INACTIVE = 0, 1, 2, 3
OUTPUT_PREFIX_FLAGS = ("-op", "--output-prefix")                  # the stock's pred_bw flag (chrombpnet/parsers.py); the kit's entry script accepts the same


def mode_arg(value: str) -> str:
    """The --mode argument: off | exact | fast; any other name is a usage error."""
    try:
        return modes.check_mode(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def mode_from(args) -> str:
    """--mode when given, else CHROMBPNET_OPT from the environment, else the package default; a --mode that disagrees with a set
    CHROMBPNET_OPT is a usage error. Unknown names come back as given (the activation refuses them by name)."""
    env = (os.environ.get(modes.ENV) or "").strip().lower() or None
    if args.mode and env and args.mode != env:
        raise SystemExit("[chrombpnet-opt] --mode {} disagrees with {}={}: set one (exit {})".format(args.mode, modes.ENV, env, EXIT_USAGE))
    return args.mode or env or modes.DEFAULT_MODE


def det_from(args, mode: str) -> bool:
    """The numerics of this run. The kit modes carry them: `exact` = upstream's determinism settings composed by the package (det.py), `fast` =
    the shipped numerics — `--det` with either is a usage error by name (`exact --det 0` and `fast --det 1` do not exist). `--mode off` takes
    `--det 0|1` (default 0): `off --det 1` = stock run with TensorFlow's own determinism settings (TF32 off, deterministic ops, seed 0), the
    reference `exact` is bit-for-bit equal to; CHROMBPNET_OPT_DET=1 is its environment spelling."""
    given = getattr(args, "det", None)
    m = (mode or "").strip().lower()
    if m in modes.MODES and m != "off":
        if given is not None:
            raise SystemExit("[chrombpnet-opt] --det is a stock-side setting (`--mode off --det 0|1`): mode {} carries its numerics itself ({}) — `{} --det {}` does not exist (exit {})".format(
                m, "upstream's determinism settings, bit-for-bit equal to `--mode off --det 1`" if m in modes.DET_MODES else "the shipped numerics, TF32", m, given, EXIT_USAGE))
        return m in modes.DET_MODES
    return int(given or 0) == 1 or det.requested()


def output_prefix(passthrough: List[str]) -> Optional[str]:
    """The -op / --output-prefix value from the stock arguments (the manifest goes beside the outputs)."""
    for i, t in enumerate(passthrough):
        if t in OUTPUT_PREFIX_FLAGS and i + 1 < len(passthrough):
            return passthrough[i + 1]
        for f in OUTPUT_PREFIX_FLAGS:
            if t.startswith(f + "="):
                return t[len(f) + 1:]
    return None


def cmd_pred_bw(args) -> int:
    mode = mode_from(args)
    det_on = det_from(args, mode)
    passthrough = list(args.args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    op = output_prefix(passthrough)
    items_path = getattr(args, "items", None)                                    # `pred_bw --items <file>`: many items in one process; the manifest beside the items file
    out_dir = manifest.output_dir(op) if op else (os.path.dirname(os.path.abspath(items_path)) if items_path else os.getcwd())
    argv_record = list(sys.argv)
    try:
        m = modes.check_mode(mode)
    except ValueError as e:
        _report.print_mode_line({"active": False, "mode": (mode or "").strip().lower(), "reason": str(e)}); return EXIT_INACTIVE
    if m == "off":
        try:
            kit = stack.kit_home() if det_on else _kit_or_none()
            env, stripped, proof = stack.stock_env(os.environ, det_on, kit=kit, tree=stack.tree_home())
        except (stack.ActivationError, FileNotFoundError) as e:
            _report.print_mode_line({"active": False, "mode": "off", "reason": "--det needs the kit's seed hook: {}".format(e)}); return EXIT_INACTIVE
        assert proof["ok"], proof
        rep = stack.activate("off", det=det_on, route_label="cli", quiet=True, args=passthrough, items=items_path)
        rep["det"] = det_on
        _report.print_mode_line(rep)
        if rep.get("reason"):
            return EXIT_INACTIVE
        proof_path = os.path.join(out_dir, stock_pred_bw.PROOF_FILENAME)
        child = [sys.executable, "-s", os.path.abspath(stock_pred_bw.__file__), "--proof", proof_path] + (["--kit", kit] if kit else []) + \
                ["--tree", stack.tree_home()] + (["--det"] if det_on else []) + ["--"] + passthrough
        rc = subprocess.run(child, env=env).returncode
        try:
            with open(proof_path, "r", encoding="utf-8") as fh:
                child_proof = json.load(fh)
        except (OSError, ValueError):
            child_proof = None
        manifest.write(out_dir, rep, command="pred_bw", argv=argv_record, arm_argv=child, exit_code=rc,
                       det=det.describe(env, det_on, route=None), env_proof={"parent": proof, "child": child_proof}, env_stripped=stripped)
        _report.print_exit("off", route="stock_cli", det=1 if det_on else 0, stripped=",".join(stripped) or "none", child_proof=("ok" if child_proof and child_proof.get("ok") else "missing" if child_proof is None else "NOT CLEAN"), rc=rc)
        return rc
    # ---- exact | fast: the kit's documented line (the mode carries the numerics)
    rep = stack.activate(m, route_label="cli", quiet=True, args=passthrough, items=items_path)
    if not rep.get("active"):
        _report.print_mode_line(rep); return EXIT_INACTIVE
    env, stripped, cache = stack.kit_env(os.environ, det_on, rep["kit"], rep["route"], (rep.get("gpu") or {}).get("class"))
    rep["cache_tar"] = cache
    _report.print_mode_line(rep)
    _report.print_ignored(rep)
    prefixes = stack.multi_prefixes(items_path) if items_path else ([op] if op else [])
    return run_fast(rep, env, stripped, cache, argv_record=argv_record, out_dir=out_dir, prefixes=prefixes, items_path=items_path, det_on=det_on, mode=m)


def run_fast(rep: dict, env: dict, stripped, cache: dict, *, argv_record, out_dir: str, prefixes, items_path: Optional[str],
              det_on: bool, mode: str, run=None) -> int:
    """The fast arm after activation, ONE path for `chrombpnet-opt pred_bw` and the environment route (_autoload.act): the kit's documented
    line (rep["line"]) run as a child and waited for (`run` = subprocess.run; injectable for tests) with the run-record file named in its environment
    (stack.KIT_RECORD_ENV, a temporary file this function owns), the record read back and judged per expected prefix (stack.applied), the
    manifest written beside the outputs with the record folded in (`kit_record`), the partial verdict line printed through report.partial_line, the exit tally
    on stderr. Returns the exit code — the exit rule: the kit refused the mode by name (the record's `refused`: a lever of the class's set
    could not start) -> the NOT ACTIVE line, EXIT_INACTIVE; else the job's own rc; rc 0 with outputs short of the request -> EXIT_FAIL; rc 0
    with a partial activation (the backstop: a forward other than the tables' route in the record) -> EXIT_INACTIVE."""
    run = run or subprocess.run
    argv = rep["line"]
    fd, record_path = tempfile.mkstemp(prefix="chrombpnet_opt_record_", suffix=".json"); os.close(fd)   # the kit's run record lands here, never beside the outputs
    try:
        rc = run(argv, env=dict(env, **{stack.KIT_RECORD_ENV: record_path})).returncode
        record = stack.read_record(record_path) or {}                            # a job that died before writing its record still reaches the EXIT line with its rc
    finally:
        for p in (record_path, record_path + ".tmp"):
            if os.path.exists(p):
                os.remove(p)
    refused = record.get("refused")                                                # the kit refused the mode BY NAME (a lever of this class's set cannot start: its stack, its kernels, its inputs); it ran nothing
    if refused:
        rep.update(active=False, reason="refused by the kit: {}".format(refused), refused=str(refused))
        exit_code = EXIT_INACTIVE
        manifest.write(out_dir, rep, command="pred_bw", argv=argv_record, arm_argv=argv, exit_code=exit_code,
                       det=det.describe(env, det_on, route=rep["route"]), env_stripped=stripped, extra={"job_rc": rc, "kit_record": record})
        print("{} NOT ACTIVE mode={} reason=the kit refused by name: {} (a mode is all of its levers on a class or nothing; --mode off runs stock); exit {}".format(_report.PREFIX, mode, refused, exit_code), flush=True)
        _report.print_exit(mode, route=rep["route"], det=1 if det_on else 0, stripped=",".join(stripped) or "none", cache_tar=cache.get("source"), rc=rc,
                           partial="refused", gated=0, incomplete="none")
        return exit_code
    rep.update(stack.applied(rep, stack.records_by_prefix(record, prefixes)))
    partial = list(rep.get("partial") or [])
    incomplete = rep.get("incomplete")
    exit_code = rc
    if rc == 0 and incomplete:
        exit_code = EXIT_FAIL
    elif rc == 0 and partial:                                                      # a degraded run never reads as success on rc alone
        exit_code = EXIT_INACTIVE
    manifest.write(out_dir, rep, command="pred_bw", argv=argv_record, arm_argv=argv, exit_code=exit_code,
                   det=det.describe(env, det_on, route=rep["route"]), env_stripped=stripped, extra={"job_rc": rc, "kit_record": record})
    if partial:
        _report.print_partial_line("levers={}: {}".format(",".join(partial), "; ".join(rep.get("partial_reasons") or [])), exit_code=EXIT_INACTIVE)
    for g in rep.get("gated") or []:
        print("{} GATED {}".format(_report.PREFIX, g), file=sys.stderr, flush=True)
    _report.print_exit(mode, route=rep["route"], det=1 if det_on else 0, stripped=",".join(stripped) or "none", cache_tar=cache.get("source"), rc=rc,
                       partial=",".join(partial) or "none", gated=len(rep.get("gated") or []), incomplete=incomplete or "none")
    return exit_code


def cmd_check(args) -> int:
    mode = mode_from(args)
    rep = stack.activate(mode, det=det_from(args, mode), dry_run=True, route_label="cli")
    _report.print_ignored(rep)
    if args.json:
        print(json.dumps(rep, indent=1, default=str))
    if not rep.get("active"):
        return EXIT_OK if rep.get("mode") == "off" and not rep.get("reason") else EXIT_INACTIVE
    return EXIT_OK


def cmd_warm(args) -> int:
    from . import warm
    mode = mode_from(args)
    try:
        m = modes.check_mode(mode)
    except ValueError as e:
        _report.print_mode_line({"active": False, "mode": (mode or "").strip().lower(), "reason": str(e)}); return EXIT_INACTIVE
    if m == "off":
        print("{} warm: nothing to warm on the stock arm; use --mode fast | exact".format(_report.PREFIX), file=sys.stderr)
        return EXIT_USAGE
    det_from(args, m)                                                          # --det with a kit mode is a usage error by name
    r = warm.run(m, regions=args.regions, n=args.n, out_dir=args.out_dir, timeout=args.timeout)
    return EXIT_OK if r["ok"] else (EXIT_INACTIVE if r.get("inactive") else EXIT_FAIL)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="chrombpnet-opt", description=__doc__.splitlines()[0], allow_abbrev=False)
    ap.add_argument("--version", action="version", version="chrombpnet_opt {}".format(__version__))
    sub = ap.add_subparsers(dest="command")
    sub.required = True

    def common(p):
        p.add_argument("--mode", type=mode_arg, default=None, metavar="off|exact|fast", help="fast (default): the shipped numerics | exact: upstream's determinism settings, bit-for-bit equal to `--mode off --det 1` | off: stock; default: {} from the environment, else {}".format(modes.ENV, modes.DEFAULT_MODE))
        p.add_argument("--det", type=int, choices=(0, 1), default=None, metavar="0|1", help="--mode off only: --det 1 runs stock with TensorFlow's determinism settings (TF32 off, deterministic ops, seed 0; == {}=1) — the reference `exact` equals; --det 0 (default): stock as shipped".format(modes.ENV_DET))

    p = sub.add_parser("pred_bw", help="the stock pred_bw arguments, passed through verbatim (everything the package does not own)", allow_abbrev=False)
    common(p)
    p.add_argument("--items", default=None, metavar="FILE", help="many items in one process: one `regions<TAB>output_prefix[<TAB>stats]` line per item (the stock -r/-op/-os of that item); the other stock arguments once on the line; exact | fast only")
    p.set_defaults(fn=cmd_pred_bw)

    p = sub.add_parser("check", help="dry run: resolve and gate the mode on this box, apply nothing"); common(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("warm", help="one small documented-line job: the caches warmed on this box"); common(p)
    p.add_argument("--regions", default=None, help="a regions BED (default: CHROMBPNET_OPT_REGIONS); its first --n rows are the job")
    p.add_argument("--n", type=int, default=None, help="rows of the regions file to run (default: the stock batch size, stock parsers.py:223)")
    p.add_argument("--out", dest="out_dir", default=None); p.add_argument("--timeout", type=float, default=1800.0)
    p.set_defaults(fn=cmd_warm)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    try:
        args, extras = ap.parse_known_args(argv)           # pred_bw: the stock arguments are whatever the package does not own (passed verbatim)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else EXIT_USAGE
    if args.command == "pred_bw":
        args.args = extras
    elif extras:
        ap.error("unrecognized arguments: {}".format(" ".join(extras)))
    try:
        return int(args.fn(args))
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr); return EXIT_USAGE
        return int(e.code or 0)
