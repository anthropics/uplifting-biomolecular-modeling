"""python -m af2ig_opt {pred,check,warm} [--mode off|exact|fast] ...

A thin command layer over the kit's driver and the stock line. It never re-implements a lever, a mode or a test:

* ``pred``   — a directory of binder/target complexes (one PDB per design, binder = the first chain) predicted into ``--out``: the driver's
               own outputs ``<out>/pdbs/<tag>_af2pred.pdb``, ``<out>/out.sc``, ``<out>/check.point`` and nothing else; its per-design timer records
               (``-timers``, a file in a temporary directory removed after the run) are relayed to stdout line by line as they land. ``--mode exact|fast``: the driver's process with the mode's flags appended
               (modes.resolve: the kit's own composition with the lever values substituted), the mode expressed once, in argv.
               ``--mode off``: the stock line — the same driver with no lever flag — through the stock caller ``stock_cli.py`` in a
               clean subprocess that prints its clean-process census before the driver starts. Both arms get the caller's environment minus
               the package's switches, the kit's two environment levers and ``CUDA_MPS_*`` (``stock/PINS.json`` "stock_environment";
               the names stripped are on the STOCK line); the kit's own ``AF2IG_DIR`` / ``AF2_PARAMS`` and upstream's XLA/JAX variables pass
               through. ``--det 1`` applies the deterministic recipe (det.py) on either arm.
* ``check``  — the activation line for the mode on this box without running anything: the flags as they would run, the levers, every
               gate's verdict, the GPU; ``--json`` prints the full report. Exit 3 when the mode would be refused.
* ``warm``   — one public complex through the stock line, outputs discarded (warm.py): the kit's warm-up process.

Exit codes: 0 done, 1 the driver failed or the run is incomplete (fewer designs than inputs; ``items=<n>/<m>`` on the EXIT line),
2 usage, 3 not active (a gate refused; the stock caller's proof refused) or partial (a lever of the mode without the driver's evidence of
it in its timer records — stack.applied; ``partial=<lever>,...`` on the EXIT line; a lever of the mode that CANNOT RUN in the driver's process — jax below the Pallas floor, the kernel
not importable — makes the driver refuse by name before any design, relayed as exit 3: a mode is all of its levers; ``--allow-partial`` records ``allow_partial=on`` and
proceeds with the run's own exit).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import List, Optional, Tuple

from opt_core import report as _core_report, stock_proof as _core_stock_proof

from . import registry, TAG, __version__, det, manifest, modes, report as _report, seed as _seed, stack, stock_cli

COMMANDS = ("pred", "check", "warm")
EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_INACTIVE = _core_report.EXIT_OK, _core_report.EXIT_FAIL, _core_report.EXIT_USAGE, _core_report.EXIT_NOT_ACTIVE
TIMERS_FILE = "timers.jsonl"                                  # the driver's -timers file, in a temporary directory of the run (removed after it), relayed line by line to stdout while it grows


class Relay(threading.Thread):
    """Prints every line the driver appends to its timers file (its own JSON records, verbatim) on this process's stdout as it lands, so a
    log reader sees the per-design record live; the file stays beside the outputs."""

    def __init__(self, path: str, stream=None):
        super().__init__(daemon=True); self.path, self.stream, self.stop, self.n = path, stream or sys.stdout, threading.Event(), 0

    def run(self):
        pos = 0
        while True:
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    fh.seek(pos)
                    for line in fh:
                        if not line.endswith("\n"):
                            break
                        pos += len(line.encode("utf-8")); self.n += 1
                        print(line.rstrip("\n"), file=self.stream, flush=True)
            except FileNotFoundError:
                pass
            if self.stop.is_set():
                return
            self.stop.wait(0.2)

    def finish(self):
        self.stop.set(); self.join(timeout=5.0)


def mode_from(args) -> str:
    """--mode when given, else AF2IG_OPT from the environment, else the package default; a --mode that disagrees with a set AF2IG_OPT is a
    usage error."""
    env = (os.environ.get(modes.ENV) or "").strip().lower() or None
    if args.mode and env and args.mode != env:
        raise SystemExit(f"{_report.PREFIX} --mode {args.mode} disagrees with {modes.ENV}={env}: set one (exit {EXIT_USAGE})")
    mode = args.mode or env or modes.DEFAULT_MODE
    if mode not in modes.MODES:
        raise SystemExit(f"{_report.PREFIX} unknown mode {mode!r} (expected {'|'.join(modes.MODES)}) (exit {EXIT_USAGE})")
    return mode


def allow_partial_from(args) -> bool:
    """--allow-partial: the one explicit, typed escape of a partial activation (no environment form)."""
    return bool(getattr(args, "allow_partial", False))


def driver_args(in_dir: str, out_dir: str, params: str, timers: str) -> List[str]:
    """The driver's I/O switches, the same on every arm: the outputs laid out under <out> (manifest.PDB_DIR / SCORE_FILE / CHECKPOINT)."""
    return ["-pdbdir", in_dir, "-outpdbdir", os.path.join(out_dir, manifest.PDB_DIR), "-scorefilename", os.path.join(out_dir, manifest.SCORE_FILE),
            "-checkpoint_name", os.path.join(out_dir, manifest.CHECKPOINT), "-af2_dir", params, "-timers", timers]


def child_env(det_level: Optional[int]) -> Tuple[dict, List[str]]:
    """The caller's environment minus the must-be-absent names (stock/PINS.json stock_environment), the recipe set under --det."""
    prefixes, _, _ = stock_cli.stock_environment(stack.pins())
    env, stripped = _core_stock_proof.strip_env(os.environ, prefixes)
    if det_level is not None:
        env.update(det.env(det_level, env))
    return env, stripped


def count_inputs(in_dir: str) -> int:
    return len(glob.glob(os.path.join(in_dir, "*.pdb")))


PRECOMPILE_DEFAULT_N = ("default N",)                             # argparse's const for a bare `--precompile`: a sentinel object (not a str: argparse leaves it unconverted; not an int: no typed N can equal it)


def precompile_from(args, mode: str):
    """`--precompile [N]`, the thread count of L7 (L7 itself is composed on every kit mode): None = switch absent (the default N); a bare `--precompile` = the default N (returned as 0, the mode table's spelling); N >= 1 = that many threads; N < 1 typed by the user, off and big are usage refusals."""
    v = getattr(args, "precompile", None)
    if v is None:
        return None, True
    if mode == "off":                                            # every kit mode precompiles (fast == big): only stock takes no lever
        print(f"{_report.PREFIX} --precompile: a lever of the kit modes only (mode {mode!r}: " + ("stock takes no lever" if mode == "off" else "the memory mode never precompiles in parallel") + ")", file=sys.stderr); return None, False
    if v is PRECOMPILE_DEFAULT_N:                                # `--precompile` with no N: the default N (PRECOMPILE_THREADS or its override)
        return 0, True
    if v < 1:                                                   # every typed N < 1 (0, -1, …) is refused: an int never equals the string sentinel
        print(f"{_report.PREFIX} --precompile {v}: N must be >= 1 (or omitted for the default N)", file=sys.stderr); return None, False
    return v, True


def no_compile_from(args) -> None:
    """``--no-compile`` = the tree's ``MODEL_OPT_LEVERS_OFF=compile`` (QoL alias): the word joins the ablation switch of this process (and so the driver child's);
    af2ig has no kit compile lever, so modes.resolve drops nothing and the activation line names it (report._compile)."""
    if getattr(args, "no_compile", False):
        cur = [w for w in os.environ.get(registry.LEVERS_OFF_ENV, "").split(",") if w]
        if registry.COMPILE_WORD not in cur:
            os.environ[registry.LEVERS_OFF_ENV] = ",".join(cur + [registry.COMPILE_WORD])


def cmd_pred(args) -> int:
    mode = mode_from(args); no_compile_from(args)
    in_dir, out_dir = os.path.abspath(args.in_dir), os.path.abspath(args.out_dir)
    if not os.path.isdir(in_dir):
        print(f"{_report.PREFIX} --pdbdir {in_dir}: not a directory", file=sys.stderr); return EXIT_USAGE
    n_in = count_inputs(in_dir)
    if n_in == 0:
        print(f"{_report.PREFIX} --pdbdir {in_dir}: no *.pdb file", file=sys.stderr); return EXIT_USAGE
    precompile, ok = precompile_from(args, mode)
    if not ok:
        return EXIT_USAGE
    rep = stack.activate(mode, route="cli", precompile=precompile, det=args.det)
    refused = rep.get("gates") is None or any(not g["ok"] and not g.get("forced") and not g.get("aside") for g in rep["gates"].values())   # every unmet, unforced gate refuses (the tiles gate too: a mode is all of its levers)
    if refused or (mode != "off" and not rep.get("active")):
        _report.log_activation(rep); return EXIT_INACTIVE
    timers_dir = tempfile.mkdtemp(prefix="af2ig_opt_timers_")        # the driver's records live outside <out>: <out> holds the driver's own outputs only
    timers = os.path.join(timers_dir, TIMERS_FILE)
    dargs = driver_args(in_dir, out_dir, stack.params_dir(), timers) + list(args.extra or [])
    env, stripped = child_env(args.det)
    if mode != "off":                                            # the deployment lever (ccache): JAX's persistent compilation cache in the driver child's environment — every kit mode; never the stock line (a must-be-absent name there, stripped above)
        env.update(rep.get("cache_env") or {})
        env[modes.ENV_TIER] = mode if mode in modes.TIER_WORDS else "exact"   # the provider tier word the kernel adapters bind with (exact binds nothing)
        env.update(rep.get("l19_env") or {})                       # L19: AF2IG_OPT_TRIATTN_CORE_DTYPE=bf16 on fast / big — before the caller's own words, which win
        env.update(stack.lever_words_env())                      # L10's declared bridge words (length row, core operand dtype, named row) reach the driver child, which reads them (af2ig_opt.triattn); stripped above with every AF2IG_OPT* name
    if rep.get("memory_env"):                                    # the memory line's environment (the XLA pool fraction): big
        env.update(rep.get("memory_env") or {})
    driver = os.path.join(stack.af2ig_dir(), stack.DRIVER)
    os.makedirs(out_dir, exist_ok=True)
    seeded = None
    if args.seed:                                            # L1: the seed switch on a seeded COPY of the checkout in a temp dir (the pinned checkout is never touched; removed after the run); --seed 0 IS the unseeded driver, bitwise (seed.py) — no copy is built, no flag is passed, the run is the mode's own line
        try:
            seeded = _seed.seeded_dir(stack.af2ig_dir())
        except RuntimeError as e:
            print(f"{_report.PREFIX} {e}", file=sys.stderr); return EXIT_FAIL
        driver = seeded["driver"]; dargs += [_seed.FLAG, str(args.seed)]
        rep["seed"] = args.seed
    if args.det is not None:
        det.print_line(args.det, env)
    relay = Relay(timers); relay.start()
    t0 = time.time()
    try:
        if mode == "off":
            child = [sys.executable, "-I", "-m", "af2ig_opt.stock_cli", "--pins", os.path.join(stack.tree_home(), "stock", "PINS.json"),
                     "--det", str(args.det or 0), "--python", sys.executable, "--driver", driver, "--"] + dargs
            rep["line_run"] = "none (the driver with no lever flag)"
            print(f"{_report.PREFIX} STOCK line={' '.join(dargs)!r} det={'level=' + str(args.det) if args.det else 'off'} stripped={','.join(stripped) or 'none'}"
                  + (f" seed={args.seed}" if seeded else ""), file=sys.stderr, flush=True)
            rc = subprocess.run(child, env=env).returncode
        else:
            _report.log_activation(rep)
            child = [sys.executable, "-I", "-u", driver] + dargs + list(rep["resolution"].flags)
            rc = subprocess.run(child, env=env).returncode
    finally:
        if seeded:
            _seed.remove(seeded)                             # the seeded copy is gone once the child has exited (no file beyond the run's outputs)
    wall = time.time() - t0
    relay.finish()
    items = manifest.read_items(out_dir)
    n_ok = sum(1 for it in items if it["ok"]) if items else 0
    incomplete = f"{n_ok}/{n_in}" if n_ok < n_in else None            # outputs short of the request: the run failed (exit 1), never a partial activation
    allow_partial = allow_partial_from(args)
    if mode != "off":
        rep.update(stack.applied(rep, timers)); rep["allow_partial"] = allow_partial
    run_rc = EXIT_OK if rc == 0 else (EXIT_INACTIVE if rc == stock_cli.EXIT_REFUSED and (mode == "off" or n_ok == 0) else EXIT_FAIL)   # the child's status in this package's codes: the stock caller's refusal, or a lever of the mode that cannot run in the driver's process (_fused.refuse: named, before any design), is 3; any other failure 1
    v = _core_report.verdict(run_rc, rep, allow_partial, incomplete)   # the shared core's exit rule: a failed child keeps its code; outputs short turn a 0 into 1; unevidenced levers turn a 0 into 3 unless --allow-partial (recorded)
    _report.print_lever_lines(rep)                                      # one evidence line per lever per arm, from the driver's own records
    _report.print_peak_lines(stack.timer_records(timers))            # per-design allocator peak (PEAK item=<tag> inuse_gib=…) + the pass note (PEAK-NOTE), from the driver's design records; both arms
    partial_line = _core_report.partial_line(TAG, v, rep.get("partial_reasons"))   # the core's one partial line: `NOT ACTIVE: partial activation — … exit 3` when that IS the code, `PARTIAL allowed: …` when recorded; a run that failed on its own names the levers on the LEVER and EXIT lines only
    if partial_line:
        print(partial_line, file=sys.stderr, flush=True)
    _report.print_exit(mode, route="cli", items=f"{n_ok}/{n_in}", det="on" if args.det else "off", driver_rc=rc, wall_s=f"{wall:.1f}",
                       partial=",".join(v["partial"]) or "none", allow_partial="on" if allow_partial else "off", **({"seed": args.seed} if seeded else {}))
    shutil.rmtree(timers_dir, ignore_errors=True)                      # the driver's records were relayed to stdout; nothing of them stays on disk
    return v["exit_code"]


def cmd_check(args) -> int:
    mode = mode_from(args)
    precompile, ok = precompile_from(args, mode)
    if not ok:
        return EXIT_USAGE
    no_compile_from(args)
    rep = stack.activate(mode, route="cli", dry_run=True, precompile=precompile, det=getattr(args, "det", None))
    _report.log_activation(rep)
    if args.json:
        print(json.dumps({k: v for k, v in rep.items() if k != "resolution"}, indent=1, default=str))
    return EXIT_INACTIVE if rep.get("would_refuse") else EXIT_OK


def cmd_warm(args) -> int:
    from . import warm
    mode = mode_from(args); no_compile_from(args)
    if args.lengths:                                              # warm the caches — synthetic complexes of the given total lengths through THIS mode's line, so the compilation cache (ccache) and the program store (L13) hold those lengths before the first real run
        r = warm.run_lengths(warm.parse_lengths(args.lengths), mode=mode, det=args.det, out_dir=args.out_dir)
    else:
        r = warm.run(out_dir=args.out_dir, det=args.det)
    return EXIT_OK if r["ok"] else (EXIT_INACTIVE if r.get("inactive") else EXIT_FAIL)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="af2ig-opt", description=__doc__.splitlines()[0])
    ap.add_argument("--version", action="version", version=f"af2ig_opt {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--mode", choices=modes.MODES, default=None, help=f"default: {modes.ENV} from the environment, else {modes.DEFAULT_MODE}")
        p.add_argument("--no-compile", dest="no_compile", action="store_true", help=f"alias of {registry.LEVERS_OFF_ENV}=compile (the tree's word): af2ig has no kit compile lever — JAX jit per distinct length is stock behaviour — so nothing is dropped and the activation line says compile=stock_jit(no-compile:…)")

    p = sub.add_parser("pred", help="predict a directory of binder/target complexes"); common(p)
    p.add_argument("--pdbdir", dest="in_dir", required=True); p.add_argument("--out", dest="out_dir", required=True)
    p.add_argument("--det", type=int, choices=sorted(det.LEVELS), default=None, help="the deterministic recipe (both arms): 1 = XLA_FLAGS=--xla_gpu_autotune_level=0")
    p.add_argument("--precompile", type=int, nargs="?", const=PRECOMPILE_DEFAULT_N, default=None, metavar="N", help=f"L7 (exact | fast): before the loop, featurise one input per distinct length and compile them all from N host threads (N omitted = {modes.PRECOMPILE_THREADS}); pays only in a process that runs several distinct lengths — at one design it costs a discarded forward")
    p.add_argument("--seed", type=int, default=None, help=f"L1, the seed lever: the driver's `{_seed.FLAG} N` (TF feature pipeline random_seed + the model call's PRNGKey) on a temporary seeded copy of the checkout "
                   f"(removed after the run); 0 = the unseeded driver, bitwise. With --mode off this is the named supplementary stock arm `af2ig-seeded` (not the stock line)")
    p.add_argument("--allow-partial", action="store_true", help=f"accept a run whose activation is partial (a lever of the mode without the driver's evidence of it in its records); "
                   f"recorded on the EXIT line. Without it a partial run exits {EXIT_INACTIVE} (the outputs stay)")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="after `--`: the driver's own settings switches (-recycle N, -runlist F, ...), never a lever flag")
    p.set_defaults(fn=cmd_pred)

    p = sub.add_parser("check", help="dry run: resolve and gate the mode on this box, run nothing"); common(p)
    p.add_argument("--json", action="store_true")
    p.add_argument("--precompile", type=int, nargs="?", const=PRECOMPILE_DEFAULT_N, default=None, metavar="N")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("warm", help="the kit's warm-up process: one public complex through the stock line, outputs discarded"); common(p)
    p.add_argument("--out", dest="out_dir", default=None)
    p.add_argument("--det", type=int, choices=sorted(det.LEVELS), default=None)
    p.add_argument("--lengths", default=None, metavar="L1,L2,…", help="warm the caches instead: one synthetic two-chain complex per total length through the --mode line (ccache + the L13 program store then hold those lengths; --det names the recipe namespace)")
    p.set_defaults(fn=cmd_warm)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else EXIT_USAGE
    undeclared = stack.undeclared_env()
    if undeclared:
        print(f"{_report.PREFIX} undeclared {modes.ENV}* variable(s) {undeclared}: the package reads {list(stack.DECLARED_ENV)} — a mistyped switch never runs silently (exit {EXIT_USAGE})", file=sys.stderr); return EXIT_USAGE
    if getattr(args, "extra", None) and args.extra[:1] == ["--"]:
        args.extra = args.extra[1:]
    if getattr(args, "extra", None):
        bad = [a for a in args.extra if a in stock_cli.FORBIDDEN_FLAGS]
        if bad:
            print(f"{_report.PREFIX} {bad} are lever flags: a mode names them (--mode), never the extra arguments", file=sys.stderr); return EXIT_USAGE
    try:
        return int(args.fn(args))
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr); return EXIT_USAGE
        return int(e.code or 0)
