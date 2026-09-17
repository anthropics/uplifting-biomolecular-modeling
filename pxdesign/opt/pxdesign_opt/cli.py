"""python -m pxdesign_opt {design,check,warm} [--mode exact|fast|big|off] ...

A thin command layer over the kit code and the upstream CLI. It never re-implements a lever, a mode or a test:

* ``design`` — one design job (a task file x the seeds x N_sample) at upstream's own options (options.py), written in the upstream dump
               layout plus ``opt_manifest.json``. ``--mode off``: the stock caller ``stock_infer.py`` in a clean subprocess (every variable
               of the kit stripped and proved absent, no kit directory on sys.path): ``--det 0`` execs the upstream console script
               (route ``cli``), ``--det 1`` runs the in-process upstream route (route ``inprocess``); ``LAYERNORM_TYPE`` reaches that
               process as the caller's environment has it (configs/<gpu>.env exports ``fast_layernorm``). Kit modes: ``pxdesign_opt.enable(mode)``,
               then the in-process route (infer_loop.py: upstream's ``main()`` replayed with the seeding call parameterised) — upstream's
               own ``InferenceRunner`` loads the checkpoint and the kit's ``install(model)`` runs at that point through the package's hook.
               Under ``--det 1`` stock and the kit modes run the in-process route and differ in the mode flag alone; under ``--det 0``
               they differ in the entry path as well (stock: the console script in a child process; kit modes: the replay), not in any
               option — every shared option is passed explicitly on every mode, through upstream's own ``build_argv``.
* ``check``  — the activation line for a mode without applying anything: kit bytes, GPU class and memory, the switches the mode
               would export, and the stock pins REPORT (the installed upstream packages against the tree's copy — `stack.pins_report`,
               one ``PINS …`` line); a kit mode whose activation would refuse on this box — an installed upstream that is not the
               pinned stock included (`stack._stock_gate`; ``off`` is not gated) — says so (``would_refuse=…``) and exits 3, as
               ``design`` would; ``--json`` prints the full report with the lever table.
* ``warm``   — one design on a public input through ``design`` so the fast-LayerNorm extension is built (warm.py).

Exit codes (report.py, the shared core's; a design job's code is the core's exit rule `opt_core.report.verdict` — `_conclude`): 0 ok |
1 warm failed, or a design job whose outputs fall short of tasks x seeds x N_sample (named on the INCOMPLETE line) | 2 usage | 3 levers
not active (reason printed: cannot engage, lever application incomplete, or the exit census names a planned lever that did not do its work;
`check`: the mode would refuse on this box), or the stock child's environment proof failed | otherwise the run's own exit code. `opt_manifest.json` is written after the rule with the code the
verb returns; a job whose every (task, seed) pair upstream skipped as already dumped prints NOTHING RAN and writes no manifest (outputs.py); a
run that raises (an out-of-memory error, any error of its own) propagates as raised — its traceback is the record, no manifest, no run line.
"""
from __future__ import annotations

from . import core_gate

core_gate()                                              # statement one here too: `python -m pxdesign_opt.cli` runs this module directly (its __main__ block); [pxdesign-opt] NOT ACTIVE: reason=…  -> exit 3

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

from opt_core import report as _core_report  # noqa: E402

from . import manifest, options, report  # noqa: E402
from .modes import DEFAULT_MODE, MODES, check_mode  # noqa: E402

from .report import EXIT_FAIL, EXIT_NOT_ACTIVE, EXIT_OK, EXIT_USAGE  # noqa: E402  — the shared core's codes (opt_core.report); 1 also = outputs short of tasks x seeds x N_sample, 3 also = the stock child's environment proof failed

PROG = "python -m pxdesign_opt"

USAGE = f"""usage: {PROG} <command> [--mode exact|fast|big|off] ...

commands
  design  [--mode M] -i FILE -o DIR [--seeds S[,S…]] [--N_step 400] [--N_sample 5] [--dtype bf16] [--eta_type const]
          [--eta_min 2.5] [--eta_max 2.5] [--num_workers 16] [--use_msa true] [--use_fast_ln true] [--load_checkpoint_dir DIR]
          [--det 0|1] [<any further pxdesign infer argument>]
                                          one design job at upstream's own options (upstream's names and defaults: -i/--input = the task
                                          list, JSON or YAML; -o/--dump_dir; --seeds absent = upstream derives one seed from the clock, on
                                          every mode; --tasks / --out_dir / --ckpt_dir are accepted spellings of -i / -o / --load_checkpoint_dir;
                                          an argument this verb does not name passes to pxdesign infer unchanged): mode off = the stock caller
                                          in a clean subprocess (upstream's console script under --det 0, the in-process upstream route under
                                          --det 1); mode exact | fast | big = the levers installed after checkpoint load, in-process
                                          route. On every mode
                                          LAYERNORM_TYPE is set from --use_fast_ln before Protenix reads it (true, the default: fast_layernorm;
                                          false: unset = the plain, non-fused LayerNorm) — the LayerNorm upstream's own flag names. Writes the upstream
                                          dump layout <dump_dir>/<task>/seed_<s>/predictions/*.cif + opt_manifest.json (+ stock_env_proof.json
                                          for off); one KERNELS evidence line on stderr on every route (the stack's versions and
                                          switches, the model's LayerNorm kind: stamps.py)
  check   [--mode M] [--json]             resolve and gate the mode on this box; applies nothing (rc 0 when activation would proceed)
  warm    [--mode M] [--out_dir DIR] [--timeout S] [--N_sample 1]
                                          one design on a public input through `design` (fast-LN extension build + first sampler call)
the mode defaults to {DEFAULT_MODE}; PXDESIGN_OPT in the environment must agree with --mode when both are given.
exit: 0 ok | 1 warm failed, or outputs short of tasks x seeds x N_sample (the INCOMPLETE line) | 2 usage | 3 levers not active (cannot engage, or a planned lever did not do its work: the NOT ACTIVE line; `check`: the mode would refuse here), or the stock child's environment proof failed
"""


def _mode_arg(ap):
    ap.add_argument("--mode", default=None, help=f"{'|'.join(MODES)} (default {DEFAULT_MODE}, or PXDESIGN_OPT)")


def _resolve_mode(args) -> str:
    """--mode or PXDESIGN_OPT (they must agree), else the default; the name is the mode table's to accept (modes.check_mode: exactly
    modes.MODES; any other name is refused by name with the table's one sentence) — a usage error otherwise."""
    env = (os.environ.get("PXDESIGN_OPT") or "").strip().lower() or None
    mode = (args.mode or "").strip().lower() or None
    if mode and env and mode != env:
        print(f"{report.PREFIX} --mode {mode} disagrees with PXDESIGN_OPT={env}", file=sys.stderr)
        raise SystemExit(EXIT_USAGE)
    try:
        return check_mode(mode or env or DEFAULT_MODE)
    except ValueError as e:
        print(f"{report.PREFIX} {e}", file=sys.stderr)
        raise SystemExit(EXIT_USAGE)


def build_parser():
    ap = argparse.ArgumentParser(prog=PROG, usage=USAGE, add_help=True)
    sub = ap.add_subparsers(dest="command")
    d = sub.add_parser("design"); _mode_arg(d)
    d.add_argument("-i", "--input", "--tasks", dest="tasks", required=True, metavar="FILE")            # upstream's -i/--input (the task list); --tasks is an accepted spelling
    d.add_argument("-o", "--dump_dir", "--out_dir", dest="out_dir", required=True, metavar="DIR")      # upstream's -o/--dump_dir; --out_dir is an accepted spelling
    options.add_arguments(d)                                                    # upstream's knobs, their names and defaults, + --seeds
    d.add_argument("--det", type=int, choices=(0, 1), default=0)
    d.add_argument("--load_checkpoint_dir", "--ckpt_dir", dest="ckpt_dir", default=None, metavar="DIR")   # upstream's name (default $PXDESIGN_CKPT_DIR, infer_loop.preflight); --ckpt_dir is an accepted spelling
    d.set_defaults(extra=[])                                                    # any argument the verb does not name passes to pxdesign infer unchanged (main: parse_known_args; a `--` separator is accepted and dropped)
    c = sub.add_parser("check"); _mode_arg(c); c.add_argument("--json", action="store_true")
    w = sub.add_parser("warm"); _mode_arg(w); w.add_argument("--out_dir", default=None); w.add_argument("--timeout", type=float, default=1800)
    w.add_argument("--N_sample", type=int, default=1); w.add_argument("--no-echo", action="store_true")
    return ap


# ------------------------------------------------------------------------------------------------------------------ design

def _clean_env(pins: dict, det: int = 0, environ=None) -> dict:
    """The stock subprocess environment: the caller's minus every forbidden-prefix variable and every package variable
    (`opt_core.stock_proof.strip_env`), no kit directory on PYTHONPATH (`strip_pythonpath`), plus the recipe's variables under --det 1
    (det.apply_env: the child proves they hold the recipe's values)."""
    from opt_core import stock_proof
    from . import det as _det, stack
    se = pins["stock_environment"]
    prefixes = list(se["must_be_absent_prefixes"]) + ["PXDESIGN_OPT"]
    env, _dropped = stock_proof.strip_env(os.environ if environ is None else environ, prefixes, names=list(_det.RECIPE_ENV))
    pp, _removed = stock_proof.strip_pythonpath(env.get("PYTHONPATH"), stack.kit_dirs())
    if pp:
        env["PYTHONPATH"] = pp
    else:
        env.pop("PYTHONPATH", None)
    _det.apply_env(det, env)
    return env


def stock_command(tree: str, tasks: str, out_dir: str, *, opts, det: int, ckpt_dir=None, extra=(), python: str = None) -> list:
    """The stock caller's command line (the one place it is built): ``<python> -s -m pxdesign_opt.stock_infer --tree … --tasks … --out_dir …
    --det … [--seeds …] [the caller's given knobs, verbatim] [--ckpt_dir …] [--extra …]``."""
    cmd = [python or sys.executable, "-s", "-m", "pxdesign_opt.stock_infer", "--tree", tree, "--tasks", tasks, "--out_dir", out_dir, "--det", str(det)]
    cmd += options.given_argv(opts)
    if ckpt_dir:
        cmd += ["--ckpt_dir", ckpt_dir]
    if extra:
        cmd += ["--extra", *extra]
    return cmd


def cmd_design(args) -> int:
    from . import stack, outputs, infer_loop
    mode = _resolve_mode(args)
    pins = stack.read_pins()
    extra = list(getattr(args, "extra", None) or [])
    out_dir = os.path.abspath(args.out_dir)
    tasks = os.path.abspath(args.tasks)
    t0 = time.perf_counter()
    st = options.from_args(args, pins=pins)
    n_sample = int(st.values["N_sample"])
    if mode == "off":
        cmd = stock_command(stack.tree_home(), tasks, out_dir, opts=st, det=args.det, ckpt_dir=args.ckpt_dir, extra=extra)
        env = _clean_env(pins, det=args.det)                                         # the kit's variables stripped; everything else as the caller's environment has it
        options.apply_layernorm_env(st, env)                                          # LAYERNORM_TYPE from --use_fast_ln: the rule the kit modes apply, on the stock child's environment (Protenix reads it at import)
        rep = stack.activate("off")
        existed = outputs.dumped_pairs(out_dir)                                       # before the run: the pairs upstream will skip as already dumped (its resume rule) — the one source of skipped_existing
        report.log(f"stock caller: {' '.join(cmd)}")
        rc = subprocess.call(cmd, env=env)
        proof_path = os.path.join(out_dir, outputs.STOCK_PROOF_NAME)
        proof = None
        if os.path.isfile(proof_path):
            with open(proof_path, "r", encoding="utf-8") as fh:
                proof = json.load(fh)
        summ = outputs.summarize(out_dir, tasks, list(st.seeds) or ((proof or {}).get("run") or {}).get("seeds") or (proof or {}).get("seeds_run"), n_sample, existed=existed)   # the seeds given, else the one upstream derived (the proof's run record / the console script's seed banner), else unknown (scope dir)
        timings = _timings(summ, time.perf_counter() - t0, (proof or {}).get("run", {}).get("load_s") if proof else None)
        v = _conclude("off", rc, summ, [], timings, out_dir,
                      notes=[f"{outputs.STOCK_PROOF_NAME} written by this invocation's stock child"] if proof else None)   # the child always writes its environment proof into -o: named when nothing else of this invocation is there
        if not summ["nothing_ran"]:                                                   # a job upstream skipped whole leaves the earlier run's manifest as written (no claim over outputs this process did not produce)
            manifest.write(out_dir, rep, mode="off", command="design", argv=sys.argv[1:], exit_code=v["exit_code"], options=st.as_dict(), outputs=summ, timings=timings,
                           stock_proof={"path": proof_path, **{k: proof.get(k) for k in ("det", "route", "layernorm_class", "layernorm_kind", "layernorm_source", "layernorm_env", "deviations", "violations")}}
                           if proof else {"path": proof_path, "missing": True}, stamps=(proof or {}).get("stamps"), extra={"incomplete": v["incomplete"]})
        return v["exit_code"]
    try:
        stack.activate(mode, strict=True)
    except stack.ActivationError:
        return EXIT_NOT_ACTIVE
    pre = infer_loop.preflight(pins, args.ckpt_dir, out_dir)                    # the deployment gate: the weights directory and the CCD cache hold the files upstream requires (never downloaded here)
    options.apply_layernorm_env(st, os.environ)                                   # LAYERNORM_TYPE from --use_fast_ln before Protenix is imported (upstream sets it after the import; the same rule on every mode)
    ckpt_dir = os.path.dirname(pre["checkpoint"])
    argv = infer_loop.upstream_argv(tasks, out_dir, ckpt_dir, st.argv(), st.seeds_arg(), extra)
    existed = outputs.dumped_pairs(out_dir)                                       # before the run: the pairs upstream will skip as already dumped (its resume rule) — the one source of skipped_existing
    rc, res, stamps_rec = EXIT_OK, None, None
    try:
        res = infer_loop.run(argv, list(st.seeds), args.det, log=report.log)
        stamps_rec = infer_loop.split_stamps(res)
    except stack.ActivationError as e:                                          # a lever of the mode's set could not run: the hook printed NOT ACTIVE (or it is printed here); nothing was sampled
        if not (stack.status() or {}).get("refusal_logged"):
            report.log(f"NOT ACTIVE: {e}")
        rc = EXIT_NOT_ACTIVE
    except SystemExit as e:
        rc = int(e.code) if isinstance(e.code, int) else EXIT_FAIL
    summ = outputs.summarize(out_dir, tasks, (res or {}).get("seeds") or list(st.seeds), n_sample, existed=existed)   # the seeds the route ran (given or derived); given ones when the run stopped before deriving
    census = stack.exit_census(produced=summ["produced"])                       # the ONE exit census (stack.py): planned levers whose run-time evidence disagrees with the plan, the hook never reached, designs written while no sampling call went through the levers
    timings = _timings(summ, time.perf_counter() - t0, (res or {}).get("load_s"))
    v = _conclude(mode, rc, summ, census["problems"], timings, out_dir)
    if not summ["nothing_ran"]:                                                   # a job upstream skipped whole leaves the earlier run's manifest as written (no mode claim over outputs this process did not produce)
        manifest.write(out_dir, stack.status(), mode=mode, command="design", argv=sys.argv[1:], exit_code=v["exit_code"], options=st.as_dict(), outputs=summ, timings=timings, stamps=stamps_rec,
                       extra={"preflight": pre, "upstream_argv": argv, "run": res, "det": int(args.det), "package_census": census["package_census"],
                              "package_gate": census["problems"], "precision": census["precision"], "incomplete": v["incomplete"]})
    return v["exit_code"]


def _conclude(mode: str, rc: int, summ: dict, problems, timings: dict, out_dir: str, notes=None) -> dict:
    """The design verb's exit rule and its lines — the shared core's rule (`opt_core.report.verdict`): the run's own non-zero code stands;
    outputs short of tasks x seeds x N_sample turn a 0 into 1 (the INCOMPLETE line); a problem sentence of the exit census (`stack.exit_census`:
    a planned lever that did not do its work, the hook never reached) turns a 0 into 3, refused by name. The census sentences print on every
    outcome, headed NOT ACTIVE when they decide the code — never silent; then the run line: DONE / INCOMPLETE with `skipped_existing=`, or NOTHING
    RAN when upstream skipped every (task, seed) pair of the job as already dumped (report.nothing_ran_line: no mode claim, no manifest; ``notes``
    name what such an invocation still wrote into the directory). Returns the verdict record (`exit_code`, `incomplete`, …); `exit_code` is what
    the verb returns and what `opt_manifest.json` records."""
    incomplete = None if summ.get("complete") else f"designs {summ.get('n_designs')}/{summ.get('expected')}"
    v = _core_report.verdict(rc, {"partial": list(problems)}, allow_partial=False, incomplete=incomplete)
    if problems:
        sentences = "; ".join(str(p) for p in problems)
        if rc == EXIT_OK and v["exit_code"] == EXIT_NOT_ACTIVE:                     # the census decides the code
            report.log(f"NOT ACTIVE: lever run-time census: {sentences} — exit {EXIT_NOT_ACTIVE}")
        else:                                                                       # recorded, not deciding: the run's own code or the INCOMPLETE count stands
            report.log(f"lever run-time census (recorded; exit {v['exit_code']} decided by {'the INCOMPLETE count' if rc == EXIT_OK else 'the run'}): {sentences}")
    report.log_done(dict(timings, mode=mode, n_designs=summ.get("n_designs"), n_tasks=summ.get("n_tasks"), n_seeds=summ.get("n_seeds"), out_dir=out_dir, scope=summ.get("scope"),
                         expected=summ.get("expected"), skipped_existing=summ.get("skipped_existing"), n_pairs=summ.get("n_pairs"), nothing_ran=summ.get("nothing_ran"),
                         complete=summ.get("complete"), nothing_ran_notes=list(notes or [])))
    return v


TIMINGS_NOTE = ("s/design and s/task are the amortised per-design cost of the sampling phase alone: wall_s minus load_s, the one-time cost "
                "(imports, CCD, checkpoint, fast-LN JIT when the cache is cold: run warm first), stated beside them; a route whose load "
                "time is not separable (the console script in a child process) states no per-design figure")


def _timings(summ: dict, wall: float, load_s) -> dict:
    """The run's speed statement: the steady-state per-design cost with the one-time cost named separately, or no per-design figure."""
    n_designs, n_tasks = summ.get("n_designs") or 0, summ.get("n_tasks") or 0
    known = load_s is not None
    sampling = (wall - load_s) if known else None
    per_design = round(sampling / n_designs, 3) if (known and n_designs) else None
    per_task = round(sampling / (n_tasks * max(1, summ.get("n_seeds") or 1)), 3) if (known and n_tasks and n_designs) else None
    return {"wall_s": round(wall, 2), "load_s": (round(load_s, 2) if known else None), "sampling_s": (round(sampling, 2) if known else None),
            "s_per_design": per_design, "s_per_task": per_task, "note": TIMINGS_NOTE}


# ------------------------------------------------------------------------------------------------------------------- check

def cmd_check(args) -> int:
    from . import stack
    from .modes import lever_table
    mode = _resolve_mode(args)
    rep = stack.activate(mode, dry_run=True)
    pins_rep = stack.pins_report()                                                # the stock pins: reported by this verb, never gated (design runs whatever checkout upstream itself would run)
    report.log(stack.pins_line(pins_rep))
    if args.json:
        out = dict(rep, pins_check=pins_rep, lever_table=lever_table())
        print(json.dumps(out, indent=1, default=str))
    return EXIT_NOT_ACTIVE if rep.get("would_refuse") else EXIT_OK                  # the mode would refuse on this box: the refusal's own code, as `design` would exit


# -------------------------------------------------------------------------------------------------------------------- warm

def cmd_warm(args) -> int:
    from . import warm
    mode = _resolve_mode(args)
    res = warm.run(mode, out_dir=args.out_dir, timeout=args.timeout, n_sample=args.N_sample, echo=not args.no_echo, keep=bool(args.out_dir))
    print(warm.summary_line(res), file=sys.stderr)
    if res["status"] == "PASS":
        return EXIT_OK
    return EXIT_NOT_ACTIVE if (res.get("exit_code") == EXIT_NOT_ACTIVE and not res.get("killed")) else EXIT_FAIL   # design's NOT ACTIVE is warm's: the verbs agree on the code


def main(argv=None) -> int:
    ap = build_parser()
    args, rest = ap.parse_known_args(argv)
    if not args.command:
        print(USAGE, file=sys.stderr)
        return EXIT_USAGE
    if rest and args.command != "design":
        ap.error(f"unrecognized arguments: {' '.join(rest)}")                   # exit 2 in argparse's own words: only the design verb forwards what it does not name
    if args.command == "design":
        args.extra = [x for x in rest if x != "--"]                              # further pxdesign infer arguments, in the caller's order (a `--` separator is dropped)
    return {"design": cmd_design, "check": cmd_check, "warm": cmd_warm}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
