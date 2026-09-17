"""``boltz2-opt`` (== ``python -m boltz2_opt``): pred | check | warm.

    boltz2-opt pred  --mode exact|fast|big [--n_gpu P] --input <yaml|fasta|dir> [...] --out_dir OUT [--seed S | --seeds 0,1] [SETTINGS] [--allow-partial]
                                                                                                    the kit route, the persistent worker (worker.py)
    boltz2-opt pred  --mode off --input <yaml|fasta|dir> --out_dir OUT [--seed S] [SETTINGS] [--num_workers N] [--no_kernels] [--det 1] [-- <any other boltz predict args>]
                                                                                                    the stock CLI in a clean, proven subprocess (stock_pred.py)
        SETTINGS = `boltz predict`'s own options by name, upstream's semantics and defaults, the same on every mode (settings.KNOBS, defaults in
        stock/PINS.json cli_defaults = boltz/main.py): [--recycling_steps 3] [--sampling_steps 200] [--diffusion_samples 1] [--max_parallel_samples 5]
        [--step_scale 1.5] [--write_full_pae] [--write_full_pde] [--output_format mmcif|pdb] [--override]; an option not given is not passed (stock)
        / upstream's default (worker). The worker route refuses by name the one it cannot serve: --no_kernels (settings.WORKER_REFUSED).
        `--det 1` is the deterministic recipe of the stock route (det.py: --num_workers 1 --no_kernels added when absent; the kernels-off stock anchor).

    boltz2-opt check --mode M [--json] [--allow-partial]                                            dry run: resolve, gate and plan the partials on this box, apply nothing
    boltz2-opt warm  --mode M --out OUT [--allow-partial]                                          four predictions of the carried public inputs, one worker process, through pred (warm.py)

`--mode` defaults to $BOLTZ2_OPT, then to modes.DEFAULT_MODE (fast); BOLTZ2_OPT unset applies nothing; a `--mode` that disagrees with a set BOLTZ2_OPT is refused (exit 2).
`--det` and `-- <stock args>` belong to `--mode off` and are refused on the worker route (exit 2), never ignored; `--seeds` (several seeds in
one pass) belongs to the worker route. The exit rule on partial activation: a lever of the row that the kit ran without its
optimization (stack.partial_activation after a run) prints
`[boltz2-opt] NOT ACTIVE: partial activation — <detail>; exit 3 (--allow-partial records and proceeds)` and exits 3; `--allow-partial` (pred, warm,
check) names the opt-out on the line `[boltz2-opt] PARTIAL allowed:
<detail> (--allow-partial, recorded)` and exits by the run's own outputs; `--mode off` runs no lever and refuses the flag (exit 2).
Exit codes: 0 ok · 1 a prediction failed, or outputs short (incomplete; an input the stock parser skipped is named `SKIPPED item=<name>` and
counted here; a batch boltz's own predict_step skipped on CUDA out of memory is named on its unit's `FAILED item=<name> seed=<s>` line) · 2 usage · 3 not active (refused by name, pins, kit files,
no GPU, stock proof failed, evidence missing — a pass without its KERNELS census included) or partial activation without --allow-partial ·
5 the KERNELS REQUIRE guard refused: an accelerator the route expects engaged is absent or fell back (kernels.py; both routes).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import List, Optional

from opt_core import stock_proof

from . import det, kernels, manifest as mf, modes, report as rep, settings, skipped, stack, tp


def _mode_arg(p):
    p.add_argument("--mode", default=None, help=f"{'|'.join(modes.MODE_NAMES)} (default: $BOLTZ2_OPT, else {modes.DEFAULT_MODE})")
    p.add_argument("--n_gpu", type=int, default=1, metavar="P",
                   help=f"GPUs for one prediction (default 1: the single-GPU line; absent == 1). P > 1 = the row-sharded pair stack, --mode big only "
                        f"(P in {{{','.join(map(str, tp.SUPPORTED_P))}}}); refused by name under any other mode, with fewer than P visible GPUs, or another P")
    p.add_argument("--no-compile", dest="no_compile", action="store_true",
                   help=f"= {modes.LEVERS_OFF_ENV}={modes.COMPILE_WORD} (one mechanism): no torch.compile lever engages — accepted on every mode; this kit adds none and "
                        f"stock boltz predict compiles nothing, so the ACTIVE / DRY-RUN lines print compile=off:none_in_kit (off:user with the flag)")


def resolve_mode(arg: Optional[str]) -> str:
    bad = sorted(k for k in os.environ if (k.startswith("BOLTZ2_OPT") and k != "BOLTZ2_OPT") or k.startswith(BIG_WORD))   # _autoload.undeclared_names' rule (that module is the .pth hook; not imported here)
    if bad:
        raise SystemExit(_usage(f"undeclared {', '.join(bad)} in the environment: the one variable is BOLTZ2_OPT (a mistyped name is never ignored; "
                                f"the memory mode's levers and settings are the mode's row, no {BIG_WORD}* word selects them)"))
    env = modes.env_mode(); m = (arg or env or modes.DEFAULT_MODE).strip().lower()
    if env and arg and arg.strip().lower() != env:
        raise SystemExit(_usage(f"--mode {arg} disagrees with BOLTZ2_OPT={env}; set one"))
    modes.resolve(m)
    return m


def _usage(msg: str) -> int:
    print(f"{rep.PREFIX} usage: {msg}", file=sys.stderr)
    return rep.EXIT_USAGE


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="boltz2-opt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pred", help="predict: --mode exact|fast|big on the kit worker, --mode off on the stock CLI")
    _mode_arg(p)
    p.add_argument("--input", nargs="+", required=True); p.add_argument("--out_dir", required=True)
    p.add_argument("--seeds", default=None, help="worker route: comma-separated seeds, one pass (default 0)")
    defaults = settings.stock_defaults()
    for knob, kind in settings.KNOBS:                                   # `boltz predict`'s own options, upstream's names / semantics / defaults, the same on every mode (not given = not passed / upstream's default)
        f_, h = settings.flag(knob), f"boltz predict's {settings.flag(knob)} (upstream default {defaults.get(knob)!r}; stock/PINS.json cli_defaults)"
        if kind == "flag":
            p.add_argument(f_, dest=knob, action="store_true", default=False, help=h)
        elif isinstance(kind, tuple):
            p.add_argument(f_, dest=knob, choices=list(kind), default=None, help=h)
        else:
            p.add_argument(f_, dest=knob, type=(str if kind in (str, "path") else kind), default=None, help=h + (" — worker route: the one seed of the pass, = --seeds S" if knob == "seed" else ""))
    p.add_argument("--det", type=int, default=None, help="stock route: the deterministic recipe (1 = --num_workers 1 --no_kernels added when absent; det.py)")
    p.add_argument("--allow-partial", action="store_true", help="worker routes: accept a kit fallback of a row lever (named on the PARTIAL line; without it a fallback exits 3)")
    p.add_argument("stock_args", nargs=argparse.REMAINDER, help="stock route: extra `boltz predict` arguments after --")
    p = sub.add_parser("check", help="dry run: resolve, gate and plan the partials of the mode on this box; apply nothing")
    _mode_arg(p); p.add_argument("--json", action="store_true", help="print the plan as JSON too")
    p.add_argument("--allow-partial", action="store_true", help="a planned kit fallback of a row lever exits 0 (reported); without it 3")
    p = sub.add_parser("warm", help="four predictions of the carried public inputs (inputs/1BRS_x2_barnase_barstar.yaml 398 tokens, 1BRS_a4t2_barnase_trpcage.yaml 480, 1BRS_a2b4_barnase_barstar.yaml 576, 1BRS_a6b4t2_barnase_barstar_trpcage.yaml 1056: one per token-count specialization class and kernel window) in one worker process through pred on the mode's route")
    _mode_arg(p); p.add_argument("--out", required=True)
    p.add_argument("--allow-partial", action="store_true", help="accept a kit fallback of a row lever (recorded); without it a fallback exits 3")
    return ap


# ---------------------------------------------------------------- stock route ----------------------------------------------------------------
BIG_WORD = "BOLTZ2_BIG_"                # a caller word under this stem (a per-lever switch / setting / census opt-out of the memory line) is refused by name: the line is the mode's row (big.py)
KERNELS_CENSUS = os.path.join("_kit", "kernels_census.json")   # the KERNELS reader's record of a stock pass, under the pass's kit directory (kernels.emit; cmd_pred_off reads it back for the exit rule and removes it)


def stock_command(yaml_path: str, out_dir: str, flags: Optional[dict] = None, extra: Optional[List[str]] = None, order: Optional[List[str]] = None,
                  det_level: Optional[int] = None) -> tuple:
    """(argv, env) for the stock caller: `python -s -m boltz2_opt.stock_pred ... -- predict <yaml> --out_dir <out> [the caller's `boltz predict`
    options, as given (settings.stock_argv: --recycling_steps … --seed … --no_kernels, in the caller's order)] [the deterministic recipe's switches
    when --det 1 (det.stock_args: --num_workers 1 --no_kernels, those absent)] [extra: the arguments after --]` — with nothing given this is
    upstream's CLI exactly as shipped: the weights and CCD come from $BOLTZ_CACHE (upstream's own --cache default,
    main.py:826; the checkpoint <cache>/boltz2_conf.ckpt is its own resolution, main.py:1293-1295; --model defaults to boltz2, main.py:974).
    The environment: every kit switch family stripped, every PYTHONPATH entry inside a kit directory removed."""
    pins = stack.load_pins(); se = pins["stock_environment"]
    args = ["predict", os.path.abspath(yaml_path), "--out_dir", os.path.abspath(out_dir)]
    extra_args = [a for a in (extra or []) if a != "--"]
    args += settings.stock_argv(flags, order)
    args += det.stock_args(det_level, args + extra_args)               # --det 1: the deterministic recipe's switches not already given — by name or after `--` (an explicit opt-in; nothing at --det 0 / absent)
    args += extra_args
    kit_dirs = [stack.kit_path("forward")]
    env = stack.stripped_env()
    pp, _ = stock_proof.strip_pythonpath(env.get("PYTHONPATH"), kit_dirs)   # every entry inside a kit directory removed
    if pp:
        env["PYTHONPATH"] = pp
    else:
        env.pop("PYTHONPATH", None)
    census = os.path.join(os.path.abspath(out_dir), KERNELS_CENSUS)
    argv = [sys.executable, "-s", "-m", "boltz2_opt.stock_pred", "--env-absent", ",".join(se["must_be_absent_prefixes"]),
            "--kit-modules", ",".join(se["kit_module_prefixes"]), "--kit-dirs", ",".join(kit_dirs), "--pins", stack.pins_path(),
            "--kernels-route", kernels.route_word("off"), "--kernels-settings", settings.settings_word(flags) + (f"+det{int(det_level)}" if det_level else ""),
            "--kernels-expect", kernels.format_expect(modes.kernels_expected("off", stock_args=args)), "--kernels-json", census, "--"] + args
    return argv, env


def cmd_pred_off(a) -> int:
    rep.arm_tally("off", "stock_cli")
    try:                                                           # the stock CLI is a single-GPU line: --n_gpu > 1 is refused by name (the core's words), nothing launched
        tp.check(a.n_gpu, "off")
    except tp.Refused as e:
        rep.set_rc(rep.EXIT_NOT_ACTIVE); rep.say(rep.not_active_line(str(e))); return rep.EXIT_NOT_ACTIVE
    except ValueError as e:
        rep.set_rc(rep.EXIT_USAGE); return _usage(f"--n_gpu: {e}")
    if len(a.input) != 1:
        rep.set_rc(rep.EXIT_USAGE); return _usage("--mode off takes one input path — a YAML / FASTA file or a directory of them — as boltz predict does (one process per call)")
    if a.seeds is not None:
        rep.set_rc(rep.EXIT_USAGE); return _usage("--seeds belongs to the worker route (several seeds in one pass); --mode off takes boltz predict's --seed S (one process per call)")
    flags = knob_flags(a)
    try:
        given = settings.given(flags); detl = det.describe(a.det)    # validated by name (a value below its minimum, a bad output_format, an unknown --det level)
        det.stock_args(a.det, [])
    except ValueError as e:
        rep.set_rc(rep.EXIT_USAGE); return _usage(str(e))
    frozen = stack.cache_check()                      # the stock CLI downloads whatever its cache lacks (main.py:198-256): refused by name first
    if frozen:
        rep.set_rc(rep.EXIT_NOT_ACTIVE); rep.say(rep.not_active_line("frozen weights: " + "; ".join(frozen))); return rep.EXIT_NOT_ACTIVE
    rep.say(stack.weights_line(stack.weights_status()))   # the checkpoint's digest status, named (pinned | unknown); an unknown checkpoint proceeds
    argv, env = stock_command(a.input[0], a.out_dir, flags, a.stock_args, getattr(a, "knob_order", None), a.det)
    os.makedirs(a.out_dir, exist_ok=True)
    rep.say(rep.not_active_line("mode off: stock — the stock CLI in a clean subprocess (proof: the printed `[boltz2-opt stock] proven stock` line)"))
    census_path = os.path.join(os.path.abspath(a.out_dir), KERNELS_CENSUS)
    import time as _time
    t0 = _time.time()
    r = subprocess.run(argv, env=env)
    try:                                                              # fold the pass's KERNELS record into the exit rule and the run record; it leaves with the verb (finally)
        census = kernels.read_record(census_path, since=t0)   # this pass's KERNELS record (a record written before the launch is an earlier pass's: ignored, never deleted)
        rc = r.returncode
        name = os.path.splitext(os.path.basename(os.path.normpath(a.input[0])))[0]
        acct = stock_unit_account(a.out_dir, a.input[0], a.seed, rc)     # every (record, seed) unit of the input path: ok, or failed with a named reason — a parser skip read from upstream's own processed manifest
        for u in acct["skipped"]:
            rep.say(skipped.line(u["name"], u["reason"]))
        for u in acct["failed_units"]:
            rep.say(skipped.failed_line(u["name"], u["seed"], u["reason"]))
        rep.count(ok=acct["ok"], failed=acct["failed"])
        rc = rep.kernels_exit("off", rc, [census] if census else [], where=KERNELS_CENSUS,   # the KERNELS census's exit rules (report.kernels_exit, both routes)
                              all_skipped=acct["all_skipped"])   # the input SKIPPED by the stock parser: the census's NO-STEP is 'nothing ran' — the SKIPPED / FAILED lines and exit 1 speak, not the REQUIRE guard
        if rc == 0 and acct["failed"]:
            rc = rep.EXIT_FAILED                                   # outputs short (the kit's exit 1): the stock process exited 0 but a record of its input produced no prediction
        rep.set_rc(rc)
        mf.record(mf.build("pred", "off", report={"active": False, "reason": "mode off"}, inputs=mf.input_records([{"name": name, "yaml": a.input[0]}]),
                                    settings={"given": given, "defaults": "upstream's (boltz/main.py; stock/PINS.json cli_defaults)", "extra": [x for x in a.stock_args if x != "--"]},
                                    det=detl, command=argv[argv.index("--") + 1:], rc=rc, kernels_census=census,
                                    outputs={"out_dir": os.path.abspath(a.out_dir), "predictions": _stock_outputs(a.out_dir), **{k: acct[k] for k in ("expected", "ok", "failed", "status", "failed_units", "skipped")}}))
        return rc if rc in (0, 1, 2, 3, rep.EXIT_KERNELS) else rep.EXIT_FAILED
    finally:
        try:
            os.remove(census_path)
        except OSError:
            pass
        try:
            os.rmdir(os.path.dirname(census_path))                       # the pass's kit directory, when nothing else is in it
        except OSError:
            pass


def stock_unit_account(out_dir: str, input_path: str, seed, rc: int) -> dict:
    """The stock route's completeness account of its (record, seed) units, the same shape as the worker route's (skipped.units). The records
    are boltz predict's for this input path — the file's stem, or one per file of a directory (worker.expand_inputs) — under upstream's
    ``boltz_results_<input stem>/``: a unit is ok when ``predictions/<record>/<record>_model_0.cif`` (``.pdb`` under ``--output_format pdb``)
    exists; else failed with a named reason — the process's own exit code when it failed, ``stock parser skipped the input`` when it exited 0
    and its processed manifest carries no record of that id (boltz's ``Failed to process … Skipping`` line above names the parser error),
    outputs short otherwise."""
    from .worker import expand_inputs
    root = os.path.splitext(os.path.basename(os.path.normpath(input_path)))[0]          # boltz: out_dir / f"boltz_results_{data.stem}" for the path it was given
    names = [os.path.splitext(os.path.basename(f))[0] for f in expand_inputs([input_path])] if os.path.exists(input_path) else [root]
    ids = skipped.record_ids(os.path.join(out_dir, f"boltz_results_{root}", "processed", "manifest.json"))   # None: the process never reached the parser
    seed_word = seed if seed is not None else "-"
    failed_units, skipped_units, ok = [], [], 0
    for name in names:
        have = any(os.path.isfile(os.path.join(out_dir, f"boltz_results_{root}", "predictions", name, f"{name}_model_0.{ext}")) for ext in ("cif", "pdb"))
        skip = skipped.reason(None) if (rc == 0 and not have and ids is not None and name not in ids) else None
        if rc == 0 and have:
            ok += 1; continue
        why = skip or (f"the stock process exited {rc}" if rc != 0 else f"outputs short (no boltz_results_{root}/predictions/{name}/{name}_model_0.cif|.pdb)")
        failed_units.append({"name": name, "seed": seed_word, "reason": why})
        if skip:
            skipped_units.append({"name": name, "reason": skip})
    n = ok + len(failed_units)
    return {"expected": n, "ok": ok, "failed": len(failed_units), "status": "complete" if not failed_units else "incomplete",
            "failed_units": failed_units, "skipped": skipped_units, "all_skipped": bool(names) and len(skipped_units) == len(names)}


def _stock_outputs(out_dir: str) -> dict:
    recs = {}
    for root, _d, files in os.walk(out_dir):
        if os.path.basename(os.path.dirname(root)) == "predictions":
            recs[os.path.relpath(root, out_dir)] = {f: mf.sha256_of(os.path.join(root, f)) for f in sorted(files)}
    return recs


# ---------------------------------------------------------------- check ----------------------------------------------------------------
def cmd_check(a) -> int:
    m = resolve_mode(a.mode)
    plan = stack.gate(m, need_gpu=True, n_gpu=a.n_gpu, refresh_weights=True)      # check hashes the checkpoint afresh and rewrites its digest memo entry; the run verbs take a memo hit
    modes.set_n_gpu(plan["n_gpu"] or 1); row = modes.resolve(m)                    # the xP line's by-name drops (modes.TP_DROPS) show on the DRY-RUN line as they will on the run's
    line_report = {"mode": m, "route": row["route"], "gpu": (plan["gpu"] or {}).get("name"), "n_gpu": plan["n_gpu"] or a.n_gpu,
                   "levers": list(row["levers"]) + (["rowpair_tp"] if (plan["n_gpu"] or 1) > 1 else []), "env": row["env"], "card_off": dict(row.get("card_off") or {}), "off": dict(row.get("off") or {}),
                   "tp_off": dict(row.get("tp_off") or {})}
    gpu_class = stack.gpu_class_check(plan["gpu"], os.environ.get("MODEL_OPT_TARGET_GPU"))
    if a.json:
        print(json.dumps({**plan, "target_gpu": gpu_class["target"],
                          "gpu_class": gpu_class, "gpu_supported": gpu_class["supported"]}, indent=1, default=str))
    if m == "off":
        if a.allow_partial:
            return _usage("--allow-partial names a kit fallback of a row lever; the stock route (--mode off) runs no lever")
        rep.say(rep.dry_run_line(line_report)); return rep.EXIT_OK
    if plan["reasons"]:
        rep.say(rep.dry_run_line(line_report)); rep.say(rep.not_active_line("; ".join(plan["reasons"]))); return rep.EXIT_NOT_ACTIVE
    note = stack.gpu_class_note(gpu_class)
    if note:
        rep.say(f"{rep.PREFIX} note: {note}")
    rep.say(rep.dry_run_line(line_report))
    rep.say(f"{rep.PREFIX} DRY-RUN ok: worker={stack.worker_name(row['mode'])} kit_files={len(plan['stage'])} boltz={plan['pins']['version']} cache={plan['cache']}")
    return rep.EXIT_OK


def knob_flags(a) -> dict:
    """The `boltz predict` options of a parsed `pred` command line: {knob: value or None / bool} (settings.KNOBS)."""
    return {k: getattr(a, k, None) for k in settings.KNOB_NAMES}


def knob_order(argv: List[str]) -> List[str]:
    """The knob names in the order the caller typed them (before any `--`): the stock command renders them in that order."""
    out = []
    for tok_ in argv:
        if tok_ == "--":
            break
        name = tok_[2:].split("=", 1)[0].replace("-", "_") if tok_.startswith("--") else None   # --preprocessing-threads -> preprocessing_threads
        if name in settings.KNOB_NAMES and name not in out:
            out.append(name)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    a = ap.parse_args(argv)
    a.knob_order = knob_order(argv)
    if getattr(a, "no_compile", False):
        modes.add_levers_off_word(modes.COMPILE_WORD)           # --no-compile = MODEL_OPT_LEVERS_OFF=compile: the word joins this process's environment before any row resolves or child launches
    try:
        m = resolve_mode(a.mode)
    except ValueError as e:
        return _usage(str(e))
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else rep.EXIT_USAGE
    if a.cmd == "check":
        return cmd_check(a)
    if a.cmd == "pred":
        if m == "off":
            if a.allow_partial:
                return _usage("--allow-partial names a kit fallback of a row lever; the stock route (--mode off) runs no lever")
            return cmd_pred_off(a)
        stray = ["--det"] if a.det is not None else []
        stray += ["`-- <stock args>`"] if [x for x in a.stock_args if x != "--"] else []
        if stray:
            return _usage(f"{', '.join(stray)} belong to the stock route (--mode off); the worker route (--mode {m}) takes boltz predict's --{' --'.join(k for k in settings.KNOB_NAMES if k not in settings.WORKER_REFUSED)}, "
                          f"--seeds, --allow-partial")
        if a.seed is not None and a.seeds is not None:
            return _usage("--seed S is the one seed of the pass (= --seeds S); give --seed or --seeds, not both")
        from . import worker
        seeds = [int(a.seed)] if a.seed is not None else worker.parse_seeds(a.seeds or "0")
        return worker.run(m, a.input, a.out_dir, seeds, allow_partial=a.allow_partial, n_gpu=a.n_gpu, settings=knob_flags(a))
    if a.cmd == "warm":
        from . import warm
        return warm.run(m, a.out, allow_partial=a.allow_partial, n_gpu=a.n_gpu)
    return _usage(f"unknown command {a.cmd}")
