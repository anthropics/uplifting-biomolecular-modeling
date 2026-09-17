
"""python -m mosaic_opt {design,check,warm} [--mode exact|off] ...

A thin command layer over the kit code. It never re-implements a lever or a mode:

* ``design`` — one binder-hallucination design (the notebook recipe on the kit's public target) at `--seed`, for a shape
               (`--binder-length`, `--target-copies`), through the kit driver in a fresh process. ``--mode exact``: the warm row
               (P1 load + P2 + P3; the shape must be warm: `warm`). ``--mode off``: the stock row (A_stock1, or D_stock2 on
               frozen features) through the stock caller (stock_design.py) in a clean subprocess (every kit and JAX-cache variable stripped and
               proved absent; upstream's `MOSAIC_CACHE_DIR` is kept). Both write the driver's own file set under `<out>/<tag>/`.
* ``check``  — the activation line for a mode without applying anything: the row, the resolved levers, the shape's features
               and P1 state, the stack pins, the GPU; ``--json`` prints the full report.
* ``warm``   — a shape's weights staged, features frozen and P1 files populated from an empty directory (warm.py).

Exit codes: 0 ok | 1 failed (design/warm failed, the driver's own exit status in the manifest as ``driver_rc``; the driver's outputs
short of the design: ``incomplete``) | 2 usage | 3 not active or partial — a mode not active on this box (reason printed), or a lever of
the row the driver's manifest does not show applied (``--allow-partial`` records and proceeds).
The partial state is the manifest's ``partial`` list and the ``partial=`` field of the ACTIVE / DONE lines. The in-process route
(``MOSAIC_OPT=exact`` in the environment, ``enable()``) records the same state on its ACTIVE line and cannot set the host process's exit —
``design`` is the gated form.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from . import __version__, det, driver, inputs, outputs, report, settings, stack, warm
from .modes import DEFAULT_MODE, MODES, describe_line, resolve, unknown_mode_message

PROG = "python -m mosaic_opt"
from .report import EXIT_FAIL, EXIT_NOT_ACTIVE, EXIT_OK, EXIT_USAGE                        # the exit codes live in report.py (the library layer)
from .report import allow_partial, exit_for                                                   # the exit rule lives in report.py (the formatter's home); cli re-exports both

USAGE = f"""usage: {PROG} <command> [--mode {'|'.join(MODES)}] ...

commands
  design  [--mode M] --out DIR [--seed S] [--binder-length L] [--target-copies N] [--tag T] [--steps1 N --steps2 N]
          [--target-fasta F [--first-record] [--msa A]] [--epitope 12,15,40-48] [--det 0|1] [--allow-partial]
                                       one design through the kit driver: mode exact = the warm row (P1 load + P2 + P3;
                                       the shape's frozen features and autotune file under $MOSAIC_OPT_CACHE_ROOT/<shape>/ — `warm` first);
                                       mode off = the stock caller in a clean subprocess (row A: in-process featurization);
                                       writes <out>/<tag>/results.json + pssm_seed<S>.npz + refold_seed<S>.npz (the driver's own);
                                       inputs (every mode, off included): --target-fasta = ONE record (a multi-record file is refused,
                                       exit 2, unless --first-record takes record 1: an INPUT line names the count), --epitope = 1-based
                                       target residue positions the contact loss is restricted to (every copy; out of range: exit 2)
  check   [--mode M] [--binder-length L] [--target-copies N] [--target-fasta F [--first-record] [--msa A]] [--epitope …] [--allow-partial] [--json]
                                       dry run: the row, the levers, the shape's cache state, the pins and the GPU; applies nothing
                                       (exit 3 when the plan would refuse, or is partial without --allow-partial)
  warm    [--mode exact] [--binder-length L] [--target-copies N] [--target-fasta F [--first-record] [--msa A]] [--epitope …] [--seed S] [--out DIR] [--det 0|1] [--allow-partial]
                                       stage the weights, freeze the shape's features (stock arm A), populate its P1 files (row B) from an
                                       empty directory; a warm shape is refused (exit 2): remove its directory to warm it again

modes: {' | '.join(MODES)} (default: {DEFAULT_MODE}; or MOSAIC_OPT in the environment); exit 3 when a mode is not active on this box, or when
the activation is partial (a lever of the row the driver's manifest does not show applied) and --allow-partial
is not given — the partial state is recorded either way (the ACTIVE / DONE lines' `partial=`)
environment (README "Variables"): MOSAIC_OPT_CACHE_ROOT (the P1 files' root), MOSAIC_CACHE_DIR (the weights cache), MODEL_OPT (this tree)
"""


class CliError(Exception):
    def __init__(self, msg, code=EXIT_USAGE):
        super().__init__(msg)
        self.code = code


COLD_SHAPE_REASON = "shape not warm: run.sh warm"                                # why P1 / P3 of the pinned row step aside on a shape `warm` has not filled (the ACTIVE line's p1=none(<reason>), -aside[...])
P1_MODE = next(m for m in MODES if stack.mode_needs(m)["populate"])    # the mode `warm` populates by default: the one whose P1 is pinned — a populate row and the frozen features (exact); a transparent row fills its own cache and has nothing to warm


def _mode_arg(value):
    m = (value or os.environ.get(stack.ENV_MODE) or DEFAULT_MODE).strip().lower()
    if m not in MODES:
        raise CliError(unknown_mode_message(m))                   # a tier word with no lever wired is named as such (exit 2)
    return m


def _add_common(ap):
    ap.add_argument("--mode", default=None, help=f"{' | '.join(MODES)} (default {DEFAULT_MODE}, or MOSAIC_OPT)")
    ap.add_argument("--binder-length", type=int, default=None)
    ap.add_argument("--target-copies", type=int, default=None)
    ap.add_argument("--target-fasta", default=None, help="the target = the ONE record of this FASTA (default: the kit's public example target); a multi-record file is refused by name (exit 2) unless --first-record; the driver's --target-fasta")
    ap.add_argument("--first-record", action="store_true", help="design against record 1 of a multi-record --target-fasta: the record count and the record used are printed (INPUT target_fasta records=<n> used=1) and recorded; the driver's --first-record")
    ap.add_argument("--msa", default=None, help="the target chain's precomputed alignment (.a3m | .csv) applied to every copy — upstream's use_msa=True with the fetch done ahead of time; the driver's --msa")
    ap.add_argument("--epitope", default=None, help="target residues the binder should contact: 1-based positions along the target sequence as given, comma-separated with ranges (12,15,40-48), applied on every --target-copies copy; the loss's BinderTargetContact term becomes upstream's BinderTargetContact(epitope_idx=...) in EVERY mode, off included (an input option, not a lever); a position outside the target is refused by name (exit 2); the driver's --epitope")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog=PROG, add_help=False, usage=USAGE)
    ap.add_argument("-h", "--help", action="store_true")
    sub = ap.add_subparsers(dest="command")
    d = sub.add_parser("design", add_help=False)
    _add_common(d)
    d.add_argument("--out", required=True)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--tag", default=None)
    d.add_argument("--steps1", type=int, default=None)
    d.add_argument("--steps2", type=int, default=None)
    d.add_argument("--det", default=str(det.DEFAULT_LEVEL))
    d.add_argument("--allow-partial", action="store_true", help=f"a partial activation is recorded and the exit is the run's own instead of {EXIT_NOT_ACTIVE}")
    c = sub.add_parser("check", add_help=False)
    _add_common(c)
    c.add_argument("--allow-partial", action="store_true", help="a partial plan is recorded and passes")
    c.add_argument("--json", action="store_true")
    w = sub.add_parser("warm", add_help=False)
    _add_common(w)
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--out", default=None)
    w.add_argument("--det", default=str(det.DEFAULT_LEVEL))
    w.add_argument("--allow-partial", action="store_true", help="as design --allow-partial, on the populate row")
    try:
        a = ap.parse_args(argv)
    except SystemExit:
        raise CliError(USAGE)
    if a.help or not a.command:
        raise CliError(USAGE, EXIT_OK if a.help else EXIT_USAGE)
    return a


def _shape(a):
    try:
        return inputs.shape_from_args(a.binder_length, a.target_copies, settings.driver_defaults(stack.kit_home()), target_fasta=a.target_fasta, msa=a.msa, kit_home=stack.kit_home(),
                                      first_record=a.first_record, epitope=a.epitope)
    except (ValueError, FileNotFoundError) as e:                    # ValueError: the recipe module's refusals by name (fasta_no_record, fasta_multi_record, msa_not_found, epitope_out_of_range, …; inputs.shape_from_args) — usage, exit 2, before any model work; any other error is a failure and propagates
        raise CliError(str(e))


def _tag(a, mode, shape):
    return a.tag or f"{mode}_{shape.key}_s{a.seed}"


# ----------------------------------------------------------------------------------------------------------------- design
def cmd_design(a) -> int:
    mode = _mode_arg(a.mode)
    lv = det.level(a.det)
    shape = _shape(a)
    kit = stack.kit_home()
    eff = settings.effective(kit, a.steps1, a.steps2)
    out_dir = os.path.abspath(a.out)
    os.makedirs(out_dir, exist_ok=True)
    tag = _tag(a, mode, shape)
    why = det.precision_refusal()
    if why:
        raise CliError(why, EXIT_NOT_ACTIVE)
    p = stack.pins()
    ok, pin_detail, why = stack.pins_gate(p, force=bool(os.environ.get(stack.ENV_FORCE)))         # the commit-level pin check holds for the stock arm too
    if not ok:
        raise CliError(why, EXIT_NOT_ACTIVE)
    why, notes = stack.data_path_gate()
    if why:
        raise CliError(why, EXIT_USAGE)
    if mode != "off":                                                                      # the kit route: the mode's row around one fresh driver process
        try:
            off = stack.levers_off()                                                        # the ablation switch (MODEL_OPT_LEVERS_OFF): an unknown id is NOT ACTIVE by name (exit 3), never ignored, never a traceback
        except stack.ActivationError as e:
            raise CliError(str(e), EXIT_NOT_ACTIVE) from None
        needs = stack.mode_needs(mode, off)
        rep = stack.activate(mode, dry_run=True, shape=shape)
        if rep.get("would_refuse"):
            rep["reason"] = rep["would_refuse"]; rep["dry_run"] = False
            report.log_activation(rep)
            return EXIT_NOT_ACTIVE
        root = stack.cache_root()
        pinned = needs["p1_form"] == "pinned"
        aside = stack.P1_ASIDE_REASON if (needs["p1"] and not pinned and not root) else None   # a transparent P1 with no cache root steps aside by name; the row runs without it
        feats = inputs.features_path(root, shape) if needs["features"] else None            # the frozen features `warm` wrote for the shape (P3)
        fs = inputs.features_state(feats)
        d = None
        if needs["p1"] and not aside:
            d = inputs.p1_dir(root, shape) if pinned else inputs.p1_dir(root, shape, mode)  # a transparent row's own directory per shape (xla_cache_<mode>), created here: the first process fills it
            if not pinned:
                aside = stack.transparent_dir_unusable(d)                                # a root this process cannot create the directory under (read-only, not permitted): P1 steps aside by name, the row runs
                d = None if aside else d
        st = stack.p1_state(d, needs["p1_form"] or "pinned")
        cold = {}
        if (needs["features"] and not fs["present"]) or (pinned and not st["autotune_present"]):
            # a cold shape (no frozen features / no P1 files under the cache root) is a cache miss, named — never a refusal and never a warm-up
            # inside this command: P1 and P3 step aside BY NAME and the design compiles and featurizes exactly as stock does (P2 stays);
            # `run.sh warm` fills the shape ahead of time, after which every process of the shape runs the row in its load form
            why = COLD_SHAPE_REASON
            cold = {l: why for l in ("P1", "P3") if l in stack.KIT_MODES[mode]["levers"]}
            report.log(f"{report.PREFIX} shape {shape.key} not warm under {root} (features {'present' if fs['present'] else 'absent'}, autotune file "
                       f"{'present' if st['autotune_present'] else 'absent'}): {','.join(cold)} step aside by name — this design compiles and featurizes as stock does; "
                       f"`run.sh warm --mode {mode}` with this shape's flags fills it ahead of time")
            d = None; feats = None
            st = stack.p1_state(None, "pinned"); fs = inputs.features_state(None) if "P3" in cold else fs
        res = resolve(mode, cache_dir=st["cache_dir"] if needs["p1"] else None, features=feats if needs["features"] else None,
                      features_sha=fs["sha256"] if needs["features"] else None, phase="warm", levers_off=off, p1_aside=aside, aside=cold)   # a row that names a placeholder its mode does not need is refused by name (modes.substitute), never fed a literal
        argv, env, cnotes = driver.compose(res, seed=a.seed, out_dir=out_dir, tag=tag, shape_flags=shape.flags(), settings_flags=settings.flags(eff), det_level=lv)
        aside = aside or cold.get("P1")
        p1_rep = (dict(st, autotune="load") if pinned else dict(st)) if (needs["p1"] and not aside) else ({"aside": aside} if aside else {})
        act = dict(rep, active=True, dry_run=False, route="driver", row=res.row, row_line=describe_line(res), levers_applied=list(res.levers), levers_unavailable=[],
                   partial=False, p1=p1_rep, features=fs if needs["features"] else None, env=res.env, flags=res.flags,
                   levers_off=list(res.levers_off), levers_aside=dict(res.aside),
                   notes=list(rep.get("notes") or []) + [n for n in res.notes if n not in (rep.get("notes") or [])])
        report.log_activation(act)
        if cnotes["dropped"]:
            report.log(f"{report.PREFIX} dropped from the driver's environment (the row decides): {','.join(cnotes['dropped'])}")
        rc, killed = driver.launch(argv, env, cwd=cnotes["cwd"], timeout=None, on_line=lambda s: report.log(report.relay(s)))
        results = outputs.read_results(out_dir, tag)
        run = outputs.run_summary(results)
        shown = outputs.classify(results)
        missing = [k for k in res.levers if not shown.get(k)] if rc == 0 and not killed and results else []   # the driver's manifest is the evidence of a completed run
        incomplete = outputs.incomplete(rc, killed, results)                                                    # the driver returned 0 without its results file
        act.update(levers_fallback=missing, partial=bool(missing), levers_record=outputs.levers_record(results))
        allowed = allow_partial(a.allow_partial)
        exit_code = exit_for(rc, killed or bool(incomplete), act, allowed, what=f"see {outputs.results_path(out_dir, tag)}")
        report.register_exit_tally(lambda: report.tally_line(mode, "driver", st["cache_dir"], autotune=st["autotune"], why_none=aside))
        report.log(report.done_line(mode, run, out_dir, exit_code, act, incomplete))
        if incomplete:
            report.log(f"{report.PREFIX} design FAILED: incomplete — the driver exited 0 without {outputs.results_path(out_dir, tag)}")
        return exit_code
    # mode off: the stock caller in a clean subprocess — the stock row A_stock1 (in-process featurization), resolved like every row
    res = resolve("off")
    argv, env, cnotes = driver.compose(res, seed=a.seed, out_dir=out_dir, tag=tag, shape_flags=shape.flags(), settings_flags=settings.flags(eff), det_level=lv)
    cmd = [argv[0], "-s", "-m", "mosaic_opt.stock_design", "--driver", argv[1], "--pins", stack.pins_path(), "--"] + argv[2:]
    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))                 # opt/: the package itself, so the child finds mosaic_opt.stock_design
    import opt_core                                                                         # and the core this process imports (the package's modules import it)
    core_parent = os.path.dirname(os.path.dirname(os.path.abspath(opt_core.__file__)))
    env["PYTHONPATH"] = os.pathsep.join([pkg_parent, core_parent] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    fc = stack.installed_fast_check(kit)                                                    # the observers the stock arm imports are the kit's own files
    if not fc["clean"]:
        raise CliError(stack.installed_fast_refusal(fc), EXIT_NOT_ACTIVE)
    act = {"active": False, "mode": "off", "route": "driver", "row": res.row, "row_line": describe_line(res), "levers_applied": [], "levers_unavailable": [], "partial": False,
           "p1": {}, "gpu": stack.gpu_identity(), "upstream": stack.upstream_versions(), "package_version": __version__, "pins": pin_detail, "kit_install": fc,
           "reason": "mode off: stock mosaic (the kit driver with every lever off in a clean subprocess; no environment set, no lever applied)", "notes": notes,
           "forced": stack.forced_word(pin_detail) if os.environ.get(stack.ENV_FORCE) else None}   # MOSAIC_OPT_FORCE=1 with upstream off its pin: `forced=upstream_pins(…)` on DONE, not stock's libraries
    report.log_activation(act)
    report.log(f"{report.PREFIX} stock subprocess: {' '.join(cmd[:5])} ... (env stripped of {','.join(stack.stock_env_absent(p))}; dropped {','.join(cnotes['dropped']) or 'nothing'})")
    rc, killed = driver.launch(cmd, env, cwd=cnotes["cwd"], timeout=None, on_line=lambda s: report.log(report.relay(s)))
    results = outputs.read_results(out_dir, tag)
    run = outputs.run_summary(results)
    shown = outputs.classify(results)
    on = [k for k, v in shown.items() if v]
    if on:
        report.log(f"{report.PREFIX} WARNING: the driver's manifest shows {','.join(on)} applied on the stock arm (see {outputs.results_path(out_dir, tag)})")
    incomplete = outputs.incomplete(rc, killed, results)
    exit_code = EXIT_FAIL if (killed or rc != 0 or incomplete) else EXIT_OK                     # the stock arm has no partial state: failed or ok
    report.register_exit_tally(lambda: report.tally_line("off", "driver", None))
    report.log(report.done_line("off", run, out_dir, exit_code, act, incomplete))            # act: partial=False (the stock arm has no partial state), forced= when the pins were overridden
    if incomplete:
        report.log(f"{report.PREFIX} design FAILED: incomplete — the driver exited 0 without {outputs.results_path(out_dir, tag)}")
    return exit_code


# ----------------------------------------------------------------------------------------------------------------- check
def cmd_check(a) -> int:
    mode = _mode_arg(a.mode)
    shape = _shape(a)
    rep = stack.activate(mode, dry_run=True, shape=shape)
    allowed = allow_partial(a.allow_partial)
    if rep.get("partial"):
        rep["allow_partial"] = allowed
    if a.json:
        print(json.dumps(rep, indent=1, default=str))
    report.log_activation(rep)
    if rep.get("would_refuse"):
        return EXIT_NOT_ACTIVE
    return exit_for(0, False, rep, allowed, what="check")                # a partial plan: the one verdict, exit 3 unless allowed


# ----------------------------------------------------------------------------------------------------------------- warm
def cmd_warm(a) -> int:
    mode = _mode_arg(a.mode or os.environ.get(stack.ENV_MODE) or P1_MODE)              # warm's default is the mode whose row carries the P1 files, whatever the package default
    lv = det.level(a.det)
    shape = _shape(a)
    why = det.precision_refusal()
    if why:
        raise CliError(why, EXIT_NOT_ACTIVE)
    rep = stack.activate(mode, dry_run=True, shape=shape)
    would = [w for w in (rep.get("would_refuse_all") or []) if "not warm" not in w and stack.WEIGHTS_MISSING_MARKER not in w and stack.P1_ASIDE_REASON not in w]   # a transparent row without a root is warm's own refusal (nothing to warm), named below
    if would:
        rep["reason"] = would[0]; rep["dry_run"] = False
        report.log_activation(rep)
        return EXIT_NOT_ACTIVE
    try:
        res = warm.warm(mode, shape, seed=a.seed, out_dir=a.out, det_level=lv, log=report.log, allow_partial=allow_partial(a.allow_partial))
    except warm.AlreadyWarm as e:
        raise CliError(str(e), EXIT_USAGE)
    except (ValueError, stack.ActivationError) as e:
        raise CliError(str(e), EXIT_NOT_ACTIVE)
    report.log(warm.summary_line(res))
    return res.get("exit", EXIT_FAIL)                                     # design's rule on the populate row: 0 ok, 1 failed, 3 partial unless --allow-partial


COMMANDS = {"design": cmd_design, "check": cmd_check, "warm": cmd_warm}


def main(argv=None) -> int:
    try:
        a = parse_args(argv)
        return COMMANDS[a.command](a)
    except CliError as e:
        stream = sys.stdout if e.code == EXIT_OK else sys.stderr
        print(str(e) if e.code in (EXIT_OK, EXIT_USAGE) and str(e).startswith("usage:") else f"{report.PREFIX} {'NOT ACTIVE: ' if e.code == EXIT_NOT_ACTIVE else ''}{e}", file=stream, flush=True)
        return e.code
    except FileNotFoundError as e:
        print(f"{report.PREFIX} {e}", file=sys.stderr, flush=True)
        return EXIT_NOT_ACTIVE

