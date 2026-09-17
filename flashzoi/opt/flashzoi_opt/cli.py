"""python -m flashzoi_opt {pred,check,warm} [--mode exact|off] ...  (exact = the default)

A thin command layer over the kit code and the upstream API. It never re-implements a component, a mode or a test:

* ``pred``   — predictions for a set of items (inputs.py: a directory of `<item>.npy` one-hots or a JSON item list) at the pinned settings
               (settings.py), written in the one output schema (outputs.py) plus ``opt_manifest.json`` (manifest.py).
               ``--mode off``: the tree's ONE stock caller ``opt/flashzoi_opt/stock_pred.py`` (standalone: nothing of this package or the kit
               imported) run in a clean subprocess (every package and kit variable stripped; the script proves its environment and
               writes the same outputs, rows and its ``opt_manifest.json`` (the environment proof inside); its lines carry its own prefix ``[flashzoi-stock]``; its exit
               code is forwarded). Kit modes: ``flashzoi_opt.enable(mode)``, the one weights loader
               (weights.py, the pins asserted), the kit's apply line with the mode's argument on each replicate as it loads (eager: stack.apply_to), then
               the same loop as the stock caller (loop.py). ``--det`` applies the deterministic recipe (det.py) on either route.
               ``--jobs <file>`` (`input<TAB>out` per line, each what --input / --out take): ONE process serves every job of the
               file — the one load + apply, then the same loop per job into the job's own output directory (multi.py); ``--input`` /
               ``--out`` / ``--items`` beside it and ``--mode off`` with it refuse by name (exit 3).
               After the run the kit's own evidence is read back (stack.settle: ``KitRunner.effective_flags()`` per replicate, stated in the
               manifest as ``partial_detection``): a lever of the mode without a true flag is ``components_fallback`` and the run is PARTIAL —
               recorded in the manifest (``partial``), the outputs kept, exit 3; ``--allow-partial`` (recorded as ``allow_partial``) proceeds to
               the run's own exit. A failed item is ``incomplete: <ok>/<items>`` in the manifest, exit 1 — never conflated with partial.
* ``check``  — the dry run: resolves and gates the mode on this machine (stack.activate(dry_run=True)) and applies nothing; the DRY-RUN
               line; ``--json`` prints the full report. Exit 3 when the mode would be refused.
* ``warm``   — imports + the kit's apply line on one replicate + one forward on the kit's shipped canary window (warm.py), to fill
               the Triton cache; the same evidence read back after the forward (WARM PARTIAL, exit 3 unless ``--allow-partial``).

Mode = ``--mode`` when given, else FLASHZOI_OPT from the environment, else the package default (exact); a ``--mode`` that disagrees
with a set FLASHZOI_OPT is refused (one mode per run). Kit modes: ``exact`` = ``KitRunner(model)`` at the kit's defaults,
numerics='tf32' (modes.MODE_ARGS).
Exit codes: 0 ok, 1 a prediction/step failed, 2 usage, 3 not active (refused) or partial (``--allow-partial`` records and proceeds).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from . import __version__, report
from .det import CUBLAS_ENV as DET_ENV
from .modes import ENV_MODE, MODES, check_mode

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_NOT_ACTIVE = 0, 1, 2, 3
COMMANDS = ("pred", "check", "warm")


class CliError(ValueError):
    """A usage error (exit 2)."""


def effective_mode(flag: str | None) -> str:
    """The mode: --mode, else $FLASHZOI_OPT, else the default; both set and different -> CliError (the flag never silently wins)."""
    env = (os.environ.get(ENV_MODE) or "").strip().lower()
    if flag is not None and env and flag.strip().lower() != env:
        raise CliError(f"--mode {flag} disagrees with {ENV_MODE}={env} in the environment; one run has one mode — drop one of them")
    try:
        return check_mode(flag)
    except ValueError as e:
        raise CliError(str(e)) from e


STOCK_SCRIPT_RELPATH = os.path.join("opt", "flashzoi_opt", "stock_pred.py")   # the tree's one stock caller: in the package's directory, run by path (nothing of the package imported)
ALLOW_PARTIAL_HELP = ("accept a PARTIAL activation (a lever of the mode without the kit's evidence after the run: components_fallback) and exit by the run's "
                      "own outcome; recorded in the manifest. Without it a partial run exits 3 (the outputs and the manifest stay)")


def stock_script() -> str:
    from . import stack
    p = os.path.join(stack.tree_home(), STOCK_SCRIPT_RELPATH)
    if not os.path.isfile(p):
        raise CliError(f"the stock caller is not in the tree: {p} (set MODEL_OPT to the flashzoi/ directory)")
    return p


def stock_env(det: bool) -> dict:
    """The clean environment of the stock subprocess: the package's and the kit's variables stripped (stack.STRIPPED_ENV; the script proves it
    before importing torch)."""
    from . import stack
    env = {k: v for k, v in os.environ.items() if k not in stack.STRIPPED_ENV}
    if not det:
        env.pop(DET_ENV, None)
    return env


def stock_command(a) -> list:
    cmd = [sys.executable, "-s", stock_script(), "--input", a.input, "--out", a.out]
    if a.det:
        cmd.append("--det")
    if getattr(a, "items", None):
        cmd += ["--items", a.items]
    if getattr(a, "tracks", None) and a.tracks != "all":
        cmd += ["--tracks", a.tracks]
    return cmd


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="flashzoi-opt", description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", action="version", version=f"flashzoi_opt {__version__}")
    sub = ap.add_subparsers(dest="command")
    p = sub.add_parser("pred", help="predictions for a set of items")
    p.add_argument("--mode", default=None, help=f"{'|'.join(MODES)} (default: $FLASHZOI_OPT, else exact)")
    p.add_argument("--input", default=None, help="a directory of <item>.npy one-hots (required without --jobs; refused with it, whose jobs file names every input directory)")
    p.add_argument("--out", default=None, help="output directory: <item>.npy, rows.jsonl, opt_manifest.json (required without --jobs; refused with it)")
    p.add_argument("--det", action="store_true", help="the deterministic recipe (det.py)")
    p.add_argument("--items", default=None, help="a comma-separated subset of item ids (stems) of --input")
    p.add_argument("--tracks", default="all", help="the tracks to return (upstream's `slices` argument): all (default) | i,lo-hi,... | @file with one entry per line; every mode")
    p.add_argument("--jobs", default=None, help="a jobs file (`input<TAB>out` per line, each what --input / --out take): ONE process serves every job — the weights load and the kit apply once; --input / --out / --items are not given with it")
    p.add_argument("--allow-partial", action="store_true", help=ALLOW_PARTIAL_HELP)
    for name in ("check", "warm"):
        q = sub.add_parser(name)
        q.add_argument("--mode", default=None)
        if name == "warm":
            q.add_argument("--allow-partial", action="store_true", help=ALLOW_PARTIAL_HELP)
        if name == "check":
            q.add_argument("--json", action="store_true")
    return ap


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = _parser()
    if not argv:
        ap.print_usage(sys.stderr); return EXIT_USAGE
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return EXIT_OK if e.code == 0 else EXIT_USAGE
    if a.command not in COMMANDS:
        ap.print_usage(sys.stderr); return EXIT_USAGE
    try:
        mode = effective_mode(a.mode)
    except CliError as e:
        sys.stderr.write(f"flashzoi-opt: {e}\n"); return EXIT_USAGE
    try:
        if a.command == "pred":
            return cmd_pred(a, mode)
        if mode == "off":
            sys.stderr.write(f"flashzoi-opt: --mode off has one command, pred (the stock arm has no check and no warm)\n"); return EXIT_USAGE
        if a.command == "check":
            return cmd_check(a, mode)
        return cmd_warm(a, mode)
    except CliError as e:
        sys.stderr.write(f"flashzoi-opt: {e}\n"); return EXIT_USAGE


def _multi_jobs(a) -> list | None:
    """The jobs of `pred --jobs <file>` (stack.read_multi_jobs, every job's items resolved before any load), else None.
    Refused by name (the NOT ACTIVE line, exit 3): `--input` / `--out` / `--items` beside `--jobs` (the jobs file names every input and
    output); an unreadable jobs file. Under `--mode off`, cmd_pred runs the stock caller once per job, in file order (one clean process
    each), and returns the worst job's exit code."""
    from . import multi, stack
    if not a.jobs:
        return None
    on_line = [f for f, v in (("--input", a.input), ("--out", a.out), ("--items", a.items)) if v]
    if on_line:
        why = f"--jobs with {', '.join(on_line)} on the line: the jobs file names every input and output — drop them"
        report.log_refusal(why)
        raise _NotActive(why)
    try:
        jobs = stack.read_multi_jobs(a.jobs)
        return multi.plan(jobs)
    except Exception as e:  # noqa: BLE001  the readers' refusals (stack.ActivationError, inputs.InputError), by name
        why = f"--jobs {a.jobs}: {str(e)[:400]}"
        report.log_refusal(why)
        raise _NotActive(why) from e


class _NotActive(RuntimeError):
    """A refusal already printed as the NOT ACTIVE line (exit 3)."""


def settled_rc(rep: dict, counts: dict, allow_partial: bool) -> int:
    """The exit of a run after the kit's evidence is read back (stack.settle): EXIT_NOT_ACTIVE on a PARTIAL activation without
    --allow-partial (the outputs and the manifest stay), else the run's own outcome — EXIT_FAILED when an item failed (incomplete), else EXIT_OK."""
    if rep.get("partial") and not allow_partial:
        return EXIT_NOT_ACTIVE
    return EXIT_OK if counts["failed"] == 0 else EXIT_FAILED


def partial_detail(rep: dict, where: str = "") -> str:
    """The detail of a partial run's line: the levers without the kit's evidence (the manifest's components_fallback), then the reason
    (the evidence they are read from is the manifest's partial_detection); `where` names the jobs under pred --jobs."""
    return f"{','.join(rep.get('components_fallback') or [])}{where}: components_fallback (no true KitRunner.effective_flags() entry after the run)"


def partial_refusal(rep: dict, where: str = "") -> str:
    """The NOT ACTIVE line of a partial run without --allow-partial (printed once, after the run; report.PARTIAL_FMT — the outputs and
    the manifest stay)."""
    return report.log_partial(partial_detail(rep, where), exit_code=EXIT_NOT_ACTIVE)


def partial_allowed(rep: dict, where: str = "") -> str:
    """The line of a partial run under --allow-partial (printed once, after the run; report.PARTIAL_ALLOWED_FMT): the opt-out is recorded
    in the manifest (allow_partial) and the exit is the run's own."""
    return report.log_partial(partial_detail(rep, where))


def run_extra(counts: dict, allow_partial: bool) -> dict:
    """The manifest's exit-rule keys: allow_partial as given; incomplete `<ok>/<items>` when an item failed."""
    return {"allow_partial": bool(allow_partial), **({"incomplete": f"{counts['ok']}/{counts['items']}"} if counts["failed"] else {})}


def cmd_pred(a, mode: str) -> int:
    from . import inputs
    from . import stack
    if mode == "off":
        try:
            planned = _multi_jobs(a)
        except _NotActive:
            return EXIT_NOT_ACTIVE
        if planned is None and not (a.input and a.out):
            raise CliError("pred needs --input and --out")
        if planned is None:
            r = subprocess.run(stock_command(a), env=stock_env(a.det))        # the script's lines go straight to this process's streams
            return r.returncode
        worst = EXIT_OK                                                        # --jobs under off: one clean stock process per job, in file order; the exit code is the worst job's
        for inp, out, items in planned:
            job = argparse.Namespace(input=inp, out=out, det=a.det, items=None, tracks=a.tracks)
            r = subprocess.run(stock_command(job), env=stock_env(a.det))
            worst = max(worst, r.returncode)
        return worst
    try:
        planned = _multi_jobs(a)
    except _NotActive:
        return EXIT_NOT_ACTIVE
    items = None
    if planned is None:
        if not (a.input and a.out):
            raise CliError("pred needs --input and --out")
        try:
            items = inputs.list_items(a.input)
            if a.items:
                want = [x.strip() for x in a.items.split(",") if x.strip()]
                missing = [w for w in want if w not in {i for i, _ in items}]
                if missing:
                    raise CliError(f"--items: not under --input: {missing}")
                items = [(i, p) for i, p in items if i in want]
        except inputs.InputError as e:
            raise CliError(str(e)) from e
    from . import det as _det, loop, manifest, multi, weights
    from . import settings as _settings
    try:
        _settings.parse_tracks(a.tracks)                                   # --tracks named wrongly is a usage error before anything loads
    except (ValueError, OSError) as e:
        raise CliError(str(e)) from e
    st = _settings.DEFAULT if (a.tracks or "all") == "all" else _settings.Settings(slices=a.tracks)
    if a.det:
        _det.export_env()                                                  # before torch is imported
    rep = stack.activate(mode, strict=False, trigger="pred", jobs=(len(planned) if planned is not None else None))
    if not rep.get("active"):
        return EXIT_NOT_ACTIVE
    import torch
    numerics = _settings.apply_numerics(st)
    det_rec = _det.apply() if a.det else None
    pins = stack.read_pins()
    out_dirs = [a.out] if planned is None else [out for _, out, _ in planned]
    t0 = time.perf_counter()
    try:
        models, wrecs = weights.load_replicates(pins, st.replicates, "cuda", after_each=lambda m, rec: stack.apply_to(m, trigger="pred"))
    except Exception as e:  # noqa: BLE001
        reason = f"{type(e).__name__}: {str(e)[:300]}"
        if not isinstance(e, stack.ActivationError):
            report.log_refusal(f"load failed: {reason}")
        for out in out_dirs:
            manifest.write(out, dict(stack.status(), active=False, reason=reason), command="pred", argv=sys.argv, exit_code=EXIT_NOT_ACTIVE,
                           settings=dict(st.as_dict(), numerics_read_back=numerics), det=det_rec, weights=None, items=None)
        return EXIT_NOT_ACTIVE
    torch.cuda.synchronize()
    load_once = time.perf_counter() - t0
    report.emit(report.ready_line(len(models), load_once))
    if planned is None:
        counts = loop.run_items(models, items, a.out, st, "cuda")
        rep = stack.settle()
        rc = settled_rc(rep, counts, a.allow_partial)
        manifest.write(a.out, rep, command="pred", argv=sys.argv, exit_code=rc, settings=dict(st.as_dict(), numerics_read_back=_settings.read_back()), det=det_rec,
                       weights=wrecs, items=counts, replicates=rep.get("models"), extra=run_extra(counts, a.allow_partial))
        if rc == EXIT_NOT_ACTIVE:
            partial_refusal(rep)
        elif rep.get("partial"):
            partial_allowed(rep)
        return rc
    n = len(planned)
    items_file = os.path.abspath(a.jobs)
    report.emit(report.multi_ready_line(mode, load_once, n))

    partial_jobs = []                                                      # the jobs whose settle read a fallback (the exit is 3 without --allow-partial)

    def after_job(i, inp, out, counts, wall):                              # the job's own manifest: the one-process route's keys + the form's block
        rep = stack.settle()
        rc = settled_rc(rep, counts, a.allow_partial)
        if rep.get("partial"):
            partial_jobs.append((i, rep))
        manifest.write(out, rep, command="pred", argv=sys.argv, exit_code=rc, settings=dict(st.as_dict(), numerics_read_back=_settings.read_back()), det=det_rec,
                       weights=wrecs, items=counts, replicates=rep.get("models"),
                       extra={"multi": {"on": True, "jobs": n, "job_index": i, "input": inp, "items_file": items_file, "load_once_s": round(float(load_once), 4),
                                        "items_wall_s": round(float(wall), 4)}, **run_extra(counts, a.allow_partial)})

    t1 = time.perf_counter()
    res = multi.run_jobs(models, planned, st, "cuda", after_job=after_job)
    rc = EXIT_OK if res["failed"] == 0 else EXIT_FAILED
    report.emit(report.multi_done_line(n, res["ok"], res["failed"], load_once, time.perf_counter() - t1))
    if partial_jobs:                                                       # every job's manifest names its own fallback; the process prints one line and exits 3 once without --allow-partial
        where = f" in {len(partial_jobs)}/{n} job(s) ({','.join(str(i) for i, _ in partial_jobs)})"
        if not a.allow_partial:
            partial_refusal(partial_jobs[-1][1], where=where)
            return EXIT_NOT_ACTIVE
        partial_allowed(partial_jobs[-1][1], where=where)
    return rc


def cmd_check(a, mode: str) -> int:
    import json
    from . import stack
    rep = stack.activate(mode, dry_run=True, trigger="check")
    report.emit(report.dry_run_line(rep))
    if getattr(a, "json", False):
        print(json.dumps({k: v for k, v in rep.items() if k != "resolution"}, indent=1, default=str))
    return EXIT_NOT_ACTIVE if rep.get("would_refuse") else EXIT_OK


def cmd_warm(a, mode: str) -> int:
    from . import warm
    res = warm.run(mode, allow_partial=a.allow_partial)
    report.emit(report.warm_line(res))
    if res.get("status") == "NOT ACTIVE":
        return EXIT_NOT_ACTIVE
    if res.get("status") == "PARTIAL":                                     # the forward ran; the kit's evidence shows a fallback: the running verb's rule
        if not a.allow_partial:
            partial_refusal(res)
            return EXIT_NOT_ACTIVE
        partial_allowed(res)
        return EXIT_OK
    return EXIT_OK if res.get("status") == "PASS" else EXIT_FAILED
