"""python -m colabfold_opt {pred,check,warm} [--mode off|exact|fast|big] [--n_gpu P] ...

A thin command layer over the stock command line and the kit's own switch. It never re-implements a lever or a mode, and it adds no
input grammar: `pred` takes ``colabfold_batch``'s own command line.

* ``pred``   — ``colabfold-opt pred <input> <results> [colabfold_batch options …] [--mode M] [--n_gpu P] [--det 0]``:
               ``colabfold_batch``'s two positionals (its input — a3m / fasta / csv / directory — and its results directory) and its
               options, found where its own parser finds them (inputs.split_argv) and handed to it verbatim at stock's names and defaults
               (settings.py), in every mode alike; the outputs are colabfold's own files in ``<results>/`` and nothing else — the
               package's record of the launch is its printed lines. The parameters root is ``colabfold_batch``'s ``--data <dir>`` when
               given, else ``COLABFOLD_OPT_DATA_DIR`` (added as ``--data``). ``--mode off``: ``colabfold_batch`` in a cleaned subprocess
               that proves itself (stock_pred.py: the STOCK line). ``--mode fast``: the same launcher with ``COLABFOLD_OPT=fast`` exported —
               the env route, the one activation implementation (stack.py). Before the launch the gates run in the parent (``pred_gates``:
               the parameters in every mode, every gate for a kit mode; rc 3 on refusal); after the model process exits the verdict is read
               from the manifest it wrote into the launch's work directory (a temporary directory named to it as ``COLABFOLD_OPT_WORK_DIR``;
               removed on return) and colabfold's ``<id>.done.txt`` markers
               (manifest.verdict: a kit mode must show its activation report and, when active, ``calls >= 1`` at exit; every job its
               marker) — the FAILED / NOT ACTIVE line names the reason; the child's return code alone is never the verdict. A mode is all
               of its levers: a lever of the mode that cannot run on this GPU (``lever_fallback``) refuses the run by name before any compute,
               and a lever that engaged no call by the end of the run (``kernel_not_engaged``) is a PARTIAL activation — the line
               ``NOT ACTIVE: partial activation — <levers>: <reason>; exit 3`` (report.partial_exit_line), exit 3; there is no route that
               runs a mode with a subset of its levers (``--mode off`` runs stock). A kernel that engaged but fell back to the stock operation
               beyond the documented class of calls (``fallback_excess``, manifest.fallback_census) is recorded and named on ONE line —
               ``PARTIAL fallback_excess: <levers>: <counters>; the calls that fell back ran the stock operation — recorded, exit 0`` — and
               the run keeps its own verdict and exit code. A command line on which ``colabfold_batch`` builds no model — ``--num-models 0`` /
               ``--msa-only``, ``--af3-json``, or a results directory whose every job is already complete (colabfold skips them; complete =
               its own test: ``<job>.result.zip`` under ``--zip``, else ``<job>.done.txt``) — exits as stock does in every mode: no marker is
               owed, the mode's levers had no call to serve, and ONE line names the case (``IDLE no_model_run=<num_models_0|all_jobs_done|af3_json>
               levers=not_applicable:<names> …``, report.idle_line) — never a partial activation, never ``missing_outputs``.
* ``check``  — dry run: resolves and gates the mode on this machine (kit, pins, jax range, GPU, parameters) and applies nothing; prints the
               DRY-RUN line; imports neither jax nor colabfold.
* ``warm``   — one public-input prediction through ``pred`` (warm.py): the route end to end.

Mode = ``--mode`` when given, else ``COLABFOLD_OPT`` from the environment, else ``modes.DEFAULT_MODE`` (fast); ``--mode`` and a set
``COLABFOLD_OPT`` must agree; a name outside the table is refused by name (modes.UnsupportedMode). ``--det 0`` (the default) applies
nothing and is recorded as ``det`` in the launch record; ``--det 1`` is refused by name before anything runs (modes.NO_DET_REASON). Exit codes: 0 ok · 1 failed, or outputs short of the request (``missing_outputs``) · 2 usage · 3 not active, or partial
— ``warm`` exits as its ``pred`` does. The env route (``COLABFOLD_OPT=fast colabfold_batch``)
owns its exit (the strict hook, stack.hook_run: 3) and records ``kernel_not_engaged`` only (``partial`` in the manifest the model process
completes at exit, manifest.record_exit — a process cannot exit by that state); ``pred`` is the gated form of both.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import uuid
from types import SimpleNamespace
from typing import List, Optional, Tuple

from . import _autoload, ablation, inputs, manifest, modes, report, settings, stack, stock_pred

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_ACTIVE = 0, 1, 2, report.EXIT_NOT_ACTIVE
COMMANDS = ("pred", "check", "warm")
KIT_VALUED = {"--mode": "mode", "--n_gpu": "n_gpu", "--det": "det", "--out": "out"}          # the kit's own flags that take a value (--out: warm's results directory)
KIT_BARE = {"--json": "json", "-h": "help", "--help": "help"}

USAGE = f"""usage: colabfold-opt <command> [--mode {'|'.join(modes.MODES)}] [--n_gpu P] [options]
  pred    <input> <results> [colabfold_batch options ...] [--det 0|1]   colabfold_batch's own command line: its input (a3m|fasta|csv|dir), its results directory and its options, verbatim (--num-recycle N --num-models N --num-seeds N --random-seed N --data DIR ...)
  check   [--json]
  warm    [--out DIR] [--json] [colabfold_batch options ...]
mode: --mode, else {modes.ENV} from the environment, else the package default ({modes.DEFAULT_MODE}: opt/colabfold_opt/modes.py DEFAULT_MODE);
`--det 0` (the default) applies nothing and `--det 1` is refused: {modes.NO_DET_REASON}.
The parameters root: colabfold_batch's `--data <dir>`, else {stack.ENV_DATA} (added as --data).
Exit: 0 ok · 1 failed (or outputs short of the request: missing_outputs) · 2 usage · 3 not active or partial — a lever of the mode cannot run on this GPU or engaged no call (a mode is all of its levers; --mode off runs stock) / pins.
"""


def _usage(msg: Optional[str] = None) -> int:
    if msg:
        report.emit(f"{report.PREFIX} usage: {msg}")
    sys.stderr.write(USAGE)
    return EXIT_USAGE


def parse(argv: List[str]):
    """The kit's own flags — wherever they stand — and the rest: the command (the first bare token) and colabfold_batch's tokens in the
    caller's order (everything after a bare `--` is left to colabfold_batch as written). Raises ValueError on a malformed kit flag."""
    a = SimpleNamespace(command=None, mode=None, n_gpu=None, det=0, out=None, json=False, help=False)
    rest: List[str] = []
    i = 0
    while i < len(argv):
        t = argv[i]
        if t == "--":
            rest += argv[i:]
            break
        name, eq, val = t.partition("=")
        if name in KIT_VALUED and (eq or i + 1 < len(argv)):
            value = val if eq else argv[i + 1]
            i += 1 if eq else 2
            if name in ("--n_gpu", "--det"):
                try:
                    value = int(value)
                except ValueError:
                    raise ValueError(f"{name} {value!r}: an integer is required") from None
            if name == "--det" and value not in (0, 1):
                raise ValueError(f"--det {value}: 0 or 1")
            setattr(a, KIT_VALUED[name], value)
            continue
        if name in KIT_VALUED:
            raise ValueError(f"{name}: a value is required")
        if t in KIT_BARE:
            setattr(a, KIT_BARE[t], True)
        elif a.command is None and not t.startswith("-"):
            a.command = t
        else:
            rest.append(t)
        i += 1
    return a, rest


def resolve_mode(flag: Optional[str]) -> str:
    """--mode, else COLABFOLD_OPT, else the default; a disagreement is a usage error; unknown names raise UnsupportedMode."""
    env = (os.environ.get(modes.ENV) or "").strip().lower()
    if flag and env and flag != env:
        raise ValueError(f"--mode {flag} disagrees with {modes.ENV}={env}; a run has one mode — drop one of them")
    mode = flag or env or modes.DEFAULT_MODE
    modes.resolve(mode)
    return mode


def n_gpu_mem_fraction(n_gpu: int, env: dict) -> Optional[str]:
    """The jax memory-pool fraction `pred` starts a ``big --n_gpu P`` model process with — ``modes.N_GPU_MEM_FRACTION[P]`` (0.90 at P = 8:
    NCCL's communicators allocate outside jax's pool and need the headroom across 8 GPUs; the P = 8 line runs at 0.90) — or None = the
    environment stays as it is: no entry for P (1, 2, 4: the image's 0.95), or the caller chose a fraction themselves — ``XLA_PYTHON_CLIENT_MEM_FRACTION``
    set to anything but the image's own preset (stock/PINS.json ``image.env``, 0.95: the stack's value, not the user's), or ``XLA_CLIENT_MEM_FRACTION``
    (jax's other name for it) set at all. A value equal to the preset cannot be told apart from the image's own ENV, so it counts as the stack's;
    any other fraction the user exported is kept."""
    want = modes.N_GPU_MEM_FRACTION.get(int(n_gpu))
    if want is None or (env.get(modes.MEM_FRACTION_ENV_ALT) or "").strip():
        return None
    current = (env.get(modes.MEM_FRACTION_ENV) or "").strip()
    if not current:
        return want
    try:
        preset = str(((stack.pins().get("image") or {}).get("env") or {}).get(modes.MEM_FRACTION_ENV) or "").strip()
    except (OSError, ValueError):                                          # no readable pins: any fraction found set is taken as the caller's
        preset = ""
    return want if preset and _same_fraction(current, preset) else None


def _same_fraction(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-9
    except ValueError:
        return a == b


def cmd_check(a, mode: str) -> int:
    rep = stack.check(mode, n_gpu=a.n_gpu, refresh_digests=True)          # `check` hashes the parameters afresh and rewrites their digest memo entries
    if a.json:
        print(report.dump(rep))
    return EXIT_NOT_ACTIVE if rep.get("would_refuse") else EXIT_OK


def pred(a, mode: str, tokens: List[str]) -> Tuple[int, Optional[dict]]:
    """`pred` over colabfold_batch's tokens (its positionals and options as written): (exit code, the launch record — manifest.final: the
    model process's activation report and exit counters, the verdict, argv, settings, the stock proof, the output listing; None when the
    launch was refused before it ran). The record lives in memory and in the launch's work directory while the model process runs;
    nothing of it is written under <results>."""
    try:
        inp, results, options, dashdash = inputs.split_argv(tokens)
    except inputs.UsageError as e:
        return _usage(f"pred: {e}"), None
    try:
        job_ids = inputs.jobs(inp, options)                                  # colabfold's own reader and job naming: the names the completion census expects
    except (OSError, ValueError, AssertionError) as e:
        return _usage(f"{inp}: {e}"), None
    passed_data = settings.flag_value(options, "--data")                   # colabfold_batch's own --data names the parameters root and rides verbatim
    data_dir = passed_data or stack.data_dir_env()
    if not data_dir:
        return _usage(f"the parameters root is required: colabfold_batch's --data <dir> or {stack.ENV_DATA} (stock/PINS.json weights; README Variables)"), None
    results_dir = os.path.abspath(results)
    refused, gate_rep = pred_gates(mode, data_dir, a.n_gpu)
    if refused:
        return refused, None
    done_before = {j: art for j in job_ids if (art := manifest.completion(results_dir, j)) is not None}   # colabfold's own finished-job test per job BEFORE the launch (<job>.result.zip | <job>.done.txt): the jobs it will skip
    expect = manifest.no_model_run(settings.models_per_seed(options), job_ids, results_dir,            # will colabfold_batch build a model at all — from its command line and the results directory as they
                                  keep_existing=not settings.flag_present(options, settings.OVERWRITE),  # stand now (--num-models 0 / --msa-only; every job complete and kept; --af3-json): the verdict follows it
                                  af3_json=settings.flag_present(options, settings.AF3_JSON))
    how = settings.MSA_ONLY if settings.flag_present(options, settings.MSA_ONLY) else ("--num-models 0" if expect == manifest.NUM_MODELS_0 else None)
    os.makedirs(results_dir, exist_ok=True)
    try:
        argv = stock_pred.compose(options, inp, results, None if passed_data else data_dir, dashdash)
    except stock_pred.LaunchError as e:
        report.emit(report.NOT_ACTIVE_FMT.format(prefix=report.PREFIX, reason=str(e)))
        return EXIT_NOT_ACTIVE, None
    launch_id = uuid.uuid4().hex                                            # this launch's id: the model process's manifest must carry it
    work = tempfile.mkdtemp(prefix="colabfold_opt_")                          # the launch's work directory (manifest.ENV_WORK_DIR): the model process's manifest, read back for the verdict —
    try:                                                                    # internal scratch, removed on return; nothing of it under <results>
        return _launch(a, mode, options, argv, results_dir, inp, data_dir, job_ids, launch_id, gate_rep, work, (expect, how, done_before))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _launch(a, mode, options, argv, results_dir, inp, data_dir, job_ids, launch_id, gate_rep, work, no_model=(None, None, None)) -> Tuple[int, dict]:
    rep, proof = None, None
    expect, how, done_before = no_model                                     # the launcher's reading of a run that builds no model (pred: manifest.no_model_run before the launch)
    fields = dict(command="pred", argv=argv, mode=mode, result_dir=results_dir, settings=settings.describe(options), det=a.det,
                  inputs=[{"id": j, "input": inp} for j in job_ids], data_dir=data_dir, launch_id=launch_id, gates=gate_rep)
    if mode == "off":
        env = stock_pred.stock_env()
        proof = stock_pred.proof(env, argv)
        report.emit(stock_pred.proof_line(proof))
        if not proof["ok"]:
            return EXIT_FAIL, manifest.final(work, None, exit_code=EXIT_FAIL, stock_proof=proof, **fields)
        rep = stack.activate("off", route="stock")
    else:
        env = stock_pred.fast_env(mode, launch_id=launch_id, work_dir=work)   # the model process's environment: the switch, the launch id, the work directory
        env.pop(modes.ENV_N_GPU, None)
        if mode in modes.TP_MODES:                                          # the model process reads the axis from its environment (modes.n_gpu_from_env)
            env[modes.ENV_N_GPU] = str(a.n_gpu)
            fraction = n_gpu_mem_fraction(a.n_gpu, env)                     # --n_gpu 8: jax's pool at 0.90 of each card in the model process (modes.N_GPU_MEM_FRACTION) unless the caller chose a fraction
            if fraction is not None:
                env[modes.MEM_FRACTION_ENV] = fraction

    def on_line(ln: str) -> None:                                           # the model process's output, streamed as it arrives
        sys.stderr.write(ln); sys.stderr.flush()
    rc = stock_pred.run_logged(argv, env, None, on_line)                     # in the caller's working directory: colabfold_batch's positionals mean what the caller wrote
    v = manifest.verdict(work, mode, job_ids, rc, launch_id=launch_id, n_gpu=a.n_gpu, result_dir=results_dir,   # the manifest of THIS launch + colabfold's completion artefacts; the GPU count asked must be the count the model process ran at;
                         no_model_run=expect, no_model_how=how, done_before=done_before)                       # a run that builds no model is named, never partial / missing_outputs
    rec = manifest.final(work, rep, exit_code=v["rc"], stock_proof=proof, outputs=manifest.listing(results_dir), verdict=v,
                         partial=v["partial"], idle=v["idle"], no_model_run=v["no_model_run"], fallback_excess=v["fallback_excess"], child_rc=rc, **fields)
    if v["ok"] and v["no_model_run"]:                                     # colabfold_batch built no model (--num-models 0 / --msa-only, every job already complete, --af3-json): ONE named line, the run's own exit code
        report.emit(report.idle_line(v["no_model_run"], v["idle"], len(job_ids), v["detail"], code=v["rc"]))
    elif v["ok"] and v["fallback_excess"]:                                # the kernel engaged but fell back beyond the documented class: ONE named line, the run's own exit code
        report.emit(report.fallback_excess_line(v["detail"]))
    if not v["ok"] and v["exit_by"] == "verdict":                          # else the model process printed its own line (exit 3)
        if v["reason"] in ("hook_never_fired", "n_gpu_mismatch"):
            report.emit(report.NOT_ACTIVE_FMT.format(prefix=report.PREFIX, reason=v["detail"]))
        elif v["reason"] == "lever_fallback":                             # (the model process printed the same line and exited 3; this branch is the verdict's own)
            report.emit(report.lever_refused_line({"lever_fallbacks": v["lever_fallbacks"]}))
        elif v["reason"] == "kernel_not_engaged":
            report.emit(report.partial_exit_line(v["detail"]))
        else:
            report.emit(report.failed_line(v))
    return v["rc"], rec


def pred_gates(mode: str, data_dir: str, n_gpu: int = 1) -> tuple:
    """Before the launch, in every mode: the parameters gate — colabfold's `main()` fetches the parameters from the network when the
    marker is absent (batch.py:2164, download.py:38-44), a network event the gate precedes — and, for a kit mode, the dry run of every
    gate (stack.check: kit, pins, jax range, GPU) without its DRY-RUN line (the run's evidence lines are the model process's; the dry-run
    report is recorded in the launch record as `gates`). Returns (rc or None, the gate report): a refusal is one NOT ACTIVE line and rc 3."""
    if mode == "off":
        try:
            reasons, det = stack.weights_check(data_dir)
        except Exception as e:  # noqa: BLE001
            reasons, det = [f"parameters gate failed: {e!r}"], None
        rep = {"mode": mode, "weights": det, "would_refuse": "; ".join(reasons) or None}
        for line in stack.weights_lines(det):                             # the parameter pin state: pinned | NOT PINNED (runs), one line per file
            report.emit(line)
        if reasons:
            report.emit(report.NOT_ACTIVE_FMT.format(prefix=report.PREFIX, reason=rep["would_refuse"]))
            return EXIT_NOT_ACTIVE, rep
        return None, rep
    rep = stack.check(mode, data_dir=data_dir, print_line=False, n_gpu=n_gpu)
    for line in stack.weights_lines(rep.get("weights")):
        report.emit(line)
    if rep.get("would_refuse"):
        report.emit(report.NOT_ACTIVE_FMT.format(prefix=report.PREFIX, reason=rep["would_refuse"]))
        return EXIT_NOT_ACTIVE, rep
    return None, rep


def main(argv: Optional[List[str]] = None) -> int:
    return run(argv)[0]


def run(argv: Optional[List[str]] = None) -> Tuple[int, Optional[dict]]:
    """The command line: (exit code, the launch record of `pred` / `warm` — see pred(); None for `check`, a usage error or a refusal
    before the launch)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        a, rest = parse(argv)
    except ValueError as e:
        return _usage(str(e)), None
    if a.help or not a.command:
        sys.stdout.write(USAGE)
        return (EXIT_OK if a.help else EXIT_USAGE), None
    if a.command not in COMMANDS:
        return _usage(f"unknown command {a.command!r}"), None
    if a.det:                                                              # before anything runs: no command of this port has a det lever
        return _usage(f"--det {a.det} is refused: {modes.NO_DET_REASON}"), None
    if a.command == "check" and rest:
        return _usage(f"check takes no colabfold_batch arguments ({' '.join(rest)})"), None
    if a.command != "warm" and a.out is not None:
        return _usage("--out is warm's (its results directory); pred takes colabfold_batch's <input> <results>"), None
    if _autoload.undeclared():                                             # a variable under the kit's prefix nothing reads (a mistyped name): refused, never stripped silently
        return _usage(f"undeclared variable(s) {', '.join(_autoload.undeclared())} (the names this kit reads: {', '.join(_autoload.ENV_NAMES)})"), None
    try:
        mode = resolve_mode(a.mode)
    except modes.UnsupportedMode as e:
        return _usage(str(e)), None
    except ValueError as e:
        return _usage(str(e)), None
    if mode == "off" and ablation.requested():                             # MODEL_OPT_LEVERS_OFF under --mode off: stock applies no lever, nothing to ablate — refused by name before any launch (ablation.py; a kit mode's names are validated by the activation, dry-run gate included)
        try:
            ablation.validate("off", ablation.requested(), ())
        except ablation.AblationError as e:
            report.emit(report.NOT_ACTIVE_FMT.format(prefix=report.PREFIX, reason=str(e)))
            return EXIT_NOT_ACTIVE, None
    a.n_gpu = 1 if a.n_gpu is None else a.n_gpu                          # --n_gpu absent == --n_gpu 1 (both print n_gpu=1 sharding=none under big)
    if a.n_gpu < 1:
        return _usage(f"--n_gpu {a.n_gpu}: a positive GPU count is required"), None
    g = stack.gate_n_gpu(mode, a.n_gpu, (stack.gpu_info() or {}).get("count") if a.n_gpu > 1 else None)
    if g:                                                                  # n_gpu>1 outside big, P outside the shipped set, fewer GPUs visible than asked, or the core's n_gpu modules absent
        report.emit(report.NOT_ACTIVE_FMT.format(prefix=report.PREFIX, reason=g) + f" (mode={mode} n_gpu={a.n_gpu})")
        return EXIT_NOT_ACTIVE, None
    if a.command == "check":
        return cmd_check(a, mode), None
    if a.command == "pred":
        return pred(a, mode, rest)
    from . import warm
    return warm.main(a, mode, rest)
