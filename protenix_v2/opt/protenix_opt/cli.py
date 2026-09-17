"""python -m protenix_opt {pred,check,warm} [--mode exact|fast|big|off] ...

A thin command layer over the kit code. It never re-implements a lever, a schema or a test:

* ``pred``   — ``protenix_opt.enable(mode)``, then the stock ``protenix pred`` IN-PROCESS with the remaining arguments passed through
               unchanged; afterwards the exit tally is printed. The output directory holds exactly stock's files.
               ``--mode off``: the stock CLI in a FRESH subprocess (``stock_pred.py``) with every name under stock/PINS.json's must-be-absent
               prefixes stripped and no kit directory on PYTHONPATH; the subprocess proves its environment before importing protenix and
               prints the proof's lines. A process under the env.sh route (the kit's sitecustomize active) refuses ``off``.
               Every stock ``protenix pred`` option (``--seeds``, ``--cycle``, ``--step``, ``--sample``, ``--dtype``, …: the stock
               parser's own names and defaults) passes through verbatim on both routes; ``[--det 0|1]``
               takes the deterministic level: ``0`` (the default) applies nothing, ``1`` applies the kit's own deterministic recipe
               (det.py: ``CUBLAS_WORKSPACE_CONFIG=:4096:8 PTX_DET=1`` + the detref scatter copy into the installed protenix package,
               restored when the run ends) on BOTH arms — before ``enable()`` on exact|fast; around the stock child on off, where the
               child's environment carries exactly the recipe and nothing else of the kit (``det.stock_exception``: the det names,
               ``$FPF_HOME/src`` alone on PYTHONPATH so the kit's sitecustomize sets ``torch.use_deterministic_algorithms``; the
               subprocess proves that carve-out and nothing beyond it). ``--mode``/``--det`` are this package's own options, taken out of the stock
               argv before either route.
* ``check``  — the pin, version and GPU gates and the activation line for a mode (a dry run: nothing else runs); with ``--det 1``
               also the recipe's preconditions (DET lines: detref files present, the protenix package and its writable utils/, no backup
               or stale copy). DET problems refuse (pred --det 1).
               The console-script test guards the ENV route only (``PROTENIX_OPT=<mode> protenix pred`` needs the installed .pth autoload in
               the script's interpreter): the package installed and the script's interpreter not importing it is the hazard (CHECK FAIL, rc 1);
               the package importable but not installed (a PYTHONPATH tree — the CLI route runs in-process) is an advisory line, rc unchanged.
* ``warm``   — the JIT warm-up: one small ``pred`` on the shipped example input (``warm_input.json``: one 76-residue protein chain,
               no MSA; one trunk cycle, a 2-step / 1-sample diffusion — local constants, not knobs) under the caller's ``--mode`` and
               ``--det``, so the mode's Triton / extension kernels are compiled into the cache directories before the first real call;
               ``--out_dir DIR`` (default: a fresh temporary directory) is the only other option; exit codes are pred's.

THE EXIT RULE: a running verb never exits 0 with a degradation only in a status field.
(a) A named fallback — a PARTIAL activation (``levers_fallback`` non-empty: a lever of the mode's set on this card cannot run — no
    launch cells, a compile / launch failure, an unsupported shape or dtype — at activation or in the kit's end-of-run records) exits
    ``EXIT_NOT_ACTIVE`` (3) by name: a mode is all of its levers on the card, it never runs a subset under its name, and there is no
    opt-out. An untested environment (an arch, library version or cells key the tables do not list, a cache miss) is NOT a fallback: the lever
    engages and the line names the uncertainty.
    ``pred`` judges it twice: at activation (before the model runs) and at the end, after the activation report is reconciled with the
    kit's own end-of-run records (``stack.LATE_RECORDS``: the graphed sampler installs after ``InferenceRunner.init_model``, i.e. after
    the activation report; a record saying installed False moves the lever to the fallbacks with the kit's why) — the FINAL line
    prints the reconciled state and a partial found there exits 3 after the work, outputs in place. ``check`` (a dry run)
    returns 3 for the same condition.
(b) A documented gate is not a fallback: an installed lever's own counters (the stack-graph memory guard, the sampler's token gate)
    are recorded under ``gates``; ``PROTENIX_OPT_FORCE=1`` is a gate OVERRIDE (the version / no-CUDA gate), distinct from
    rc 0.
(c) OUTPUTS SHORT: after the stock CLI returns, ``pred`` counts what it wrote against the request (every input entry × seeds × samples,
    a CIF and its summary JSON each: outputs.py) — fewer than expected is the OUTPUTS line's ``incomplete`` with the expected / found
    counts and the missing names, ``EXIT_FAIL`` (1); a stock failure keeps its own exit code. Both routes.
(d) The env route (``PROTENIX_OPT=<mode>``, the autoload) RECORDS the same states in its exit line and lever-report record
    (``exit_gate: env-route-records`` — it cannot change the stock process's exit code); the CLI verb GATES (``exit_gate: cli``).
Exit codes, one per condition across verbs:
    EXIT_OK 0          ok
    EXIT_FAIL 1        the verb's own check failed: incomplete outputs, DET preconditions, console script
    EXIT_USAGE 2       usage
    EXIT_NOT_ACTIVE 3  the mode did not activate, or activated only part of its lever set — refused by name (pred, check)
    otherwise          pred passes the stock exit code through

The core interface (``enable``/``status``/``MODES``/``registry``) lives in the package itself; this module only calls it.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

from . import _core  # noqa: F401
from . import _frozen, det, manifest, outputs, report, tp
from opt_core import stock_proof as _proof
from opt_core.report import EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_ACTIVE

PROG = "python -m protenix_opt"
DEFAULT_MODES = ("exact", "fast", "big", "off")
DEFAULT_MODE = "fast"
LEVER_REPORT_NAME = "opt_lever_report.jsonl"


def records_dir() -> str:
    """The directory of this run's kit records (the lever report when PTX_LEVER_REPORT is unset, the multi-GPU line's ``tp/``: phase
    logs, torchrun logs, the ranks' lever reports, run metadata): a fresh temporary directory, printed once — never the stock output
    directory, which holds exactly stock's files."""
    import tempfile
    d = tempfile.mkdtemp(prefix="protenix_opt_records_")
    print(f"{report.PREFIX} records: {d} ({LEVER_REPORT_NAME}, {tp.PHASE_LOG_DIR}/)", file=sys.stderr, flush=True)
    return d
STOCK_PROOF_NAME = "stock_env_proof.json"                          # pred --mode off: the stock subprocess's environment proof (stock_pred.py), in a temporary directory of the call


USAGE = f"""usage: {PROG} <command> [--mode exact|fast|big|off] ...

commands
  pred    [--mode exact|fast|big|off] [--det 0|1] <stock `protenix pred` arguments>
                                                             drop-in: levers on, then the stock CLI in-process (off: in a clean, proven subprocess); the output directory holds exactly stock's files;
                                                             every stock option (--seeds/--cycle/--step/--sample/--dtype ...: the stock parser's names and defaults) passes through verbatim;
                                                             --det 0 applies nothing (the default); --det 1 applies the kit's deterministic recipe on both arms
                                                             ({' '.join(f'{k}={v}' for k, v in det.DET_ENV.items())} + the detref scatter copy into the installed protenix package, restored at the end);
                                                             a PARTIAL activation (a lever of the mode's set on this card cannot run, at activation or in the kit's end-of-run
                                                             records) exits 3 by name: a mode is all of its levers, never a subset under its name (an untested environment is named, not refused)
  check   [--mode exact|fast|big|off] [--det 0|1] [--json]
                                                             the pin / version / GPU gates and the activation line (what would be active), with --det 1 the
                                                             recipe's preconditions (DET lines); runs nothing;
                                                             exit 3 when the dry run is not active or partial; the console-script test guards
                                                             the env route only: CHECK FAIL (exit 1) when the package is installed here and the stock `protenix`
                                                             script's interpreter does not import it, an advisory line when the package is not installed (PYTHONPATH)
  warm    [--mode exact|fast|big|off] [--det 0|1] [--out_dir DIR]
                                                             the JIT warm-up: one small pred on the shipped example input (warm_input.json: one 76-residue chain, no MSA;
                                                             --cycle 1 --step 2 --sample 1) so the mode's kernels are compiled before the first real call;
                                                             --out_dir defaults to a fresh temporary directory; takes no stock knob; exit codes are pred's
exit codes: 0 ok | 1 the verb's own check failed (DET, console script) | 2 usage | 3 levers not active — refused by name,
            or partial (pred, check, warm) | pred and warm otherwise pass the stock exit code through
"""


class CliError(Exception):
    def __init__(self, msg: str, code: int = EXIT_USAGE):
        super().__init__(msg)
        self.code = code


# ----------------------------------------------------------------------------------------------------------------- core interface
def valid_modes(core=None) -> tuple[str, ...]:
    modes = getattr(core, "MODES", None) if core is not None else None
    return tuple(modes) if isinstance(modes, dict) and modes else DEFAULT_MODES


def check_mode(core, mode: str) -> str:
    modes = valid_modes(core)
    if mode not in modes:
        raise CliError(f"unknown --mode {mode!r} (choose from {', '.join(modes)})")
    return mode


def activate(core, mode: str, dry_run: bool = False, det_level: int | None = None) -> dict:
    """``enable(mode)`` and the activation report; ``dry_run`` is the core's real dry run (``stack.activate(dry_run=True)``: resolve,
    gate, report, apply nothing; ``det_level`` 1 applies the deterministic recipe's lever exclusion to it). Exceptions become a NOT
    ACTIVE report (never a traceback without a reason). The core logs the activation line itself (``rep["logged"]``); the CLI never
    prints a second copy."""
    from . import stack
    try:
        if dry_run:
            rep = stack.activate(mode, dry_run=True, det=True if det_level else None)
        else:
            rep = core.enable(mode)
    except Exception as e:  # noqa: BLE001 — the reason is reported on the activation line
        return {"active": False, "mode": mode, "reason": f"protenix_opt.enable({mode!r}) raised {e!r}"}
    if not isinstance(rep, dict):
        return {"active": False, "mode": mode, "reason": f"protenix_opt.enable({mode!r}) returned {type(rep).__name__}, not the activation report dict"}
    rep.setdefault("mode", mode)
    return rep


def split_option(argv: list[str], option: str, values: str) -> tuple[str | None, list[str]]:
    """Remove every ``<option> X`` / ``<option>=X`` from ``argv`` (last one wins) — the port's own options, which the stock parser
    rejects; returns (value or None, everything else unchanged and in order)."""
    value = None
    rest: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == option:
            if i + 1 >= len(argv):
                raise CliError(f"{option} needs a value ({values})")
            value = argv[i + 1]
            i += 2
            continue
        if a.startswith(option + "="):
            value = a.split("=", 1)[1]
            i += 1
            continue
        rest.append(a)
        i += 1
    return value, rest




def split_mode(argv: list[str], default: str | None = DEFAULT_MODE) -> tuple[str | None, list[str]]:
    """Remove every ``--mode X`` / ``--mode=X`` from ``argv`` (last one wins); everything else is returned unchanged and in order."""
    mode, rest = split_option(argv, "--mode", "exact|fast|big|off")
    return (mode if mode is not None else default), rest



def split_det(argv: list[str], default: int = det.DEFAULT) -> tuple[int, list[str]]:
    """Remove every ``--det X`` / ``--det=X`` (last one wins); returns (level, rest) with the level one of ``det.LEVELS`` (anything else
    is a usage error)."""
    raw, rest = split_option(argv, "--det", "|".join(str(l) for l in det.LEVELS))
    if raw is None:
        return default, rest
    try:
        level = int(raw)
    except ValueError:
        raise CliError(f"--det {raw!r}: expected {'|'.join(str(l) for l in det.LEVELS)}") from None
    if level not in det.LEVELS:
        raise CliError(f"--det {level}: expected {'|'.join(str(l) for l in det.LEVELS)}")
    return level, rest


EXIT_GATE_CLI = "cli"                                             # report.set_exit_gate: this verb exits with the states its lines name


def outputs_gate(out_dir: str | None, params: dict | None, rc: int, failures: list | None = None) -> tuple[dict, int]:
    """The exit rule for the outputs: the census against the request, printed; a stock exit 0 with fewer outputs than asked (or an
    expectation that cannot be computed) becomes EXIT_FAIL — a stock failure keeps its own code. ``failures`` (the runner's per-item
    failure records, runner_hooks.failures()) are named in the census and on the line. Returns (census, rc)."""
    cen = outputs.census(out_dir, params)
    if failures:
        cen["failures"] = list(failures)
    print(outputs.line(cen, report.PREFIX), file=sys.stderr, flush=True)
    if rc == 0 and cen["status"] != "complete":
        return cen, EXIT_FAIL
    return cen, rc


def partial_gate(rep: dict, when: str) -> bool:
    """The exit rule for one activation report: a partial activation prints the refusal line naming every fallen-back lever with the
    kit's reason and returns False (the caller exits EXIT_NOT_ACTIVE: a mode is all of its levers on this card, never a subset under its
    name; there is no opt-out); a full activation returns True silently."""
    if not rep.get("partial"):
        return True
    print(report.partial_refusal_line(rep, when), file=sys.stderr, flush=True)
    return False


def det_patch(level: int) -> det.Patch | None:
    """Level 1: the recipe's plan, refused by name on any problem, else applied (env exported, detref copy in place) — BEFORE any
    protenix import; the caller restores it when the run ends. Level 0: None."""
    if not level:
        return None
    p = det.plan()
    if p["problems"]:
        raise CliError("--det 1 refused: " + "; ".join(p["problems"]), EXIT_FAIL)
    det.apply_env()
    try:
        return det.Patch(p).apply()
    except det.DetError as e:
        raise CliError(str(e), EXIT_FAIL)


# ----------------------------------------------------------------------------------------------------------------- pred
def stock_cli():
    """The stock click group ``protenix`` (``runner.batch_inference:protenix_cli``, the console-script entry point of protenix 2.0.0)."""
    try:
        from runner.batch_inference import protenix_cli
    except ImportError as e:
        raise CliError(f"the stock protenix CLI is not importable (runner.batch_inference): {e!r}", EXIT_NOT_ACTIVE) from e
    return protenix_cli


def stock_pred_params(rest: list[str]) -> tuple[dict | None, str | None]:
    """The stock ``pred`` command's own parser applied to ``rest`` (resilient: no callbacks, no exit) -> (params, error)."""
    try:
        cmd = stock_cli().commands["pred"]
        ctx = cmd.make_context("pred", list(rest), resilient_parsing=True)
        return dict(ctx.params), None
    except CliError:
        raise
    except Exception as e:  # noqa: BLE001
        return None, f"stock pred argument parsing failed: {e!r}"


def run_stock_cli(args: list[str]) -> int:
    """Invoke the stock CLI in-process exactly like the ``protenix`` console script; returns its exit code."""
    cli = stock_cli()
    from . import phase_timing
    phase_timing.install()                                               # the per-item PHASE line: the stock model's trunk / sampler / confidence seconds (timing only)
    try:
        cli.main(args=list(args), prog_name="protenix", standalone_mode=True)
    except SystemExit as e:
        code = e.code
        if code is None:
            return 0
        return code if isinstance(code, int) else 1
    return 0


def stock_env_names(environ, keep: tuple[str, ...] = ()) -> list[str]:
    """The names of ``environ`` under stock/PINS.json's must-be-absent prefixes (opt_core.stock_proof.forbidden), sorted, minus ``keep``."""
    from . import stack
    return [k for k in _proof.forbidden(environ, stack.stock_env_absent()) if k not in keep]


def strip_stock_env(environ, keep: tuple[str, ...] = ()) -> list[str]:
    """Pop every must-be-absent name from ``environ`` except ``keep``; returns the names popped."""
    names = stock_env_names(environ, keep)
    for k in names:
        environ.pop(k, None)
    return names


def stock_command(rest: list[str], proof_path: str, environ: dict | None = None, det_exception: dict | None = None) -> tuple[list[str], dict, dict]:
    """The stock subprocess (``stock_pred.py``) for ``pred --mode off``: its command line and its environment — ``environ`` (the process
    environment by default) without every name under stock/PINS.json's must-be-absent prefixes and without the PYTHONPATH entries inside a
    kit directory. With ``det_exception`` (det.stock_exception(): the recipe's names and values, its one kit PYTHONPATH entry) the
    environment carries exactly that on top of the stripped one and the subprocess is told to prove it. Returns ``(command, env, stripped)``
    with ``stripped = {"env": [names removed], "pythonpath": [entries removed]}``."""
    from . import stack
    environ = os.environ if environ is None else environ
    absent = stack.stock_env_absent()
    kit_dirs = stack.kit_class_dirs()
    env, stripped_names = _proof.strip_env(environ, absent)
    kept, stripped_entries = _proof.strip_pythonpath(environ.get("PYTHONPATH"), kit_dirs)
    keep = [p for p in kept.split(os.pathsep) if p] if kept else []
    if det_exception:
        env.update(det_exception["env"])                                     # the recipe's values, whatever the caller had
        keep = list(det_exception["pythonpath"]) + [p for p in keep if p not in det_exception["pythonpath"]]
    cmd = _proof.stock_command(sys.executable, "protenix_opt.stock_pred", proof_json=proof_path, env_absent=absent, kit_dirs=kit_dirs,
                               args=["pred", *rest], det=det_exception)
    stripped = {"env": sorted(k for k in stripped_names if k not in env), "pythonpath": [p for p in stripped_entries if p not in keep]}   # what the child does NOT get (a det name re-added is not stripped)
    if keep:
        env["PYTHONPATH"] = os.pathsep.join(keep)
    else:
        env.pop("PYTHONPATH", None)
    return cmd, env, stripped


def stock_pred_route(core, rest: list[str], record: dict | None = None, patch: det.Patch | None = None) -> int:
    """``pred --mode off``: the activation report for off (refused under the env.sh route), the stock parser for ``--out_dir``, then the
    stock CLI in a clean subprocess that proves its environment first (``stock_pred.py`` prints the proof's lines; its JSON goes to a
    temporary directory of the call, never the output directory). With ``patch`` (``--det 1``, applied by the caller before the child
    spawns) the child carries exactly the recipe (``det.stock_exception``) and is restored after it exits."""
    rep = activate(core, "off")
    report.log_activation(rep)
    if rep.get("refused"):
        return EXIT_NOT_ACTIVE
    params, perr = stock_pred_params(rest)                                  # the stock parser, for the output directory only
    out_dir = (params or {}).get("out_dir")
    proof_path = os.path.join(tempfile.mkdtemp(prefix="protenix_opt_stock_"), STOCK_PROOF_NAME)
    exception = det.stock_exception() if patch is not None else None
    cmd, env, stripped = stock_command(rest, proof_path, det_exception=exception)
    print(f"{report.PREFIX} stock subprocess: {' '.join(cmd[:4])} ... (env stripped of {','.join(stripped['env']) or 'nothing'}; "
          f"PYTHONPATH stripped of {','.join(stripped['pythonpath']) or 'nothing'}"
          + (f"; det exception: {' '.join(f'{k}={v}' for k, v in exception['env'].items())} PYTHONPATH+={':'.join(exception['pythonpath'])}" if exception else "")
          + ")", file=sys.stderr, flush=True)
    try:
        rc = subprocess.call(cmd, env=env)
    finally:
        if patch is not None:
            patch.restore()                                                  # after the child exits
    cen, rc = outputs_gate(out_dir, params, rc)                             # the exit rule: outputs short -> incomplete, EXIT_FAIL
    return rc


def tp_pred_route(core, n: int, rest: list[str], record: dict, patch: det.Patch | None, level: int) -> int:
    """``pred --mode big --n_gpu P`` (P > 1): the multi-GPU line (tp.py). This process launches and judges; it activates nothing itself (the
    autoload is disarmed here — every rank activates the base mode through the ENV route). The exit rule: the preflight (P visible GPUs,
    the unit fully present) and a partial base mode refuse by name (EXIT_NOT_ACTIVE); the launcher's exit code passes through; the
    outputs gate and the ranks' events gate (world size N, every rank to its last phase) turn a clean exit into EXIT_FAIL by name."""
    from . import stack
    stack._disarm_autoload()
    pf = tp.preflight(n)
    if not pf["ok"]:
        print(f"{report.PREFIX} NOT ACTIVE: {pf['reason']} (mode={tp.LINE_MODE} line={tp.LINE})", file=sys.stderr, flush=True)
        return EXIT_NOT_ACTIVE
    rep = activate(core, tp.BASE_MODE, dry_run=True, det_level=level)   # what every rank will activate: resolved and gated here, applied there
    if rep.get("env") is None:
        report.log_activation(rep)
        return EXIT_NOT_ACTIVE
    if not partial_gate(rep, f"dry run of the base mode {tp.BASE_MODE}"):
        return EXIT_NOT_ACTIVE
    params, perr = stock_pred_params(rest)
    out_dir = (params or {}).get("out_dir")
    if not out_dir:
        raise CliError(f"the multi-GPU line needs --out_dir (stock's outputs live under it): {perr or 'no out_dir in the stock parameters'}")
    why = tp.gate_refusal()                                              # a caller's open size gate of the unit: refused by name, before anything is ACTIVE
    if why:
        print(f"{report.PREFIX} NOT ACTIVE: {why} (mode={tp.LINE_MODE} line={tp.LINE} {tp.active_fields(n)})", file=sys.stderr, flush=True)
        return EXIT_NOT_ACTIVE
    rec_dir = records_dir()                                              # the ranks' phase logs, torchrun logs, lever reports and run metadata: never under --out_dir
    env, seed_word, seed_fields = tp.launch_env(n, rec_dir, exports=rep.get("env"), det_level=level)   # ONE reading of this process's environment: the ranks' environment,
                                                                         # the ACTIVE line's hashseed= word and the record's hashseed / hashseed_source fields
    rep = dict(rep, mode=tp.LINE_MODE, line=tp.LINE, base_mode=tp.BASE_MODE, n_gpu=n, sharding=tp.SCHEME, active=True, dry_run=False,
               gates=dict(tp.TP_GATES), census={k: list(v) for k, v in tp.CENSUS.items()}, **seed_fields)
    print(tp.active_line(report.PREFIX, n, rep, det_level=level, hashseed=seed_word), file=sys.stderr, flush=True)
    inp = tp.input_path(rest)
    if inp and os.path.isfile(inp):
        toks = tp.item_tokens(inp); block, why = tp.block_for(toks, n)
        if block is None:
            print(f"{report.PREFIX} NOT ACTIVE: {why} (mode={tp.LINE_MODE} line={tp.LINE} {tp.active_fields(n)})", file=sys.stderr, flush=True)
            return EXIT_NOT_ACTIVE
        env[tp.ENV_BLOCK] = str(block)
        print(f"{report.PREFIX} TP-LAYOUT items={len(toks)} tokens_min={min(toks)} tokens_max={max(toks)} n_gpu={n} block={block} {tp.rows_line(toks, n, block)} ({why})", file=sys.stderr, flush=True)
    if patch is not None:
        env.update(det.DET_ENV)                       # the kit's recipe on every rank (the detref copy is already in place on disk)
    rc, log_path = tp.run(n, rest, rec_dir, env=env)
    ev = tp.events(rec_dir, n, block=int(env[tp.ENV_BLOCK]) if env.get(tp.ENV_BLOCK) else None)
    cen, rc = outputs_gate(out_dir, params, rc)                          # the exit rule: outputs short -> incomplete, EXIT_FAIL
    if rc == 0 and not ev["ok"]:
        rc = EXIT_FAIL                                                   # a run that did not hold N ranks to the end is never exit 0
    rep = tp.reconcile_ranks(rep, env["PTX_LEVER_REPORT"], n)             # every rank's own records: applied in all, fallen back in any, what executed
    print(tp.execution_line(report.PREFIX, rep), file=sys.stderr, flush=True)
    print(tp.final_line(report.PREFIX, rep, ev, n), file=sys.stderr, flush=True)   # the one FINAL of a tp run (line=tp; never the single line's grammar)
    if not partial_gate(rep, "after the run: the ranks' end-of-run records"):
        rc = EXIT_NOT_ACTIVE if rc == 0 else rc                          # a clean run with a lever fallen back in a rank exits 3 (the kit's exit rule)
    if patch is not None:
        patch.restore()
    return rc



def weights_note(refresh: bool = False) -> dict:
    """Print the WEIGHTS line (pinned | unknown | absent — manifest.weights_status against stock/PINS.json) and return the
    record. A checkpoint that is not the pinned one is named and the command proceeds; no entry point refuses on it. The digest comes
    through the on-disk memo (manifest.cache_dir()/weights_digests.json): `pred` takes a memo hit ('… (cached digest <utc>)'),
    `check` passes ``refresh=True`` and always hashes afresh. Under --n_gpu > 1 this launching process digests before the ranks start;
    the ranks never hash."""
    from . import stack
    ws = manifest.weights_status((stack.pins().get("checkpoint") or {}).get("sha256"), refresh=refresh)
    print(f"{report.PREFIX} {ws['line']}", file=sys.stderr, flush=True)
    return ws


def cmd_pred(argv: list[str]) -> int:
    mode, rest = split_mode(argv)
    level, rest = split_det(rest)                     # the port's own options come out before either route: the stock parser rejects them
    try:
        n_gpu_flag, rest = tp.split_n_gpu(rest)
        n_gpu, _selector = tp.selection(n_gpu_flag)
    except tp.TpError as e:
        raise CliError(str(e))
    if any(a in ("-h", "--help") for a in rest):
        from . import stack
        stack._disarm_autoload()                      # a help text activates nothing: the import of the stock CLI must not trigger the finder
        print(f"{PROG} pred [--mode exact|fast|big|off] [{tp.FLAG_NGPU} P] [--det 0|1] <stock `protenix pred` arguments>; "
              f"the stock arguments are:", file=sys.stderr)
        return run_stock_cli(["pred", "--help"])
    record = {}                                       # recorded beside the effective argv
    import protenix_opt as core                       # the core interface: enable/status/MODES (this package)
    check_mode(core, mode)
    why = tp.refusal(mode, n_gpu)                     # n_gpu > 1 belongs to mode big (the shared core's sentence; usage, never a silent 1-GPU run); n_gpu 1 runs under any mode
    if why:
        raise CliError(why)
    why = _frozen.check(os.environ, rest)             # frozen-weights gate (PROTENIX_ROOT_FROZEN=1; every pred route): the stock CLI downloads an absent
    if why:                                           # cache/checkpoint (stock/src/runner/inference.py:291-347) — refused by name before any route runs;
        print(f"{report.PREFIX} NOT ACTIVE: {why}", file=sys.stderr, flush=True)   # the checkpoint = the effective --model_name's; the switch unset: nothing
        return EXIT_NOT_ACTIVE
    weights_note()                                    # WEIGHTS pinned | unknown | absent: named, never a refusal
    patch = det_patch(level)                          # --det 1: env exported + detref copy in place BEFORE any protenix import (both routes)
    guard = det.guard_signals(patch)                  # a TERM / HUP / INT while the copy is in place unwinds through the finally below (det.Terminated)
    try:
        return _pred_routes(core, mode, n_gpu, rest, record, patch, level)
    finally:                                          # EVERY exit path of every route restores the detref copy (Patch.restore is idempotent; atexit stays the net)
        if patch is not None:
            patch.restore()
        det.unguard_signals(guard)


def _pred_routes(core, mode: str, n_gpu: int, rest: list[str], record: dict, patch: det.Patch | None, level: int) -> int:
    """The three pred routes (the multi-GPU line, the stock subprocess, the kit modes in-process) under pred's det guard."""
    if tp.line_selected(mode, n_gpu):
        return tp_pred_route(core, n_gpu, rest, record, patch, level)
    if mode == "off":
        return stock_pred_route(core, rest, record, patch)
    from . import stack
    user_lever_report = os.environ.get("PTX_LEVER_REPORT")
    report.register_exit_tally()                      # before the levers load: atexit is LIFO, the tally must run after the kit's dump
    report.set_exit_gate(EXIT_GATE_CLI)               # this verb exits with the states it records (the env route only records them)
    rc = None; out_dir = None; perr = None; params = None; cen = None
    try:
        rep = activate(core, mode)
        os.environ["PROTENIX_OPT"] = mode             # child processes (MSA search, featurizers) autoload the same mode
        report.log_activation(rep)
        if not rep.get("active"):
            return EXIT_NOT_ACTIVE
        params, perr = stock_pred_params(rest)
        out_dir = (params or {}).get("out_dir")
        rec_dir = records_dir()                           # the kit's own run records (the lever report) live OUTSIDE --out_dir: the output dir holds exactly stock's files
        if not partial_gate(rep, "at activation"):           # the exit rule, before the model runs: a mode is all of its levers on this card
            rc = EXIT_NOT_ACTIVE
        else:
            if not user_lever_report:
                os.environ["PTX_LEVER_REPORT"] = os.path.join(rec_dir, LEVER_REPORT_NAME)
            from . import runner_hooks
            runner_hooks.install_failure_record()                           # the runner's per-item failure lines become records (OOM named by class)
            rc = run_stock_cli(["pred", *rest])
            cen, rc = outputs_gate(out_dir, params, rc, runner_hooks.failures())   # the exit rule: outputs short -> incomplete, EXIT_FAIL; failures named
            rep = stack.reconcile(rep, stack.late_records(os.environ.get("PTX_LEVER_REPORT"), os.getpid()))
            print(report.final_line(rep), file=sys.stderr, flush=True)
            report.log_lever_lines(rep)                                  # one LEVER line per registry lever (S6: the per-lever evidence of this arm's process)
            if not partial_gate(rep, "after the run: the kit's end-of-run records"):   # the exit rule again, on the reconciled state
                rc = EXIT_NOT_ACTIVE if rc == 0 else rc                     # a stock failure / incomplete outputs keep their code; a clean run with a late fallback exits 3
    finally:
        if patch is not None:
            patch.restore()                           # the stock call has returned (the module is loaded): the package dir goes back to stock
    return rc


# ----------------------------------------------------------------------------------------------------------------- check
STOCK_CONSOLE_SCRIPT = "protenix"                                 # the stock CLI entry point (a pip console script bound to ONE interpreter)


def tp_check(core, n: int, a) -> int:
    """``check --mode big --n_gpu P`` (P > 1): the line's preflight and the base mode's dry run; prints one DRY-RUN line for the line;
    exit 3 when the preflight or the base mode refuses (not active, or partial)."""
    pf = tp.preflight(n)
    rep = activate(core, tp.BASE_MODE, dry_run=True, det_level=a.det)
    report.log_activation(rep)
    ok = pf["ok"] and rep.get("env") is not None and partial_gate(rep, f"dry run of the base mode {tp.BASE_MODE}")
    print(f"{report.PREFIX} {'DRY-RUN' if ok else 'NOT ACTIVE:'} mode={tp.LINE_MODE} line={tp.LINE} {tp.active_fields(n)} impl={tp.IMPL} base={tp.BASE_MODE} "
          f"visible_gpus={pf.get('visible_gpus', 'unknown')} unit_files={pf.get('unit_files', 0)} nccl={pf.get('nccl', 'unknown')} "
          f"line_args={','.join(tp.LINE_ARGS)} strategy={tp.STRATEGY}" + ("" if pf["ok"] else f" reason={pf['reason']}"), file=sys.stderr, flush=True)
    if a.json:
        print(json.dumps({"tp_preflight": pf, "base": rep}, indent=1, default=str))
    return EXIT_OK if ok else EXIT_NOT_ACTIVE


def console_script_interpreter(script: str) -> str | None:
    """The interpreter a console script's shebang names (pip writes the absolute path), or None when it is not a `#!` script."""
    try:
        with open(script, "rb") as fh:
            first = fh.readline(200)
    except OSError:
        return None
    if not first.startswith(b"#!"):
        return None
    line = first[2:].decode("utf-8", "replace").strip().split()
    if not line:
        return None
    if os.path.basename(line[0]) == "env" and len(line) > 1:                 # `#!/usr/bin/env python`: whatever PATH resolves
        return shutil.which(line[1]) or line[1]
    return line[0]


PACKAGE_DIST = "protenix_opt"                                     # the distribution name (opt/pyproject.toml [project] name)
NOT_INSTALLED_REASON = "protenix_opt not installed (PYTHONPATH route; the CLI route runs in-process)"


def package_installed(name: str = PACKAGE_DIST) -> dict:
    """Whether this package is installed as a distribution in the calling interpreter (pip / pip -e: the .pth autoload lives in its site),
    or only importable through PYTHONPATH: {"installed": bool, "version": str | None}."""
    import importlib.metadata
    try:
        return {"installed": True, "version": importlib.metadata.distribution(name).version}
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False, "version": None}


def check_console_script(core, script_name: str = STOCK_CONSOLE_SCRIPT) -> dict:
    """The console-script test guards the ENV route only: `PROTENIX_OPT=<mode> protenix pred` activates only when the interpreter of the
    stock script's shebang imports this protenix_opt (the .pth hook never loads in another one — stock would run silently while the env
    check passes). The CLI route (run.sh / python -m protenix_opt) imports the package in the calling interpreter and runs the stock CLI
    in-process, so it never depends on the script. Decisive test: the script's interpreter imports protenix_opt (-I: PYTHONPATH ignored)
    and it is this package (same file). ``status``: ``ok`` — it does; ``fail`` (``ok`` False) — the package IS installed as a distribution
    here and the script's interpreter still does not import it: the real hazard; ``advisory`` (``ok`` None) — the env route is unavailable
    for a named reason that is not a hazard: the package is importable but not installed (a PYTHONPATH tree: NOT_INSTALLED_REASON), no
    script on PATH, or a script whose interpreter cannot be read."""
    ours = os.path.abspath(core.__file__)
    script = shutil.which(script_name)
    dist = package_installed()
    out = {"status": None, "script": script, "interpreter": None, "ours": sys.executable, "package": ours, "installed": dist["installed"],
           "version": dist["version"], "imports_this_package": None, "ok": None, "reason": None}
    if script is None:
        out["status"] = "advisory"
        out["reason"] = f"no `{script_name}` command on PATH: the env route (PROTENIX_OPT=<mode> {script_name} pred) is unavailable here; the CLI route runs in-process"
        return out
    interp = console_script_interpreter(script)
    out["interpreter"] = interp
    if interp is None:
        out["status"] = "advisory"
        out["reason"] = f"{script} is not a `#!` console script; its interpreter could not be read"
        return out
    try:
        r = subprocess.run([interp, "-I", "-c", "import protenix_opt, os; print(os.path.abspath(protenix_opt.__file__))"], capture_output=True, text=True, timeout=60)
        theirs = r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired) as e:
        theirs = None; r = None
        out["reason"] = f"could not run {interp}: {e!r}"
    out["imports_this_package"] = theirs is not None and os.path.realpath(theirs) == os.path.realpath(ours)
    if out["imports_this_package"]:
        out["status"], out["ok"] = "ok", True
        return out
    if not dist["installed"]:                                     # importable here through PYTHONPATH alone: the env route cannot exist; not a hazard
        out["status"], out["ok"] = "advisory", None
        out["reason"] = NOT_INSTALLED_REASON + (f"; {out['reason']}" if out["reason"] else "")
        return out
    out["status"], out["ok"] = "fail", False
    if out["reason"] is None:
        if theirs is None:
            out["reason"] = (f"the `{script_name}` command on PATH ({script}) runs {interp}, which cannot import protenix_opt (installed into {sys.executable}): "
                             f"PROTENIX_OPT=<mode> {script_name} pred would run STOCK silently. Install protenix_opt into the interpreter that owns the "
                             f"{script_name} command: {interp} -m pip install -e {os.path.dirname(os.path.dirname(ours))}")
        else:
            out["reason"] = f"{interp} imports a different protenix_opt ({theirs}) than this one ({ours}); one install per interpreter"
    return out


def console_script_line(cs: dict) -> str | None:
    """The line `check` prints for the console-script result: FAIL for the hazard, an advisory naming why the env route is unavailable,
    nothing for ok."""
    if cs["status"] == "fail":
        return f"{report.PREFIX} CHECK FAIL: {cs['reason']}"
    if cs["status"] == "advisory":
        return f"{report.PREFIX} CHECK console script: env route unavailable — {cs['reason']}"
    return None


def cmd_check(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog=f"{PROG} check", allow_abbrev=False)
    p.add_argument("--mode", default=DEFAULT_MODE)
    p.add_argument("--det", type=int, default=det.DEFAULT, choices=det.LEVELS, help="1: also dry-run the deterministic recipe's preconditions (DET lines)")
    p.add_argument("--json", action="store_true", help="also print the full activation report as JSON")
    p.add_argument(tp.FLAG_NGPU, type=int, default=None, help=f"the GPU count P (default 1; P > 1 with --mode {tp.LINE_MODE} = the multi-GPU line: preflight + the base mode's dry run)")
    a = p.parse_args(argv)
    import protenix_opt as core                       # the core interface: enable/status/MODES (this package)
    mode = check_mode(core, a.mode)
    try:
        n_gpu, _selector = tp.selection(a.n_gpu)
    except tp.TpError as e:
        raise CliError(str(e))
    why = tp.refusal(mode, n_gpu)
    if why:
        raise CliError(why)
    if tp.line_selected(mode, n_gpu):
        return tp_check(core, n_gpu, a)
    det_plan = det.plan() if a.det else None               # --det 1: the recipe's preconditions, printed whole, ok or the named problems
    if det_plan is not None:
        for line in det.lines(det_plan, report.PREFIX):
            print(line, file=sys.stderr, flush=True)
    rep = activate(core, mode, dry_run=True, det_level=a.det)   # the core prints the DRY-RUN / NOT ACTIVE line (one formatter: report.py)
    weights_note(refresh=True)                                                # the checkpoint's digest note (never a refusal; `check` hashes afresh and rewrites the memo entry)
    report.log_activation(rep)                             # only when the core could not (rep["logged"] unset: the call raised)
    would_activate = (mode == "off" and not rep.get("refused")) or rep.get("env") is not None
    would_run = would_activate and partial_gate(rep, "dry run")   # the exit rule on the dry run: a partial activation is refused by name
    rep["det"] = det_plan
    cs = check_console_script(core)                        # the env route's guard: FAIL on the hazard, an advisory line otherwise; the JSON report carries the result always
    rep["console_script"] = cs
    line = console_script_line(cs)
    if line:
        print(line, file=sys.stderr, flush=True)
    if a.json:
        print(json.dumps(rep, indent=1, default=str), flush=True)
    if cs["ok"] is False or (det_plan is not None and det_plan["problems"]):
        return EXIT_FAIL                                   # the check's own failures first: the console-script hazard / the recipe cannot apply
    return EXIT_OK if would_run else EXIT_NOT_ACTIVE       # not active or partial: the code pred exits with


# ----------------------------------------------------------------------------------------------------------------- entry
WARM_INPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "warm_input.json")   # the shipped example input: one 76-residue protein chain
WARM_STOCK_ARGS = ("--seeds", "101", "--cycle", "1", "--step", "2", "--sample", "1", "--use_msa", "false")   # one trunk cycle, a 2-step / 1-sample
# diffusion, no MSA search: every stage of the pipeline runs once, which is what compiles the mode's kernels; none of these is a user-facing knob
WARM_PORT_FLAGS = ("--mode", "--det")                    # the port's own valued options warm forwards to pred


def warm_argv(argv: list[str]) -> tuple[list[str], str]:
    """``warm``'s pred argv: the port options given (``--mode``/``--det``) first, then the shipped input, the out_dir
    (``--out_dir DIR`` when given, else a fresh temporary directory) and ``WARM_STOCK_ARGS``; any other token is a usage error (warm takes
    no stock knob)."""
    port, out_dir, i = [], None, 0
    while i < len(argv):
        a = argv[i]
        name = a.split("=", 1)[0]
        if name in WARM_PORT_FLAGS or name == "--out_dir":
            if "=" in a:
                val = a.split("=", 1)[1]; i += 1
            elif i + 1 < len(argv):
                val = argv[i + 1]; i += 2
            else:
                raise CliError(f"warm: {name} needs a value")
            if name == "--out_dir":
                out_dir = val
            else:
                port += [name, val]
        else:
            raise CliError(f"warm takes [--mode M] [--det 0|1] [--out_dir DIR] only, not {a!r}: it runs one fixed small pred ({' '.join(WARM_STOCK_ARGS)})")
    out_dir = out_dir or tempfile.mkdtemp(prefix="protenix_opt_warm_")
    return port + ["--input", WARM_INPUT, "--out_dir", out_dir, *WARM_STOCK_ARGS], out_dir


def cmd_warm(argv: list[str]) -> int:
    """The JIT warm-up: one small ``pred`` on the shipped example input under the caller's mode and det level (``warm_argv``); prints
    what it runs, then pred's own lines; returns pred's exit code."""
    pargv, out_dir = warm_argv(argv)
    print(f"{report.PREFIX} WARM: pred on {os.path.basename(WARM_INPUT)} (one 76-residue chain; {' '.join(WARM_STOCK_ARGS)}) -> {out_dir}", file=sys.stderr, flush=True)
    return cmd_pred(pargv)


COMMANDS = {"pred": cmd_pred, "check": cmd_check, "warm": cmd_warm}


def main(argv: list[str] | None = None) -> int:
    from . import _autoload
    _autoload.refuse_undeclared(os.environ)           # a mistyped PROTENIX_OPT* name: the hook's own refusal (exit 3), here too for a process without the .pth
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE, end="")
        return EXIT_OK if argv else EXIT_USAGE
    cmd = argv[0]
    fn = COMMANDS.get(cmd)
    if fn is None:
        print(f"{report.PREFIX} unknown command {cmd!r}\n{USAGE}", end="", file=sys.stderr)
        return EXIT_USAGE
    try:
        return fn(argv[1:])
    except CliError as e:
        print(f"{report.PREFIX} ERROR: {e}", file=sys.stderr, flush=True)
        return e.code
