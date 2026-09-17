"""python -m colabdesign_opt {design,check,warm} ...

A thin command layer over the design script and the package's arms. It re-implements no lever, no mode, no test:

* ``design`` — one binder design (stock_design.py: the upstream recipe) for a case named as BindCraft names it (`--starting-pdb`,
  `--chains`, `--binder-len`, `--target-hotspot-residues`, `--seed`, `--binder-name`; the settings files as `bindcraft.py --advanced /
  --filters`, default its own two) in a subprocess per arm.
  ``--mode off``: the stock arm (stock_launch.py, the environment proven clean). ``--mode exact|fast|<word>``: the kit arm (kit_launch.py:
  the mode's levers installed before the script starts). Outputs under ``--out``: design.pdb,
  design.fasta, trajectory.jsonl and the arm's log run.log, beside the tree BindCraft's design step creates there; the arm's counts and timing come back to this process as its
  printed ``[run]`` line(s), read back in memory (report.run_lines; `design()`'s return value).
* ``check`` — the activation line for a mode without applying anything (route=check); ``--json`` prints the full report.
* ``warm``      — one design through the mode's own line (warm.py): BindCraft's bundled example (the case its
                  `settings_target/PDL1.json` names: `example/PDL1.pdb`, its chains, hotspot and first binder length), or the case given
                  by `--starting-pdb/--chains/--binder-len`, designed once under `<out>/warm/`; the product is the measured one-time
                  costs (params load, the first call of each stage) on the ``WARM`` line.

Mode = ``--mode`` when given, else COLABDESIGN_OPT, else the package default; a ``--mode`` that disagrees with a set COLABDESIGN_OPT is
refused. The params root = ``--params-dir``, else COLABDESIGN_PARAMS_DIR.
Exit codes: 0 ok | 1 the design failed, or the design's file set is short (named on a line) | 2 usage |
3 not active — refused by name, the reason printed: colabdesign absent or installed from another commit than the pin (that changes what "stock"
is), the kit's core absent, an unknown mode (configuration, never a lever); and PARTIAL — a lever that INSTALLED in the arm and did not engage
(``evidence.verdict``: no exit line of its own, a kernel that served no call or fell back for an undeclared reason): the outputs are not what
the LEVER lines claim, a defect, exit 3. A lever that cannot engage here is neither: it steps aside BY NAME at install (the kit rule —
`LEVER … state=skipped reason=cannot_run|no_attention_kernel`; e.g. a Pallas kernel with no GPU lowering, the compile cache on an unwritable
directory, proj / txla without an attention-kernel lever in the run), the mode runs the rest of its set, ``skipped: <lever>: <why>`` is
printed and EVIDENCE carries ``skipped=``, exit 0. nosub at or below the size gate is ``gated`` (stock's programs by
construction), recorded, exit 0 | else the arm's code. The rest of the environment — a card other than the tested one or below the kernel's
compute-capability floor, a stack distribution off its pin — is named on the ACTIVE line (``gpu=``, ``stack_drift=``) and never refused.
``warm`` exits as its ``design`` does. The env and api routes (``COLABDESIGN_OPT=fast`` in front of the stock command, ``colabdesign_opt.enable``)
cannot set the host process's exit code after activation: a lever that disengages there is what the levers' own LEVER lines record and ``design`` is the gated form.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

from . import driver, evidence, inputs, modes, names, outputs, report, settings, stack

PROG = "colabdesign-opt"
ENV_PARAMS = "COLABDESIGN_PARAMS_DIR"
EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_ACTIVE = report.EXIT_OK, report.EXIT_FAIL, report.EXIT_USAGE, report.EXIT_NOT_ACTIVE   # the tree's exit codes (opt_core.report)


class CliError(Exception):
    def __init__(self, msg: str, code: int = EXIT_USAGE):
        super().__init__(msg)
        self.code = code


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog=PROG, description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    d = sub.add_parser("design", help="one BindCraft design trajectory (binder_hallucination at the pinned settings), in a subprocess per arm")
    d.add_argument("--mode", default=None, help=f"{'|'.join(modes.MODES)} (default: {modes.ENV} or {modes.DEFAULT_MODE})")
    d.add_argument("--starting-pdb", required=True, help="target PDB file (BindCraft's starting_pdb)")
    d.add_argument("--chains", required=True, help="target chain(s): A or A,B,C (BindCraft's chains)")
    d.add_argument("--binder-len", type=int, required=True, help="the binder's length (one value; ColabDesign's binder_len)")
    d.add_argument("--target-hotspot-residues", default=None, help="BindCraft's target_hotspot_residues, verbatim (56 | 31,33,57 | A31,A33 | A1-10,56); absent = none — the case decides, nothing is invented")
    d.add_argument("--seed", type=int, default=0, help="the design's seed (default 0)")
    d.add_argument("--advanced", default=None, metavar="FILE", help="bindcraft.py's --advanced: the advanced settings json (default its settings_advanced/default_4stage_multimer.json, vendored under stock/src/bindcraft)")
    d.add_argument("--filters", default=None, metavar="FILE", help="bindcraft.py's --filters: the filters json (default its settings_filters/default_filters.json, vendored under stock/src/bindcraft)")
    d.add_argument("--out", required=True, help="output directory (created)")
    d.add_argument("--params-dir", default=None, help=f"AlphaFold params root (default {ENV_PARAMS}, else BindCraft's folder stock/src/bindcraft)")
    d.add_argument("--binder-name", default="design", help="BindCraft's binder_name: the trajectory is named <binder-name>_l<binder-len>_s<seed>")

    c = sub.add_parser("check", help="dry run: resolve and gate a mode on this box; nothing is applied")
    c.add_argument("--mode", default=None)
    c.add_argument("--json", action="store_true", help="print the full activation report as JSON on stdout")
    c.add_argument("--params-dir", default=None, help=f"AlphaFold params root (default {ENV_PARAMS}): its params files are digested (sha256) and worded pinned | NOT PINNED by name")

    w = sub.add_parser("warm", help="one design through the mode's line: BindCraft's bundled example (or the case you name) designed once; the one-time costs measured")
    w.add_argument("--mode", default=None)
    w.add_argument("--out", required=True, help="output directory: the design's files under <out>/warm/")
    w.add_argument("--params-dir", default=None, help=f"AlphaFold params root (default {ENV_PARAMS}, else BindCraft's folder stock/src/bindcraft)")
    w.add_argument("--starting-pdb", default=None, help="a target PDB instead of the bundled example (with --chains and --binder-len)")
    w.add_argument("--chains", default=None); w.add_argument("--binder-len", type=int, default=None)
    w.add_argument("--seed", type=int, default=0)

    try:
        return ap.parse_args(argv)
    except SystemExit as e:
        raise CliError("usage", EXIT_USAGE if e.code else EXIT_OK)


def _mode_arg(mode: Optional[str]) -> Optional[str]:
    env = modes.mode_from_env()
    if mode and env and mode.strip().lower() != env:
        raise CliError(f"--mode {mode} disagrees with {modes.ENV}={env} in the environment: unset one", EXIT_USAGE)
    return mode or env


def _params_dir(arg: Optional[str]) -> str:
    """--params-dir, else COLABDESIGN_PARAMS_DIR, else BindCraft's own default: its folder (functions/generic_utils.py:241-242 af_params_dir =
    bindcraft_folder; ColabDesign reads <root>/params/). A root without params/ is refused by name, as the design would fail on it."""
    v = arg or os.environ.get(ENV_PARAMS) or os.path.join(stack.tree_home(), "stock", "src", "bindcraft")
    if not os.path.isdir(os.path.join(v, "params")):
        raise CliError(f"--params-dir {v}: no params/ directory holding params_model_*_multimer_v3.npz (colabdesign/af/model.py:120 reads <root>/params/; "
                       f"--params-dir or {ENV_PARAMS} names another root)", EXIT_USAGE)
    return os.path.abspath(v)


def _resolve(mode) -> modes.Resolved:
    try:
        return modes.resolve(_mode_arg(mode))
    except ValueError as e:
        raise CliError(str(e), EXIT_NOT_ACTIVE)


def cmd_check(a) -> int:
    res = _resolve(a.mode)
    rep = stack.activate(res.mode, route="check", dry_run=True, settings=settings.NAME)
    report.log(report.active_line(rep))
    params = a.params_dir or os.environ.get(ENV_PARAMS)
    if params and os.path.isdir(os.path.join(params, "params")):             # the weights' digests, computed now (sha256 per file, no memo): pinned | NOT PINNED by name — a report, never a gate
        cert, unc = weights_words(os.path.abspath(params), cached=False); rep["weights"] = cert + unc
        for line in cert + unc:
            report.log(line)
    else:
        report.log(f"weights=not checked (no --params-dir / {ENV_PARAMS} with a params/ directory)")
    if a.json:
        print(json.dumps(rep, indent=1, default=str))
    return EXIT_OK if rep.get("active") else EXIT_NOT_ACTIVE




def check_pins_module():
    """(`stock/check_pins.py` as a module, stock/PINS.json) — stack.load_check_pins + stack.pins: the ONE home of the pin words (the weights words
    check_weights / weights_lines / sha256_cached; the interpreter rule python_pin)."""
    return stack.load_check_pins(), stack.pins()


def weights_words(params_dir: str, cached: bool = True) -> tuple:
    """(pinned lines, not-pinned lines) — warn-and-run weights digests by sha256 (`stock/check_pins.py` `check_weights` + `weights_lines`, the ONE
    wording) over --params-dir. `cached` (the design start): digests through check_pins' (path, size, mtime_ns) memo, so an unchanged params file is
    hashed once per box, never identified by size alone; `cached=False` (the `check` verb): digests now. An unknown checkpoint = one
    `weights=<file> sha256=<12> NOT PINNED — …` line per file whose digest is not the pin and the design PROCEEDS — never refused; the pin
    (stock/PINS.json `weights`) is a note of which weights this tree was tested with, nothing more. An ABSENT params file is bindcraft.refusals' (by name)."""
    m, pins = check_pins_module()
    _absent, detail = m.check_weights(pins, params_dir, cache=m.WEIGHTS_CACHE if cached else None)
    return m.weights_lines(detail)


def design(a) -> dict:
    """One `design`: the activation, the arm in its subprocess, the exit rule. Returns the in-memory record {exit_code, activation, case,
    mode, levers, evidence, verdict, runs, files, out_dir} — `runs` = the arm's `[run]` line per design that ended, read
    back (report.run_lines: tokens, steps, terminate, timing with `ready_s` = the arm's launch up to that design call), in printed order;
    [] when the arm printed none (refused, or failed before a design ended). `files` = the outputs' digests read from ``--out``
    (outputs.files). Under ``--out``: the outputs and run.log, beside BindCraft's own directory tree."""
    res = _resolve(a.mode)
    params_dir = _params_dir(a.params_dir)
    try:
        S = settings.load(stack.tree_home(), a.advanced, a.filters)        # the settings files (bindcraft.py's --advanced / --filters, default its own), sha256 recorded; the arm reads them again in its own process
    except settings.SettingsError as e:
        raise CliError(str(e), EXIT_NOT_ACTIVE)
    try:
        tgt = inputs.target(a.starting_pdb, a.chains)
    except (ValueError, FileNotFoundError) as e:
        raise CliError(str(e), EXIT_USAGE)
    out_dir = os.path.abspath(a.out)
    os.makedirs(out_dir, exist_ok=True)
    case = inputs.case(tgt, a.binder_len, a.seed, res.mode, settings.iteration_counts(S["advanced"]), hotspot=a.target_hotspot_residues)
    rep = stack.activate(res.mode, route="subprocess", dry_run=True, settings=S["name"])
    report.log(report.active_line(rep))
    rec = {"exit_code": EXIT_NOT_ACTIVE, "activation": rep, "case": case, "mode": res.mode, "levers": list(res.levers), "evidence": None,
           "verdict": {"partial": {}, "gated": {}, "skipped": {}}, "runs": [], "files": {}, "out_dir": out_dir}
    if not rep.get("active"):
        return rec
    report.log(report.run_line(case))
    for line in weights_words(params_dir)[1]:                                # warn-and-run: an unknown checkpoint (sha256 ≠ pin) is named here and the design proceeds
        report.log(line)
    hotspot_args = ["--target-hotspot-residues", a.target_hotspot_residues] if a.target_hotspot_residues else []   # the case's hotspots verbatim; absent = BindCraft's none
    settings_args = [*(["--advanced", S["path"]] if a.advanced else []), *(["--filters", S["filters"]] if a.filters else [])]   # given files travel by absolute path; absent = BindCraft's defaults in the arm too
    script_args = ["--starting-pdb", tgt.path, "--chains", ",".join(tgt.chains), "--binder-len", str(a.binder_len), *hotspot_args, "--seed", str(a.seed),
                   "--params-dir", params_dir, "--out", out_dir, "--binder-name", a.binder_name, *settings_args]
    argv, env, notes = driver.compose(res, script_args, out_dir=out_dir)
    result = driver.launch(argv, env, cwd=out_dir, log_path=os.path.join(out_dir, names.RUN_LOG), timeout=None)
    rec["runs"] = report.run_lines(result["lines"], result["stamps"])       # the arm's [run] line per design that ended, read back: its counts and timing as printed, ready_s from the SETTINGS line's arrival
    rc = result["exit_code"]
    ev = None
    if res.route == "kit":
        ev = evidence.classify(result["lines"], res.levers, res.ablated, getattr(res, "restored", ()))
        report.log(report.evidence_line(ev))
    else:
        stock_lines = evidence.stock_lever_lines(result["lines"])
        if stock_lines:
            report.log(f"NOT ACTIVE reason=the stock arm printed {len(stock_lines)} lever line(s): {stock_lines[0][:120]}")
            rc = EXIT_NOT_ACTIVE if rc == 0 else rc
    fs = outputs.files(out_dir)
    missing = outputs.complete(fs)
    if rc == 0 and result["timed_out"]:
        rc = EXIT_FAIL
    verdict = evidence.verdict(ev) if ev is not None else {"partial": {}, "gated": {}, "skipped": {}}
    rc = partial_exit(rc, verdict)                                       # one precedence for every engine: a failed arm > partial > incomplete
    if rc == 0 and missing:
        report.log(f"design incomplete: missing {','.join(missing)} (the file set is short of the request)")
        rc = EXIT_FAIL
    rec.update({"exit_code": rc, "evidence": ev, "verdict": verdict, "files": fs})
    report.log(report.exit_line(rc, res, ev, out_dir))
    return rec


def cmd_design(a) -> int:
    return design(a)["exit_code"]


def partial_exit(rc: int, verdict: dict) -> int:
    """The exit rule after the arm returned ``rc``: a partial activation (``verdict["partial"]``, evidence.verdict: an INSTALLED lever of the
    mode did not engage — a defect) exits EXIT_NOT_ACTIVE by name; an arm that failed (or timed out) keeps its code with the state recorded; a
    gated lever (stock's programs by construction) and a lever that stepped aside by name at install (``verdict["skipped"]``, the kit rule)
    are named and never change the code."""
    for lever, why in verdict["gated"].items():
        report.log(f"gated: {lever}: {why}")
    for lever, why in (verdict.get("skipped") or {}).items():                 # stepped aside by name at install (the kit rule): named, exit unchanged
        report.log(f"skipped: {lever}: {why}")
    if not verdict["partial"]:
        return rc
    what = ", ".join(f"{k}: {v}" for k, v in verdict["partial"].items())
    if rc != 0:
        report.log(f"PARTIAL: {what} (recorded; the arm failed, rc={rc})")
        return rc
    report.log(report.partial_line(what))
    return EXIT_NOT_ACTIVE


def cmd_warm(a) -> int:
    from . import warm
    mode = _mode_arg(a.mode)
    try:
        res = warm.run(mode, a.out, a.params_dir, target=a.starting_pdb, chain=a.chains, binder_len=a.binder_len, seed=a.seed)
    except warm.WarmError as e:
        raise CliError(str(e), EXIT_USAGE)
    return res["exit_code"] if res["exit_code"] != 0 else (EXIT_OK if res["status"] == "PASS" else EXIT_FAIL)


COMMANDS = {"design": cmd_design, "check": cmd_check, "warm": cmd_warm}


def _dispatch(argv: Optional[List[str]]) -> tuple:
    """(exit_code, record) for one command line: the ONE home of the usage / refusal handling (`main` and `design_argv` share it)."""
    try:
        a = parse_args(argv)
        if a.command == "design":
            rec = design(a)
            return rec["exit_code"], rec
        return COMMANDS[a.command](a), None
    except CliError as e:
        if str(e) == "usage":
            return e.code, None
        report.log(f"{'NOT ACTIVE reason=' if e.code == EXIT_NOT_ACTIVE else ''}{e}")
        return e.code, None
    except FileNotFoundError as e:
        report.log(f"NOT ACTIVE reason={e}")
        return EXIT_NOT_ACTIVE, None


def main(argv: Optional[List[str]] = None) -> int:
    return _dispatch(argv)[0]


def design_argv(args: List[str]) -> dict:
    """`design <args>` in this process (warm's use): `design()`'s record; a usage error or refusal is {"exit_code": code} alone."""
    rc, rec = _dispatch(["design", *args])
    return rec if rec is not None else {"exit_code": rc}
