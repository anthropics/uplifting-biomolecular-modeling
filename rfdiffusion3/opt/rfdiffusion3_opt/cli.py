"""python -m rfdiffusion3_opt {design,check,warm} [--mode exact|fast|off] ...

A thin command layer over the kit code and the upstream command line. It never re-implements a lever, a mode or a test:

* ``design`` — one `rfd3 design` run: the line is upstream's own key=value overrides, verbatim (settings.py; `inputs=<spec>` `out_dir=<dir>` …),
               plus `ckpt_path=` from ``--ckpt`` / RFD3_CKPT when the line names none; `opt_manifest.json` beside the outputs.
               ``--mode off``: the stock caller ``stock_design.py`` in the pristine interpreter (RFDIFFUSION3_STOCK_PYTHON, default:
               this one) in isolated mode (``-I``: its own site-packages only, no PYTHONPATH, no user site), every lever name
               stripped and their absence proved (the STOCK line; the proof is recorded in ``opt_manifest.json`` ``stock_env_proof``). ``--mode exact``:
               ``rfdiffusion3_opt.enable("exact")`` (the delta exported, the gates passed), then upstream's own entry point
               ``rfd3.cli:app`` in this process with the same command line; the kit's module reads the delta at its import.
               ``--mode fast``: the same route with the graph switch in the delta; the sampler's evidence line follows the run.
* ``check``  — the activation line for a mode without exporting anything: kit bytes, this interpreter's file state, pins, GPU,
               environment; ``--mode off`` proves the pristine tree by sha (in RFDIFFUSION3_STOCK_PYTHON when set); ``--json``
               prints the full report.
* ``warm``   — the checkpoint load under the mode (warm.py): imports, checkpoint, model on the GPU; nothing designed. ``--mode off``
               runs in the pristine interpreter with every lever name stripped and the stock caller's proof first (``NOT STOCK`` exits 3).

Mode = ``--mode`` when given, else RFDIFFUSION3_OPT from the environment, else the package default (``modes.DEFAULT_MODE``) — the
core's mode argument (``opt_core.modes.mode_argument``): the command line wins over the variable. Exit codes (``opt_core.report``, through
report.py): 0 ok, 1 failed, 2 usage, 3 NOT ACTIVE — a refused gate, a stock process not proven stock, a line a kit mode has no route
for (upstream's compile_model=true under exact | fast, modes.COMPILE_REFUSED: `--mode off` runs it as stock), a line that asks a computation
the mode's levers do not drive (classifier-free guidance, low_memory_mode, the symmetry sampler: modes.UNSERVED — `NOT ACTIVE: mode=<m> cannot serve …`, nothing of
the run starts; `--mode off` serves it), or a PARTIAL activation without ``--allow-partial`` —, 5 KERNELS refused (a lever's accelerator the
KIT route expects absent or fallen back, census.py; the stock routes report such a miss and proceed); `design` checks its line in this order
and stops at the first refusal: usage (2) → a kit name in the environment the mode refuses (3) → a line the mode cannot serve (3) →
activation (3) → the run.

The exit rule: a kit-mode run whose planned lever has no evidence of application is partial — the kit's module read ``RFD3_HOIST``
off at its import (``design`` refuses it there, before any roll-out, through the activation's APPLIED hook; ``warm`` from its child's ``APPLIED``
line) or a completed run never imported it (``APPLIED none``) — and exits 3 with the family line
``[rfdiffusion3-opt] NOT ACTIVE: partial activation — <lever (reason)>; exit 3 (--allow-partial records and proceeds)``.
``--allow-partial`` (``design`` / ``warm``; ``RFDIFFUSION3_OPT_ALLOW_PARTIAL=1`` is the same switch in the environment, the only form on
upstream's own command line) records the allowance — ``allow_partial`` beside ``partial`` / ``levers_fallback`` / ``partial_reason`` in
``opt_manifest.json`` — prints ``PARTIAL allowed: ...`` and the exit is the run's own. ``check`` accepts and records the flag; its plan
is one delta on every GPU class (modes.resolve), so a plan is never partial and the flag changes no code there.
The checkpoint's sha256 in a ``design`` manifest comes from the package's digest cache when the file is unchanged since it was last
hashed (manifest.checkpoint_sha256).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

from . import __version__, det as _det, modes, settings
from ._emit import emit
from . import report
from .report import ENV_ALLOW_PARTIAL, EXIT_FAIL, EXIT_NOT_ACTIVE, EXIT_OK, EXIT_USAGE, PREFIX   # one exit table: opt_core.report, through report.py

COMMANDS = ("design", "check", "warm")
DESIGN_FLAGS = ("--mode", "--ckpt", "--det", "--allow-partial")   # the kit's own words on a design line; every other token is upstream's (key=value, verbatim)
DESIGN_USAGE = ("rfdiffusion3-opt design [--mode exact|fast|off] [--ckpt <file>] [--det 0|1] [--allow-partial] [--] <rfd3 design settings: key=value ...>  "
                "e.g. design --mode fast inputs=spec.json out_dir=out diffusion_batch_size=8 seed=101 — every key=value token is upstream's own override "
                "(rfd3/configs/inference_engine/rfdiffusion3.yaml), passed through verbatim; after a bare `--` everything is upstream's")
_SEALED_SPELLINGS = {"--spec": "inputs", "--out_dir": "out_dir"}   # `--spec` / `--out_dir` are read as the upstream keys `inputs=` / `out_dir=`; accepted, not documented, not printed
ENV_MODE = "RFDIFFUSION3_OPT"
ENV_CKPT = "RFD3_CKPT"
ENV_STOCK_PYTHON = "RFDIFFUSION3_STOCK_PYTHON"
PROOF_FILENAME = "stock_env_proof.json"                      # the stock child's proof, written to a temporary directory and folded into opt_manifest.json `stock_env_proof`


def _err(msg: str) -> None:
    emit(msg if msg.startswith(PREFIX) else f"{PREFIX} {msg}")


def resolve_mode_arg(mode_arg: Optional[str]) -> str:
    """The mode of this run: ``--mode`` when given, else RFDIFFUSION3_OPT, else the package default (modes.mode_of = the core's
    mode_argument over the one table). Raises ValueError for a name that is not a mode."""
    try:
        return modes.mode_of(mode_arg, os.environ.get(ENV_MODE))
    except modes.ModeError as e:
        raise ValueError(str(e)) from None


def ckpt_of(arg: Optional[str]) -> Optional[str]:
    return arg or os.environ.get(ENV_CKPT) or None


def allow_partial_of(a) -> bool:
    """``--allow-partial``, or RFDIFFUSION3_OPT_ALLOW_PARTIAL=1 in the environment (report.allow_partial_env): one value, every verb."""
    return bool(getattr(a, "allow_partial", False)) or report.allow_partial_env()


def _add_allow_partial(p, what: str) -> None:
    p.add_argument("--allow-partial", action="store_true", dest="allow_partial",
                   help=f"a partial activation (a planned lever without evidence of application) is recorded and {what}; without it a partial run exits {EXIT_NOT_ACTIVE}. {ENV_ALLOW_PARTIAL}=1 is the same switch in the environment")


def _add_mode(p) -> None:
    """``--mode exact|fast|off`` — exactly the table (modes.MODES); default None: RFDIFFUSION3_OPT, else modes.DEFAULT_MODE (resolve_mode_arg)."""
    p.add_argument("--mode", choices=list(modes.MODES), default=None, metavar="|".join(modes.MODES))


def split_design_line(tokens: List[str]):
    """(kit_flag_tokens, upstream_tokens, error) of a design line: the kit's own flags (DESIGN_FLAGS, `--x v` or `--x=v`) wherever they
    stand before a bare `--`; every `key=value` token, and everything after a bare `--`, is upstream's, kept in order. An unknown `--word`
    is a usage error named as argparse names it (`unrecognized arguments: …`) — the kit invents no name for an upstream setting."""
    kit: List[str] = []; ups: List[str] = []
    value_flags = ("--mode", "--ckpt", "--det")
    i, n = 0, len(tokens)
    while i < n:
        t = str(tokens[i])
        if t == "--":
            ups.extend(str(x) for x in tokens[i + 1:])
            break
        if t.startswith("-") and "=" not in t.lstrip("-").split("=")[0] and not t[1:2].isdigit():
            name, eq, val = t.partition("=")
            if name in _SEALED_SPELLINGS:
                if not eq:
                    if i + 1 >= n:
                        return kit, ups, f"argument {name}: expected one argument"
                    val = str(tokens[i + 1]); i += 1
                ups.append(f"{_SEALED_SPELLINGS[name]}={val}")
            elif name in value_flags:
                if eq:
                    kit.extend([name, val])
                else:
                    if i + 1 >= n:
                        return kit, ups, f"argument {name}: expected one argument"
                    kit.extend([name, str(tokens[i + 1])]); i += 1
            elif name in ("--allow-partial", "-h", "--help") and not eq:
                kit.append(name)
            else:
                return kit, ups, f"unrecognized arguments: {t} (this kit's design flags: {' '.join(DESIGN_FLAGS)}; upstream's settings are its own key=value tokens, e.g. inputs=<spec.json> out_dir=<dir> seed=<n> compile_model=true)"
        else:
            ups.append(t)
        i += 1
    return kit, ups, None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="rfdiffusion3-opt", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", metavar="|".join(COMMANDS))

    p = sub.add_parser("design", usage=DESIGN_USAGE, help="one rfd3 design run: upstream's own key=value settings verbatim (inputs=<spec> out_dir=<dir> seed=<n> …); opt_manifest.json beside the outputs")
    _add_mode(p)
    p.add_argument("--ckpt", default=None, help=f"the checkpoint file (default: ${ENV_CKPT}); becomes upstream's ckpt_path= unless the line names one")
    p.add_argument("--det", type=int, default=_det.DEFAULT_LEVEL, choices=list(_det.LEVELS), help="1 = the deterministic recipe (det.py: cuBLAS workspace, torch's deterministic algorithms, the fixed-order scatter-mean) applied identically on every arm, the stock child included; 0 (default) = production numerics")
    _add_allow_partial(p, "the run proceeds (opt_manifest.json: allow_partial)")

    p = sub.add_parser("check", help="dry run: gate the mode on this box, export nothing")
    _add_mode(p)
    p.add_argument("--json", action="store_true")
    _add_allow_partial(p, "recorded in the report (a plan is one delta on every GPU class: never partial, the flag changes no code here)")

    p = sub.add_parser("warm", help="the checkpoint load under the mode")
    _add_mode(p)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--json", action="store_true")
    _add_allow_partial(p, "the warm-up is the run's own outcome")
    return ap


# --- design ---------------------------------------------------------------------------------------------------------------

def stock_command(python: str, proof_json: str, line_args: List[str], det: int = 0) -> List[str]:
    return [python, "-I", "-m", "rfdiffusion3_opt.stock_design", "--proof-json", proof_json, "--det", str(int(det)), "--", "design"] + line_args


def stock_env(environ: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """The stock child's environment: the caller's own without the package switches, the kit levers and upstream's switches
    (stack.strip_env; the strip rule covers the kit's names and upstream's two attention switches — torch's own variables reach the child unchanged)."""
    from . import registry, stack
    env = stack.strip_env(environ)
    env.setdefault("MODEL_OPT", registry.tree_home())
    return env


def _design_off(a, line_args: List[str], pins: dict) -> int:
    from . import manifest, registry, stack
    python = stack.stock_python()
    proof_dir = tempfile.mkdtemp(prefix="rfdiffusion3_opt.")   # the stock child's proof travels through a temporary file; the out_dir holds stock's files and opt_manifest.json only
    proof_path = os.path.join(proof_dir, PROOF_FILENAME)
    cmd = stock_command(python, proof_path, line_args, det=a.det)
    env = stock_env()
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
    _err(f"design mode=off det={a.det} stock interpreter={python} command={' '.join(cmd[4:])}")
    try:
        proc = subprocess.run(cmd, env=env)
        proof = json.load(open(proof_path, encoding="utf-8")) if os.path.isfile(proof_path) else None
    finally:
        shutil.rmtree(proof_dir, ignore_errors=True)
    rep = {"active": False, "mode": "off", "env": {}, "env_line": "none",   # the stock child's environment carries no kit delta
           "stock_python": python, "package_version": __version__, "det": (proof or {}).get("det", {"level": a.det}),
           "levers_planned": [], "levers_applied": [], "levers_fallback": [], "levers_unavailable": [], "partial": False, "partial_reason": None,
           "allow_partial": allow_partial_of(a),           # recorded as given; the stock arm plans no lever, so there is nothing to allow
           "interpreter": {"label": "pristine" if (proof or {}).get("ok") else "not-proven", "detail": "opt_manifest.json stock_env_proof", "python": python}}
    if a.out_dir:                                          # beside the designs; a line without out_dir= wrote none (upstream refused it)
        manifest.write(a.out_dir, rep, settings=_settings_block(a, line_args), command=["rfd3", "design"] + line_args, argv=sys.argv, exit_code=proc.returncode,
                       checkpoint=a.ckpt, spec=a.spec, pins=pins, stock_env_proof=proof,
                       kernels=((proof or {}).get("after") or {}).get("kernels"))   # the stock child's KERNELS census (stock_design: route stock | default)
    if proc.returncode == 3 and proof is not None and not proof.get("ok"):
        _err("NOT STOCK: " + "; ".join(proof.get("reasons") or []))
        return EXIT_NOT_ACTIVE
    return proc.returncode


def _design_kit(a, mode: str, line_args: List[str], pins: dict) -> int:
    from . import ActivationError, _autoload, census, manifest, stack
    allow_partial = allow_partial_of(a)
    _autoload.disarm()                                    # an explicit --mode wins over an armed RFDIFFUSION3_OPT finder: one activation, this one
    try:
        stack.activate(mode, strict=True, trigger="design", allow_partial=allow_partial, argv=line_args, det=a.det)   # recorded whether or not the run turns out partial; installs the APPLIED hook; applies the deterministic recipe when --det 1
    except ActivationError:
        return EXIT_NOT_ACTIVE
    hook = stack.applied_hook()                           # the APPLIED line at the moment the kit's module reads the delta, and the exit rule there (None for a declined run: nothing planned)
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
    from .stock_design import run_stock                   # the same entry point as the stock arm: upstream's rfd3.cli:app
    _err(f"design mode={mode} det={a.det} command=rfd3 design {' '.join(line_args)}")
    rc = run_stock(["design"] + line_args)                # a partial refused at the import: the hook printed the line, rc is EXIT_NOT_ACTIVE; a KERNELS refusal during the run: rc is EXIT_KERNELS; a computation the mode does not drive met at initialize: its NOT ACTIVE line, rc EXIT_NOT_ACTIVE (census.NotServed)
    if census.not_served() is None:                      # refused at initialize (census.not_served): nothing ran — no lever evidence to report, no partial verdict to reach (stack._planned gates the reporters the same way)
        if hook is not None and hook.applied is None:
            stack.report_applied()                        # never imported: a completed run without evidence is partial (the report says so)
        stack.report_graph()                              # the graph sampler's evidence (exact, fast): APPLIED + LEVER lines, or partial
        stack.report_fzt()                                # the fused transition's evidence (exact, fast): APPLIED line with its call census, or partial
        stack.report_init_chunk()                         # the row-blocked initialiser embedding's evidence (exact, fast): APPLIED line with its block tally
        stack.report_gather()                             # the gather_attn lever's evidence (fast): APPLIED line with its served/stock census, or partial
    rc = census.finish(rc)                               # the KERNELS line (once) and the REQUIRE guard at the pass's end: EXIT_KERNELS when an expected accelerator is absent / fell back; EXIT_NOT_ACTIVE and no line after a refusal at initialize
    v = report.verdict(rc, stack.status(), allow_partial)
    if a.out_dir:
        manifest.write(a.out_dir, stack.status(), settings=_settings_block(a, line_args), command=["rfd3", "design"] + line_args, argv=sys.argv, exit_code=v["exit_code"],
                       checkpoint=a.ckpt, spec=a.spec, pins=pins, kernels=census.record())
    if v["partial"] and not (hook is not None and hook.partial):   # the hook did not print it (the module was never imported): the one line, here
        _err(report.partial_line(v["partial"], v["partial_reason"], v["allow_partial"], exit_code=v["exit_code"]))
    return v["exit_code"]


def _settings_block(a, line_args) -> dict:
    """The manifest's `settings`: the line's overrides as {key: value} (the spec, the output directory and the checkpoint have their own manifest keys) and the route."""
    own_keys = (settings.INPUTS_KEY, settings.OUT_DIR_KEY, settings.CKPT_KEY)
    return {"values": {k: v for k, v in settings.kv_of(line_args).items() if k not in own_keys}, "route": getattr(a, "route", None)}


def cmd_design(a) -> int:
    from . import stack
    try:
        mode = resolve_mode_arg(a.mode)
        res = modes.resolve(mode)
    except ValueError as e:
        _err(f"NOT ACTIVE: {e}")
        return EXIT_USAGE
    tokens = [str(t) for t in a.upstream]
    kv = settings.kv_of(tokens)
    named = kv.get(settings.CKPT_KEY)
    if named not in (None, ""):                           # the line names its checkpoint itself: upstream's token stands; --ckpt may only agree
        if a.ckpt and os.path.abspath(a.ckpt) != os.path.abspath(named):
            _err(f"design: --ckpt {a.ckpt} and {settings.CKPT_KEY}={named} on the line disagree")
            return EXIT_USAGE
        a.ckpt = named
    else:
        ckpt = ckpt_of(a.ckpt)
        if not ckpt:
            _err(f"design: no checkpoint named — --ckpt <file>, {ENV_CKPT}, or upstream's {settings.CKPT_KEY}=<file> on the line (the manifest hashes it; the line names it as {settings.CKPT_KEY}=)")
            return EXIT_USAGE
        a.ckpt = ckpt
        tokens.append(f"{settings.CKPT_KEY}={os.path.abspath(ckpt)}")
    a.out_dir = kv.get(settings.OUT_DIR_KEY)              # upstream's own out_dir= (the manifest goes beside the designs); a line without it is upstream's to refuse
    a.spec = kv.get(settings.INPUTS_KEY)
    line_args = settings.line(tokens)[2:]
    refused = stack.line_refusal(line_args, res.mode)     # a line the kit mode cannot serve (compile_model=true, classifier-free guidance, low_memory_mode, the symmetry sampler under exact | fast): nothing activated, nothing run (activate() gates the same line again)
    if refused:
        _err(report.not_active_line(refused))
        return EXIT_NOT_ACTIVE
    a.route = modes.route_of(res.mode, line_args).name   # the route this line runs under (modes.ROUTES): the KERNELS census expects its words
    pins = stack.pins()
    if res.mode == "off":
        return _design_off(a, line_args, pins)
    return _design_kit(a, res.mode, line_args, pins)


# --- check ----------------------------------------------------------------------------------------------------------------

def _stock_python_check() -> Optional[dict]:
    """stock/check_pins.py --expect pristine in the named stock interpreter (when it is not this one)."""
    from . import registry, stack
    python = stack.stock_python()
    if os.path.abspath(python) == os.path.abspath(sys.executable):
        return None
    r = subprocess.run([python, "-I", os.path.join(registry.tree_home(), "stock", "check_pins.py"), "--expect", "pristine", "--json"],
                       capture_output=True, text=True)
    try:
        d = json.loads(r.stdout)
    except ValueError:
        d = {"error": (r.stdout + r.stderr)[-400:]}
    d["python"], d["exit_code"] = python, r.returncode
    return d


def cmd_check(a) -> int:
    from . import stack
    try:
        mode = resolve_mode_arg(a.mode)
    except ValueError as e:
        _err(f"NOT ACTIVE: {e}")
        return EXIT_USAGE
    rep = stack.activate(mode, dry_run=True)
    not_a_mode = "env" not in rep                          # modes.resolve refused the name (not a mode)
    rep["allow_partial"] = allow_partial_of(a)             # recorded; a plan is one delta on every GPU class: never partial (modes.resolve)
    if rep.get("mode") == "off" and not not_a_mode:
        sp = _stock_python_check()
        if sp is None:
            interp = rep.get("interpreter") or {}
            sp = {"python": sys.executable, "pristine": interp.get("label") == "pristine", "exit_code": 0 if (rep.get("pins") or {}).get("ok") else 3,
                  "bad": (rep.get("pins") or {}).get("notes") or [], "detail": interp.get("detail")}
            _err(f"stock interpreter = this one: tree {interp.get('label')} ({interp.get('detail')}); " + ("pins ok" if sp["exit_code"] == 0 else "pins (reported): " + "; ".join(sp["bad"])))
        else:
            _err(f"stock interpreter {sp['python']}: " + ("pristine" if sp.get("pristine") else "NOT pristine: " + "; ".join(sp.get("bad") or [sp.get("error", "?")]))
                 + ("" if sp.get("exit_code") == 0 else " (pins reported: " + "; ".join(sp.get("bad") or [sp.get("error", "?")]) + ")"))
        rep["stock_interpreter"] = sp
    if a.json:
        print(json.dumps(rep, indent=1, default=str))
    if not_a_mode:
        return EXIT_USAGE
    if rep.get("mode") == "off":
        sp = rep["stock_interpreter"]
        return EXIT_OK if sp.get("pristine") else EXIT_NOT_ACTIVE   # the stock interpreter's rfd3 TREE is the gate; its pins are reported
    return EXIT_OK if not rep.get("findings") else EXIT_NOT_ACTIVE


# --- warm -----------------------------------------------------------------------------------------------------------------

def cmd_warm(a) -> int:
    from . import stack, warm
    try:
        mode = resolve_mode_arg(a.mode)
        modes.resolve(mode)
    except ValueError as e:
        _err(f"NOT ACTIVE: {e}")
        return EXIT_USAGE
    ckpt = ckpt_of(a.ckpt)
    python = stack.stock_python() if mode == "off" else sys.executable
    allow_partial = allow_partial_of(a)
    res = warm.run(mode, ckpt or "", python=python, allow_partial=allow_partial)
    _err(warm.summary_line(res))
    if res.get("partial"):
        _err(report.partial_line(res["partial"], res.get("partial_reason"), allow_partial))
    if a.json:
        print(json.dumps(res, indent=1, default=str))
    if res.get("error") == "CheckpointMissing":
        return EXIT_USAGE
    if res.get("error") in ("NotActive", "NotStock", "Partial"):
        return EXIT_NOT_ACTIVE
    return EXIT_OK if res["status"] == "PASS" else EXIT_FAIL


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "design":                      # the design line: the kit's flags to argparse, every other token to upstream (split_design_line)
        kit, upstream, err = split_design_line(argv[1:])
        if err:                                           # argparse's own shape (usage, then `<prog> design: error: …`), through the package's one printer
            emit(f"usage: {DESIGN_USAGE}")
            emit(f"{ap.prog} design: error: {err}")
            return EXIT_USAGE
        a = ap.parse_args(["design"] + kit)
        a.upstream = upstream
        return cmd_design(a)
    if "--" in argv:
        _err(f"unexpected arguments after --: {argv[argv.index('--') + 1:]} (only design takes upstream's tokens)")
        return EXIT_USAGE
    a = ap.parse_args(argv)
    if a.command is None:
        ap.print_help()
        return EXIT_USAGE
    return {"design": cmd_design, "check": cmd_check, "warm": cmd_warm}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
