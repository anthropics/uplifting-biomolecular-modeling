"""python -m rosettafold3_opt {pred,check,warm,install} [--mode fast|exact|big|off] ...

A thin command layer over the add-on kits and the upstream CLI. It never re-implements a lever or a mode:

* ``pred``    — one ``rf3 fold`` per seed from an input file (inputs.py: the upstream JSON) with the caller's own ``key=value`` overrides (settings.py), on the
               interpreter the mode names with the mode's row exported (fold.py). ``--mode off``: the
               stock caller (stock_fold.py) on the pristine interpreter in a clean subprocess that proves its environment and its
               tree by sha.
* ``check``   — the activation line for a mode without applying anything: the tree state of every interpreter, the row, the
               levers, the pins, the GPU, the kernel key, and whether ``cuequivariance_torch`` imports on each interpreter (upstream runs its
               plain triangle kernels silently when it does not: on a GPU machine ``check`` refuses); ``--json`` prints the full report.
* ``warm``    — one public-input prediction through ``pred`` so the caches are filled (warm.py); ``--mode off`` is refused.
* ``install`` — the pinned ``opt_core`` installed into the patched interpreter and the add-on's ``install.sh`` applied once into it;
               both tree states asserted (install.py);
               ``--no-addon`` does everything but that (the patched interpreter's tree left as found, said so on one line).

Exit codes: 0 ok | 1 check/warm/pred failed | 2 usage | 3 levers not active (reason printed) | otherwise the child's exit code.
The mode is ``--mode`` when given, else ``ROSETTAFOLD3_OPT`` from the environment, else the package default (``modes.DEFAULT_MODE``,
``fast``); a ``--mode`` that disagrees with a set ``ROSETTAFOLD3_OPT`` is refused.
Each kit mode applies its FPF arm in-process (stack.py): the FPF APPLIED line and the exit tally are part of every kit-mode run's record.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from . import report
from .modes import DEFAULT_MODE, ENV, MODES, check_mode

PROG = "rosettafold3-opt"
from ._core import EXIT_NOT_ACTIVE                       # 3: the one refusal code (the pin gate's, the producer check's, every NOT ACTIVE line's)

EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
USAGE = f"""usage: {PROG} <command> [options]
  pred     --input <json|cif|dir> --out_dir <dir> [--mode fast|exact|big|off] [--n_gpu P] [--seeds S[,S...]] [--ckpt <file>] [key=value ...]
           [--log <file>] [--allow-partial]
  check    [--mode fast|exact|big|off] [--ckpt <file>] [--n_gpu P] [--json]
  warm     [--mode fast|exact|big] [--ckpt <file>] [--out <dir>] [--allow-partial]
  install  [--make-venv] [--no-addon] [--stock-python <python>] [--opt-python <python>] [--json]
modes: {'|'.join(MODES)} (default {DEFAULT_MODE}; or {ENV} in the environment)
key=value: rf3 fold's own hydra overrides, appended verbatim before seed=<s> (upstream keys and defaults: n_recycles=10 diffusion_batch_size=5 num_steps=50 early_stopping_plddt_threshold=0.5; settings.py)
"""


class CliError(Exception):
    def __init__(self, msg: str, code: int = EXIT_USAGE):
        super().__init__(msg)
        self.code = code


def mode_of(arg: Optional[str]) -> str:
    """``--mode`` against ROSETTAFOLD3_OPT: a disagreement is refused."""
    env = (os.environ.get(ENV) or "").strip().lower() or None
    if arg:
        try:
            m = check_mode(arg)
        except ValueError as e:
            raise CliError(str(e)) from e
        if env and env != m:
            raise CliError(f"--mode {m} disagrees with {ENV}={env}")
        return m
    if env:
        try:
            return check_mode(env)
        except ValueError as e:
            raise CliError(str(e)) from e
    return DEFAULT_MODE


def _seeds(s: Optional[str]) -> List[Optional[int]]:
    """``--seeds S[,S…]`` → one fold per seed; absent → one unseeded fold (no ``seed=`` token: rf3 fold's own default)."""
    if not s:
        return [None]
    try:
        return [int(x) for x in s.split(",") if x.strip()]
    except ValueError as e:
        raise CliError(f"--seeds must be integers: {s!r}") from e


def _ckpt(a) -> Optional[str]:
    from . import stack
    return a.ckpt or stack.checkpoint_path()


# ----------------------------------------------------------------------------------------------------------------- pred
def _allow_partial_gate(mode: str, allow_partial: bool) -> None:
    """``--allow-partial`` is the big line's opt-out: refused by name (usage) under the other modes, which have no partial units."""
    from . import stack as _stack
    if allow_partial and mode != _stack.BIG_MODE:
        raise CliError(f"--allow-partial is the big line's partial-unit opt-out; --mode {mode} has no partial units "
                       f"({'stock runs no lever' if mode == 'off' else 'a lever that does not engage there fails the run; --mode off runs stock'})")


def cmd_pred(argv: list[str]) -> int:
    from . import fold, settings as _settings
    p = argparse.ArgumentParser(prog=f"{PROG} pred", allow_abbrev=False)
    p.add_argument("--input", required=True, help="rf3 fold's inputs= verbatim: a JSON of items, a CIF-like structure file, a directory of either, or a comma list")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--mode", default=None)
    p.add_argument("--seeds", default=None, help="S[,S...]: one fold per seed into <out_dir>/seed-<S>/ (seed=<S>); absent: one fold into <out_dir>/ with rf3 fold's own default (unseeded)")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--n_gpu", type=int, default=1, help="the memory mode's resource axis: P GPUs, the pair stack row-sharded over them (big only; default 1)")
    p.add_argument("--allow-partial", action="store_true", help="big: a lever of the line that ran the stock path on part of its units (a partial unit) is recorded and the run proceeds; without it such a run exits 3")
    p.add_argument("--log", default=None)
    p.add_argument("overrides", nargs="*", metavar="key=value", help="rf3 fold hydra overrides, passed through verbatim (upstream's keys and defaults)")
    a = p.parse_intermixed_args(argv)
    mode = mode_of(a.mode)
    _allow_partial_gate(mode, a.allow_partial)
    try:
        ov = _settings.check(a.overrides)
        if a.seeds:
            _settings.refuse_double_seed(ov)                    # --seeds with a seed= token names the seed twice
    except ValueError as e:
        raise CliError(str(e)) from e
    if "," not in a.input and not a.input.startswith("[") and not os.path.exists(a.input):
        raise CliError(f"--input {a.input}: no such file or directory")      # a plain path that is not there (a comma list / hydra list is rf3's to read)
    os.makedirs(a.out_dir, exist_ok=True)
    try:
        rec = fold.run(mode, inputs=a.input, out_dir=a.out_dir, ckpt=_ckpt(a), overrides=ov, seeds=_seeds(a.seeds),
                       log_path=a.log or os.path.join(a.out_dir, "pred.log"),
                       n_gpu=a.n_gpu, allow_partial=a.allow_partial)
    except fold.NotActive:
        return EXIT_NOT_ACTIVE                                  # the NOT ACTIVE line was printed (the --n_gpu rules: opt_core.mem.ngpu's words)
    except (ValueError, RuntimeError) as e:
        raise CliError(str(e), EXIT_FAIL) from e
    print(fold.summary_line(rec), flush=True)
    if rec["status"] == "PASS":
        return EXIT_OK
    if rec.get("n_gpu_mismatch"):
        return EXIT_NOT_ACTIVE                                  # a fold that ran on fewer GPUs than --n_gpu asked for (the NOT ACTIVE n_gpu_mismatch line was printed)
    rcs = [r["rc"] for r in rec["runs"] if r["rc"] != 0]
    return rcs[0] if rcs and rcs[0] not in (0, EXIT_OK) else EXIT_FAIL


# ---------------------------------------------------------------------------------------------------------------- check
def cmd_check(argv: list[str]) -> int:
    from . import mem, stack, tree
    p = argparse.ArgumentParser(prog=f"{PROG} check", allow_abbrev=False)
    p.add_argument("--mode", default=None)
    p.add_argument("--n_gpu", type=int, default=1)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    mode = mode_of(a.mode)
    rep = stack.activate(mode, dry_run=True, n_gpu=a.n_gpu)
    ckpt = _ckpt(a)
    if ckpt and os.path.isfile(ckpt):                                                          # the checkpoint's digest against the pin, hashed AFRESH by check (the memo entry rewritten)
        from . import weights
        rep["checkpoint"] = weights.announce(ckpt, stack.pins(), verb="check")
    interps = {}
    try:
        lists = stack.tree_digests()
        for kind, py in (("stock", stack.stock_python()), ("opt", stack.opt_python())):
            try:
                ts = tree.state_of(py, lists)
                interps[kind] = {"python": py, "state": ts.state, "line": ts.line(), "files": ts.files, "pins": stack.pin_check(py),
                                 "cueq": stack.cueq_probe(py)}
            except Exception as e:
                interps[kind] = {"python": py, "state": "unavailable", "error": str(e)[-300:]}
    except Exception as e:
        interps["error"] = str(e)[-300:]
    rep["interpreters"] = interps
    used = interps.get("stock" if mode == "off" else "opt")                                   # the interpreter this mode folds on
    cueq = (used or {}).get("cueq") if isinstance(used, dict) else None
    cueq_missing = rep.get("gpu") is not None and cueq is not None and not cueq.get("ok")     # on a GPU machine upstream would fall back to its plain
    ok = True                                                                                 # triangle kernels silently: check refuses instead
    if mode == "off":
        st = interps.get("stock", {})
        ok = st.get("state") == "stock" and bool((st.get("pins") or {}).get("ok"))
        line = (f"{report.PREFIX} DRY-RUN mode=off stock_python={st.get('python')} {st.get('line') or st.get('error')} "
                f"pins={'ok' if (st.get('pins') or {}).get('ok') else 'FAIL'} {stack.cueq_word(st.get('cueq'))} gpu={report.gpu_label(rep.get('gpu'))}")
    else:
        ok = rep.get("reason") is None
        line = report.active_line(rep)
    print(line, flush=True)
    if cueq_missing:
        ok = False
        print(f"{report.PREFIX} check FAIL: cuequivariance_torch does not import on {used.get('python')} ({cueq.get('error')}): rf3 would run its plain "
              f"triangle attention / multiplicative update instead of the cuEquivariance kernels", flush=True)
    if mode != "off" and rep.get("mem"):
        print(mem.line(rep["mem"]["policy"], rep["mem"].get("fpf_tg_max")), flush=True)                                             # the memory policy of the dry run (stdout, beside the DRY-RUN line)
    for kind, st in interps.items():
        if isinstance(st, dict):
            pin = st.get("pins") or {}
            print(f"{report.PREFIX} interpreter {kind}: {st.get('python')} {st.get('line') or st.get('error')} pins={pin.get('route')}:{'ok' if pin.get('ok') else 'FAIL'} "
                  f"{stack.cueq_word(st.get('cueq'))}", flush=True)
    if a.json:
        print(json.dumps(rep, indent=1, default=str), flush=True)
    return EXIT_OK if ok else EXIT_NOT_ACTIVE


# ----------------------------------------------------------------------------------------------------------------- warm
def cmd_warm(argv: list[str]) -> int:
    from . import warm
    p = argparse.ArgumentParser(prog=f"{PROG} warm", allow_abbrev=False)
    p.add_argument("--mode", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--allow-partial", action="store_true", help="big: as pred --allow-partial")
    a = p.parse_args(argv)
    mode = mode_of(a.mode)
    _allow_partial_gate(mode, a.allow_partial)
    try:
        res = warm.run(mode, ckpt=_ckpt(a), out_dir=a.out, allow_partial=a.allow_partial)
    except (ValueError, RuntimeError) as e:
        raise CliError(str(e), EXIT_FAIL) from e
    print(warm.summary_line(res), flush=True)
    return EXIT_OK if res["status"] == "PASS" else EXIT_FAIL


# -------------------------------------------------------------------------------------------------------------- install
def cmd_install(argv: list[str]) -> int:
    from . import install
    p = argparse.ArgumentParser(prog=f"{PROG} install", allow_abbrev=False)
    p.add_argument("--make-venv", action="store_true", help="create opt/venv from the stock interpreter when it does not exist yet")
    p.add_argument("--no-addon", action="store_true", help="every step but the add-on: the patched interpreter's five rf3 files are left as found (a later install applies them)")
    p.add_argument("--stock-python", default=None)
    p.add_argument("--opt-python", default=None)
    p.add_argument("--log", default=None)
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    try:
        res = install.run(make_venv_flag=a.make_venv, addon=not a.no_addon, stock_python=a.stock_python, opt_python=a.opt_python, log_path=a.log)
    except (RuntimeError, ValueError, OSError) as e:
        raise CliError(str(e), EXIT_FAIL) from e
    print(install.summary_line(res), flush=True)
    if a.json:
        print(json.dumps(res, indent=1, default=str), flush=True)
    return EXIT_OK


# ----------------------------------------------------------------------------------------------------------------- entry
COMMANDS = {"pred": cmd_pred, "check": cmd_check, "warm": cmd_warm, "install": cmd_install}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    from . import _core
    try:
        _core.require_or_exit()                                      # every opt_core module this package loads is present, or no command resolves anything: the NOT ACTIVE line, exit 3
    except SystemExit as e:
        return int(e.code)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE, end="")
        return EXIT_OK if argv else EXIT_USAGE
    cmd = argv[0]
    fn = COMMANDS.get(cmd)
    if fn is None:
        print(f"{report.PREFIX} unknown command {cmd!r}\n{USAGE}", end="", file=sys.stderr)
        return EXIT_USAGE
    from . import stack
    undeclared = stack.undeclared_env()
    if undeclared:                                                   # a mistyped ROSETTAFOLD3_OPT_* name is refused, never ignored
        print(f"{report.PREFIX} undeclared ROSETTAFOLD3_OPT_* / ROSETTAFOLD3_BIG_* names in the environment (this tree reads {', '.join(stack.ENV_NAMES)}): "
              + ", ".join(f"{k}={v!r}" for k, v in undeclared.items()), file=sys.stderr)
        return EXIT_USAGE
    try:
        return fn(argv[1:])
    except CliError as e:
        print(f"{report.PREFIX} ERROR: {e}", file=sys.stderr, flush=True)
        return e.code
