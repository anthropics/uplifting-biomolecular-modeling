"""borzoi-opt — the command line: ``sad`` (the documented command, drop-in, in one mode) and ``check`` (dry run).
``python -m borzoi_opt ...`` is the same program; ``run.sh`` wraps it after sourcing a configuration.

    borzoi-opt sad [--mode exact|off] [--det 0|1] [--allow-partial] [--] <the stock borzoi_sad.py arguments>
    borzoi-opt check [--mode M] [--det 0|1] [--json]

``sad``: the mode is resolved and gated (modes.resolve), its line printed, then ONE subprocess runs the whole job — the kit's
``v17/borzoi_sad.py`` (exact: the composition in the environment — the kit's forward switch ``KIT_FWD=1`` on, no other switch (a
user's ``KIT_*`` name the kit reads is refused by name); ``BORZOI_OPT`` removed so the hook does not fire twice; ``KIT_STAMP_DIR`` pointed at a private scratch directory the
``sad`` process reads and removes) or the stock script through the proven clean caller (``stock_sad.py``: every ``KIT_*`` name and
``BORZOI_OPT`` stripped, kit directories dropped from PYTHONPATH, ``-s``; its proof record goes to the same private directory). The stock's
arguments pass through unchanged and unparsed: whatever the stock script accepts, every mode accepts (both of its invocation forms
included). Nothing is written beside the stock's outputs; what ran is on the printed lines.

After a kit run the stamp is judged (modes.applied): a lever of the mode that fell back or left no evidence in the kit's own record —
the forward's stock calls, a writer other than the mode's, no stamp at all — is ``partial`` (``NOT ACTIVE: partial activation — <detail>;
exit 3 (--allow-partial records and proceeds)``, report.partial_line): the exit is 3 unless ``--allow-partial``
(``BORZOI_OPT_ALLOW_PARTIAL=1`` in the environment) accepts the run as it ran (``PARTIAL allowed: <detail> (--allow-partial, recorded)``),
the exit then the job's own. The chunked post outside its declared option set, or a variant outside its shape class, is per-call STOCK
ROUTING — ``ROUTED post=stock:<reason>`` on one line, the exit the job's own. The post's thread pool sized from the OS core count because
the cores probe could not run is a named FALLBACK (``FALLBACK cores_probe=os_count (<reason>) pool=<n>``), outputs unaffected, the exit
the job's own. A declared precondition that stops the whole mode (none today) is ``gated`` (a ``GATED`` line) and refused like a partial.
The ``BORZOI_OPT=exact`` hook route ends at the exec of the kit entry: there the kit's own ``KIT_STAMP`` line on stdout is the record and
the exit is the job's — ``borzoi-opt sad`` is the gated form.

``--det 1`` (any mode) exports the reproducibility recipe into the job (modes.DET_RECIPE: numpy's CPU dispatch pinned to one
instruction set, one BLAS thread — byte-stable host arithmetic across hosts; ``--det 0``, the default, = production).

Exit codes: 0 = the job's own exit (0); the job's non-zero code as is; 2 = usage / a refused request (an unknown mode, ``--mode`` and
``BORZOI_OPT`` disagreeing, ``--det`` other than 0|1); 3 = NOT ACTIVE (the gate refused the mode: the kit absent or the stock not the
pinned one, a kit switch set in the environment; mode off: the stock caller's environment proof failed) or PARTIAL (a lever of the mode
fell back, by the kit's stamp) without ``--allow-partial``.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import List, Optional, Tuple

from . import kit, modes, report

PROG = "borzoi-opt"
EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_ACTIVE = 0, 1, 2, 3
USAGE = f"""usage: {PROG} <command> [--mode exact|off] ...

  sad     [--mode M] [--det 0|1] [--allow-partial] [--] <stock borzoi_sad.py arguments>
                                                                   the documented command in mode M (default: {modes.DEFAULT_MODE});
                                                                   --allow-partial: accept a run whose levers fell back (PARTIAL, by the kit's stamp) — recorded; without it exit 3
  check   [--mode M] [--det 0|1] [--json]                          dry run: resolve and gate M on this machine, apply nothing

  Modes: exact = the kit (the traced forward from the second call, the copy-free return, the LUT one-hot, the pipelined post with its exact writer — the stock's bytes; the default) |
         off = the stock script in a proven clean subprocess. BORZOI_OPT=<mode> is the same switch.
  --det 1: the reproducibility recipe exported into the job (numpy's CPU dispatch pinned to one instruction set, one BLAS thread); --det 0 (default): production.
"""
ALLOW_PARTIAL_ENV = "BORZOI_OPT_ALLOW_PARTIAL"                     # =1: the environment form of `sad --allow-partial` (beside BORZOI_OPT=<mode>)
STOCK_ENV_KEEP = ("BORZOI_HG38",)                                 # the stock's own data-path variable (borzoi_sad.py -f default) — never stripped
KIT_STAMP_FILE = "kit_stamp.json"                                 # the kit entry's exit record, written into KIT_STAMP_DIR (v17/borzoi_sad.py)
STOCK_PROOF_FILE = "stock_env_proof.json"                          # the stock caller's proof record (stock_sad.py --proof-json), same private directory
DET_VALUES = {"0": False, "1": True}


class CliError(Exception):
    def __init__(self, msg: str, code: int = EXIT_USAGE):
        super().__init__(msg)
        self.code = code


# ---- own options ------------------------------------------------------------------------------------------------------------
def split_own(argv: List[str], own_value: Tuple[str, ...] = ("--mode",), own_flag: Tuple[str, ...] = ()) -> Tuple[dict, List[str]]:
    """The package's own options are taken from the FRONT of the argument list (``--mode M`` / ``--mode=M`` / flags); ``--`` ends them;
    everything from the first non-own token on is the stock's, untouched."""
    own: dict = {}
    rest = list(argv)
    while rest:
        a = rest[0]
        if a == "--":
            rest.pop(0)
            break
        name, eq, val = a.partition("=")
        if name in own_value:
            rest.pop(0)
            if not eq:
                if not rest:
                    raise CliError(f"{name} requires a value")
                val = rest.pop(0)
            own[name.lstrip("-").replace("-", "_")] = val
        elif name in own_flag and not eq:
            rest.pop(0)
            own[name.lstrip("-").replace("-", "_")] = True
        else:
            break
    return own, rest


def resolve_mode(own_mode: Optional[str], environ=None) -> str:
    environ = os.environ if environ is None else environ
    env_mode = (environ.get(modes.ENV) or "").strip().lower() or None
    if own_mode is not None and env_mode is not None and own_mode.lower() != env_mode:
        raise CliError(f"--mode {own_mode} and {modes.ENV}={env_mode} disagree — one mode per run")
    return modes.check_mode(own_mode or env_mode or modes.DEFAULT_MODE)


def det_from(own: dict) -> bool:
    """``--det 0|1`` (default 0 = production numerics)."""
    v = str(own.get("det", "0")).strip()
    if v not in DET_VALUES:
        raise CliError(f"--det takes 0 or 1 (got {v!r})")
    return DET_VALUES[v]


def allow_partial_from(own: dict, environ=None) -> bool:
    """--allow-partial, or BORZOI_OPT_ALLOW_PARTIAL=1 in the environment."""
    environ = os.environ if environ is None else environ
    return bool(own.get("allow_partial")) or (environ.get(ALLOW_PARTIAL_ENV) or "").strip() == "1"


# ---- sad ------------------------------------------------------------------------------------------------------------------------
def kit_command(rep: dict, stock_args: List[str], stamp_dir: str, environ=None) -> Tuple[List[str], dict]:
    env = dict(os.environ if environ is None else environ)
    env.pop(modes.ENV, None)                                       # the hook must not fire again in the child (one ACTIVE line per run)
    env.update(rep["env"])                                          # the composition (KIT_FWD=1) and, under --det 1, the recipe
    env["KIT_STAMP_DIR"] = stamp_dir
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return [sys.executable, rep["entry"]] + list(stock_args), env


def stock_command(rep: dict, stock_args: List[str], stamp_dir: str, environ=None) -> Tuple[List[str], dict]:
    env = {k: v for k, v in (os.environ if environ is None else environ).items()
           if not (k.startswith(modes.KIT_ENV_PREFIX) and k not in STOCK_ENV_KEEP) and k != modes.ENV}
    kd = kit.kit_dir()
    if env.get("PYTHONPATH"):
        keep = [p for p in env["PYTHONPATH"].split(os.pathsep) if p and not os.path.realpath(p).startswith(os.path.realpath(kd))]
        if keep:
            env["PYTHONPATH"] = os.pathsep.join(keep)
        else:
            env.pop("PYTHONPATH")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.update(rep.get("env") or {})                                # under --det 1: the recipe (the stock arm's only environment gain)
    st = rep["stock"]
    cmd = [sys.executable, "-s", "-m", "borzoi_opt.stock_sad", "--proof-json", os.path.join(stamp_dir, STOCK_PROOF_FILE),
           "--stock-entry", st["entry"], "--stock-sha256", st["sha256"] or "", "--kit-dirs", kd, "--"] + list(stock_args)
    return cmd, env


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def cmd_sad(argv: List[str]) -> int:
    own, stock_args = split_own(argv, ("--mode", "--det"), ("--allow-partial",))
    mode = resolve_mode(own.get("mode"))
    det = det_from(own)
    allow_partial = allow_partial_from(own)
    rep = modes.resolve(mode, route="cli", det=det)
    report.log(report.activation_line(rep))
    if not rep["ok"]:
        return EXIT_NOT_ACTIVE
    stamp_dir = tempfile.mkdtemp(prefix="borzoi_opt_")             # private: the kit's stamp / the stock caller's proof; nothing lands beside the outputs
    try:
        cmd, env = stock_command(rep, stock_args, stamp_dir) if mode == "off" else kit_command(rep, stock_args, stamp_dir)
        rc = subprocess.call(cmd, env=env)
        stamp = _read_json(os.path.join(stamp_dir, KIT_STAMP_FILE)) if mode != "off" else None
        proof = _read_json(os.path.join(stamp_dir, STOCK_PROOF_FILE)) if mode == "off" else None
    finally:
        shutil.rmtree(stamp_dir, ignore_errors=True)
    if mode == "off":
        exit_code = rc
        if proof is not None and not proof.get("ok"):               # a failed environment proof never reads as a stock run
            exit_code = rc or EXIT_NOT_ACTIVE
        report.log(report.exit_line(mode, rc, proof=proof))
        return exit_code
    rep.update(modes.applied(rep, stamp)); rep["allow_partial"] = allow_partial
    partial = list(rep.get("partial") or [])
    gated_levers = list(rep.get("gated_levers") or [])
    exit_code = rc
    if rc == 0 and (partial or gated_levers) and not allow_partial:   # a degraded run never reads as success on rc: a lever that fell back, or a declared precondition taken (gated) — refused by name
        exit_code = EXIT_NOT_ACTIVE
    for g in rep.get("gated") or []:
        report.log(f"{report.PREFIX} GATED {g}")
    for r_ in rep.get("routed") or []:                              # per-call stock routing (post=stock:<reason>): one named line each, the exit the job's own
        report.log(report.routed_line(r_))
    for lv, detail in rep.get("fallbacks") or []:                   # a tuned heuristic at its broad default (the cores probe): one named line, the exit the job's own
        report.log(report.fallback_line(lv, detail))
    if partial or gated_levers:
        report.log(report.partial_line(partial + gated_levers, list(rep.get("partial_reasons") or []) + list(rep.get("gated") or []), allow_partial, EXIT_NOT_ACTIVE))
    report.log(report.exit_line(mode, rc, stamp, partial=partial, allow_partial=allow_partial, gated=rep.get("gated"), routed=rep.get("routed")))
    return exit_code


# ---- check ----------------------------------------------------------------------------------------------------------------------
def cmd_check(argv: List[str]) -> int:
    own, rest = split_own(argv, ("--mode", "--det"), ("--json",))
    if rest:
        raise CliError(f"check takes no stock arguments (got {rest})")
    mode = resolve_mode(own.get("mode"))
    rep = modes.resolve(mode, route="cli", dry_run=True, det=det_from(own))
    report.log(report.activation_line(rep))
    if own.get("json"):
        print(json.dumps(rep, indent=1, default=str))
    else:
        from . import registry
        for name, lv in registry.by_mode(mode).items():
            print(f"  lever {name}: class={lv.cls} tier={lv.tier} ({lv.file})")
    return EXIT_OK if rep["ok"] else EXIT_NOT_ACTIVE


# ---- main -----------------------------------------------------------------------------------------------------------------------
COMMANDS = {"sad": cmd_sad, "check": cmd_check}


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return EXIT_OK if argv else EXIT_USAGE
    cmd = argv[0]
    if cmd not in COMMANDS:
        print(f"{PROG}: unknown command {cmd!r}\n{USAGE}", file=sys.stderr)
        return EXIT_USAGE
    try:
        return COMMANDS[cmd](argv[1:])
    except CliError as e:
        report.log(f"{report.PREFIX} {'NOT ACTIVE' if e.code == EXIT_NOT_ACTIVE else 'usage'}: {e}")
        return e.code
    except modes.ActivationError as e:
        report.log(f"{report.PREFIX} NOT ACTIVE: {e}")
        return EXIT_NOT_ACTIVE
