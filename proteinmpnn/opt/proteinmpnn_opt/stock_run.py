"""The stock route (mode ``off``): the upstream command line, unchanged, in a clean subprocess — the tree's only stock caller.

Base variants: ``python $MPNN_DIR/helper_scripts/parse_multiple_chains.py --input_path <pdbs> --output_path <out>/parsed.jsonl`` (when
the input is a directory; the entries keep the helper's own order), then ``python $MPNN_DIR/protein_mpnn_run.py --jsonl_path ...
[--chain_id_jsonl <the caller's>] --out_folder <out> --path_to_model_weights $MPNN_DIR/<variant>_model_weights <stock options>
[--fixed_positions_jsonl F]`` (``--chain_id_jsonl`` only when the caller gives one: without it upstream's own default runs — every chain of each
entry designed, none fixed; inputs.py) — upstream's own script where it lives, its module, helper
scripts, ``.git`` (the .fa header's git_hash) and weights the checkout's own; ``<stock options>`` are the pass's own, verbatim (settings.parse /
stock_argv; none given = upstream's defaults); a pass that names its weights by upstream's own selectors (``--path_to_model_weights`` /
``--use_soluble_model``) gets no ``--path_to_model_weights`` from the kit; an ``--input`` that is one PDB file is upstream's ``--pdb_path`` (no
parse step, no chain assignment file: protein_mpnn_run.py designs every chain, or its ``--pdb_path_chains``).
One CMD line (report.log_cmd) precedes each process; the child's streams are the caller's; the scratch copy is removed after the pass.

Clean subprocess (``clean_env``): the package's own variables and every variable under the must-be-absent prefixes of stock/PINS.json
(``MPNN_DIR`` excepted: it names the checkout) are removed; PYTHONPATH entries under the tree's opt/ or a staged kit are removed;
PYTHONDONTWRITEBYTECODE=1; PYTHONSAFEPATH is removed (upstream's scripts import their siblings through the script directory, which that
interpreter switch would drop) and recorded under ``removed``. The proof (``env_proof``) is taken in a child of that environment — the core's clean-process
proof (opt_core.stock_proof) plus what is present and what is on sys.path — written to ``<out>/stock_env_proof.json`` and printed as one ``ENV-CLEAN`` line;
the line lists, never judges; a proof child that cannot report is a StockError (the pass does not start).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

from . import inputs as _inputs
from . import report as _report
from . import settings as _settings
from . import stack
from .modes import WORKER_DIR, PARSER_DIR

PROOF_FILE = "stock_env_proof.json"
KIT_DIR_MARKERS = (WORKER_DIR, PARSER_DIR)                                   # the kit directory names (modes.py), matched anywhere in a path entry


class StockError(RuntimeError):
    """The stock route could not run (no MPNN_DIR, no weights, no input)."""


def _is_kit_path(p: str) -> bool:
    ap = os.path.abspath(p) if p else ""
    opt = os.path.join(stack.tree_home(), "opt")
    return bool(p) and (ap.startswith(opt + os.sep) or ap == opt or any(m in ap for m in KIT_DIR_MARKERS))


def clean_env(base: Optional[dict] = None) -> Tuple[dict, dict]:
    """(env, removed): the caller's environment with the package variables, the must-be-absent names and kit PYTHONPATH entries removed."""
    base = dict(os.environ if base is None else base)
    p = stack.pins()
    prefixes, keep = tuple(p["must_be_absent_prefixes"]), set(p["must_be_absent_except"])
    removed: Dict[str, List[str]] = {"variables": [], "pythonpath": []}
    env = {}
    for k, v in base.items():
        if k in stack.PACKAGE_ENV or k in stack.INTERPRETER_ENV or (k.startswith(prefixes) and k not in keep):
            removed["variables"].append(k)
            continue
        env[k] = v
    pp = [e for e in env.get("PYTHONPATH", "").split(os.pathsep) if e]
    kept = [e for e in pp if not _is_kit_path(e)]
    removed["pythonpath"] = [e for e in pp if _is_kit_path(e)]
    if kept:
        env["PYTHONPATH"] = os.pathsep.join(kept)
    else:
        env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env, removed


def core_root() -> str:
    """The directory that holds the ``opt_core`` package this process imports (handed to the proof child explicitly: the proof must not depend on how the
    clean environment happens to reach the core)."""
    import opt_core
    return os.path.dirname(os.path.dirname(os.path.abspath(opt_core.__file__)))


def env_proof(env: dict, python: str = sys.executable) -> dict:
    """Run a child in ``env`` and record what it sees: the core's clean-process proof (opt_core.stock_proof.env_proof: the tree's directories on
    sys.path — the carried ``opt/forward`` and this package's —, kit or core modules loaded, an armed autoload finder, a kit sitecustomize, torch
    loaded; ``ok`` is its verdict over those), plus the variables under the prefixes (``MPNN_DIR`` excepted), PYTHONPATH and sys.path as the clean
    environment gives them (the core's directory is put on the child's path for the one import and taken off before the proof reads sys.path).
    A child that cannot report raises StockError (``child_failed`` names why): the stock route never proceeds on an unproven environment."""
    p = stack.pins()
    prefixes, keep = list(p["must_be_absent_prefixes"]), list(p["must_be_absent_except"])
    root = core_root()
    code = ("import json, os, sys; pre = %r; keep = %r; core = %r; sys.path.insert(0, core); from opt_core.stock_proof import env_proof; sys.path.remove(core); "
            "mods = {n: m for n, m in list(sys.modules.items()) if getattr(m, '__file__', None) or n.startswith('proteinmpnn_opt')}; "   # by file: built-ins have none
            "proof = env_proof(env_absent=(), kit_dirs=%r, module_prefixes=('proteinmpnn_opt',), modules=mods); "
            "proof.update({'present': sorted(k for k in os.environ if k.startswith(tuple(pre)) and k not in keep), "
            "'pythonpath': os.environ.get('PYTHONPATH'), 'sys_path': sys.path, 'python': sys.executable}); "
            "print(json.dumps(proof, default=str))") % (prefixes, keep, root, [stack.kit_home(), os.path.dirname(os.path.abspath(__file__))])   # the carried opt/forward, this package
    try:
        out = subprocess.run([python, "-c", code], capture_output=True, text=True, env=env, timeout=120)
        failure = None if out.returncode == 0 else "rc %d: %s" % (out.returncode, out.stderr.strip()[-300:])
    except (OSError, subprocess.TimeoutExpired) as e:
        out, failure = None, str(e)[-300:]
    seen = None
    if failure is None:
        try:
            seen = json.loads(out.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            failure = "no report on stdout: %s" % out.stderr.strip()[-300:]
    if failure is not None:
        raise StockError("stock env proof: the proof child could not report (%s) — the stock route does not proceed on an unproven environment" % failure)
    seen["kit_dirs_on_path"] = sorted(set(seen.get("kit_dirs_on_path") or []) |
                                      {e for e in seen.get("sys_path", []) if _is_kit_path(e) and os.path.exists(os.path.abspath(e))})   # an entry that is no path holds no kit file
    seen["checked_prefixes"] = prefixes
    seen["excepted"] = keep
    seen["mpnn_dir"] = env.get(stack.ENV_MPNN_DIR)
    seen["core_root"] = root
    return seen


def _run(cmd: List[str], env: dict, cwd: Optional[str], echo: bool = True) -> Tuple[int, float]:
    """Launch the child with the caller's own streams (``echo``; silenced otherwise); (rc, wall_s)."""
    t0 = time.time()
    out = None if echo else subprocess.DEVNULL
    rc = subprocess.call(cmd, env=env, cwd=cwd, stdout=out, stderr=out)
    return rc, time.time() - t0

def run(variant: str, input_path: str, out_dir: str, stock_args: Optional[List[str]] = None, chain_id_jsonl: Optional[str] = None,
        fixed_positions_jsonl: Optional[str] = None, echo: bool = True, python: str = sys.executable) -> dict:
    """One stock pass. Returns the record: {rc, wall_s, commands, env_proof, n_inputs, parsed, assigned (the caller's --chain_id_jsonl, else None), parser, order}."""
    os.makedirs(out_dir, exist_ok=True)
    env, removed = clean_env()
    proof = env_proof(env, python)
    proof["removed"] = removed
    with open(os.path.join(out_dir, PROOF_FILE), "w", encoding="utf-8") as fh:
        json.dump(proof, fh, indent=1)
    print(_report.env_clean_line(proof), file=sys.stderr, flush=True)
    rec: dict = {"route": "stock", "variant": variant, "commands": [], "env_proof": proof, "rc": None, "wall_s": 0.0, "n_inputs": None}
    pairs = _settings.parse(stock_args, variant)
    knobs = _settings.stock_argv(pairs)                                      # the stock command line's own options, verbatim (none the package supplies, no lever)
    mdir = stack.mpnn_dir()
    if not mdir or not os.path.isfile(os.path.join(mdir, "protein_mpnn_run.py")):
        raise StockError(f"{stack.ENV_MPNN_DIR}={mdir!r} does not hold protein_mpnn_run.py (set it to your ProteinMPNN clone: README 'Variables')")
    kind = _inputs.classify_base(input_path)
    rec["order"] = "given"                                                # a parsed jsonl or one PDB: the caller's order (a directory: the parse helper's, below)
    if kind == "pdb":                                                    # one PDB file: upstream's own --pdb_path (its parse_PDB; every chain designed, or --pdb_path_chains)
        rec["n_inputs"] = 1
        cmd = [python, os.path.join(mdir, "protein_mpnn_run.py"), "--pdb_path", os.path.abspath(input_path), "--out_folder", os.path.abspath(out_dir)]
        if chain_id_jsonl:
            cmd += ["--chain_id_jsonl", os.path.abspath(chain_id_jsonl)]
    else:
        parsed = os.path.join(out_dir, _inputs.PARSED)
        if kind == "dir":
            cmd = [python, os.path.join(mdir, "helper_scripts", "parse_multiple_chains.py"), "--input_path", os.path.abspath(input_path), "--output_path", parsed]
            _report.log_cmd("stock", variant, cmd)                       # the CMD line: the parse step's argv, before it starts
            rc, w = _run(cmd, env, None, echo)
            rec["commands"].append({"step": "parse", "cmd": cmd, "rc": rc, "wall_s": w})
            rec["wall_s"] += w
            rec["parser"] = "stock"                                       # upstream's helper parsed the directory
            rec["order"] = "parse helper (file-system order)"
            if rc != 0:
                rec["rc"] = rc
                return rec
        else:
            parsed = os.path.abspath(input_path)
        rec["parsed"] = parsed
        rec["n_inputs"] = _inputs.parse_count(parsed)
        rec["assigned"] = os.path.abspath(chain_id_jsonl) if chain_id_jsonl else None   # the caller's --chain_id_jsonl verbatim; none given: no chain flag on the line (upstream's default: every chain designed, none fixed)
        cmd = [python, os.path.join(mdir, "protein_mpnn_run.py"), "--jsonl_path", parsed] + (["--chain_id_jsonl", rec["assigned"]] if rec["assigned"] else []) + ["--out_folder", os.path.abspath(out_dir)]
    if not _settings.weights_given(pairs):
        cmd += ["--path_to_model_weights", os.path.join(mdir, f"{variant}_model_weights")]
    if fixed_positions_jsonl:
        cmd += ["--fixed_positions_jsonl", os.path.abspath(fixed_positions_jsonl)]
    cmd += knobs
    _report.log_cmd("stock", variant, cmd)                                   # the CMD line: the child's whole argv, before it starts
    rc, w = _run(cmd, env, None, echo)
    rec["commands"].append({"step": "design", "cmd": cmd, "cwd": None, "rc": rc, "wall_s": w})
    rec["wall_s"] += w
    rec["rc"] = rc
    return rec
