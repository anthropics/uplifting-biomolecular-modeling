"""The tree's one stock caller: the upstream CLI ``rf3 fold`` on the PRISTINE interpreter in a clean subprocess.

Stock is upstream at the pin (``stock/PINS.json``) with nothing from this tree on the path: the interpreter is ``stock/venv``
(``ROSETTAFOLD3_OPT_STOCK_PYTHON``), which ``install.sh`` never touched, proved by sha before every run (tree.py state ``stock``); its environment carries no variable of the kits' namespaces nor of this package
(``stock/PINS.json`` "stock_environment"), proved and printed as

    [rosettafold3-opt stock] ENV-CLEAN ok: absent=<prefixes> files=stock(5/5) kit_dirs=none
    [rosettafold3-opt stock] TREE tree=stock(5/5) site-packages=<sp>

``kit_dirs=none`` = none of this tree's kit directories (``opt/forward``) nor the package directory on the stock
interpreter's ``sys.path``, and ``rosettafold3_opt`` not importable there. The command is upstream's own CLI, one process per (input, seed) — or one unseeded process when ``pred`` is given no ``--seeds``:

    <stock python> -m rf3.cli fold inputs=<json> out_dir=<dir> ckpt_path=<ckpt> <the caller's key=value overrides> seed=<s>

(``models/rf3/src/rf3/cli.py:85-86`` runs the same typer app the ``rf3`` console script runs). Outputs are the CLI's own file set
under ``<out_dir>/seed-<s>/`` (``<out_dir>/`` itself, no ``seed=``, for the unseeded fold). Nothing here is a lever, a mode, or a kit file.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple, Sequence

from . import _core
from . import report as _report
from . import settings as _settings
from . import tree as _tree

_stock_proof = _core.load("stock_proof")

PREFIX = _report.PREFIX_STOCK


def must_be_absent(pins: dict) -> List[str]:
    return list(pins["stock_environment"]["must_be_absent_prefixes"])


def clean_env(pins: dict, environ=None) -> Tuple[Dict[str, str], List[str]]:
    """The child's environment: every name under the kits' must-be-absent prefixes stripped; everything else as the caller's."""
    prefixes = must_be_absent(pins)
    environ, stripped = _stock_proof.strip_env(os.environ if environ is None else environ, prefixes)
    environ["PYTHONDONTWRITEBYTECODE"] = environ.get("PYTHONDONTWRITEBYTECODE", "1")
    return environ, stripped


PROBE_SENTINEL = "ROSETTAFOLD3_OPT_PATH_PROBE "


def kit_dirs_on_path(python: str, opt_root: str, env: Dict[str, str]) -> List[str]:
    """Directories of this tree's opt/ on the stock interpreter's sys.path, plus the package if importable there (must be empty).
    The probe's verdict is the one stdout line prefixed ``PROBE_SENTINEL`` (a JSON list); anything else the interpreter prints at start-up
    (a site hook's banner) is not a path."""
    code = ("import sys, os, json, importlib.util; root = os.path.realpath(sys.argv[1]); "
            "kits = [os.path.join(root, d) for d in ('forward', 'rosettafold3_opt')]; "
            "bad = [p for p in sys.path if p and any(os.path.realpath(p).startswith(k) for k in kits)]; "
            "s = importlib.util.find_spec('rosettafold3_opt'); "
            f"print({PROBE_SENTINEL!r} + json.dumps(bad + (['rosettafold3_opt@' + str(s.origin)] if s else [])))")
    r = subprocess.run([python, "-c", code, opt_root], capture_output=True, text=True, env=env, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"stock interpreter {python} failed the path probe: {(r.stderr or r.stdout)[-400:]}")
    verdicts = [ln[len(PROBE_SENTINEL):] for ln in r.stdout.splitlines() if ln.startswith(PROBE_SENTINEL)]
    if len(verdicts) != 1:
        raise RuntimeError(f"stock interpreter {python} gave no path-probe verdict: {r.stdout[-400:]!r}")
    return [str(x) for x in json.loads(verdicts[0])]


def proof(python: str, pins: dict, lists: _tree.Digests, opt_root: str) -> Tuple[Dict[str, str], _tree.TreeState, str]:
    """The two proof lines (printed) and the child environment; raises on any failure."""
    env, stripped = clean_env(pins)
    ts = _tree.state_of(python, lists)
    _tree.expect(ts, "stock")
    kd = kit_dirs_on_path(python, opt_root, env)
    if kd:
        raise RuntimeError(f"stock interpreter {python} sees this tree's kit directories or package: {kd}")
    prefixes = must_be_absent(pins)
    present = [k for k in env if any(k.startswith(p) for p in prefixes)]
    if present:
        raise RuntimeError(f"stock environment still carries {present}")
    line1 = f"{PREFIX} ENV-CLEAN ok: absent={','.join(prefixes)} files={ts.rf3_count} kit_dirs=none" + (f" stripped={','.join(stripped)}" if stripped else "")
    line2 = f"{PREFIX} TREE {ts.line()}"
    print(line1, file=sys.stderr, flush=True)
    print(line2, file=sys.stderr, flush=True)
    return env, ts, line1


def command(python: str, inputs: str, out_dir: str, ckpt: str, overrides: Optional[Sequence[str]], seed: Optional[int]) -> List[str]:
    """The one ``rf3 fold`` argv of every route: the interpreter, the three inputs of a run, the caller's overrides verbatim, ``seed=<s>``."""
    ov = _settings.overrides(overrides, seed)
    return [python, "-m", "rf3.cli", "fold", f"inputs={inputs}", f"out_dir={out_dir}", f"ckpt_path={ckpt}"] + ov


def _seed_dir(out_dir: str, seed) -> str:
    """``<out_dir>/seed-<s>/`` for a seed, ``<out_dir>/`` for the unseeded fold (no ``seed=`` token: upstream's ``seed: null``)."""
    return os.path.join(out_dir, f"seed-{seed}") if seed is not None else out_dir


def run(python: str, pins: dict, lists: _tree.Digests, opt_root: str, *, inputs: str, out_dir: str, ckpt: str, overrides: Sequence[str] = (),
        seeds: Sequence[Optional[int]] = (), log_path: Optional[str] = None) -> dict:
    """One ``rf3 fold`` per seed into ``<out_dir>/seed-<s>/`` (the one unseeded fold, ``seeds=[None]``, into ``<out_dir>/`` without ``seed=``);
    returns {status, runs: [{seed, cmd, rc, wall_s, out_dir}], proof, tree}."""
    env, ts, line = proof(python, pins, lists, opt_root)
    runs = []
    os.makedirs(out_dir, exist_ok=True)
    for s in seeds:
        od = _seed_dir(out_dir, s)
        cmd = command(python, inputs, od, ckpt, overrides, s)
        t0 = time.time()
        with (open(log_path, "a", encoding="utf-8") if log_path else open(os.devnull, "w")) as lf:
            lf.write("$ " + " ".join(cmd) + "\n")
            r = subprocess.run(cmd, env=env, stdout=lf if log_path else None, stderr=subprocess.STDOUT if log_path else None)
        runs.append({"seed": s, "cmd": cmd, "rc": r.returncode, "wall_s": round(time.time() - t0, 1), "out_dir": od})
        if r.returncode != 0:
            break
    ok = all(x["rc"] == 0 for x in runs) and len(runs) == len(seeds)
    return {"status": "PASS" if ok else "FAIL", "runs": runs, "proof": line, "tree": ts.line(), "tree_state": ts.state, "python": python}
