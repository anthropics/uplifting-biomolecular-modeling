"""The stock caller — mode ``off``: the upstream CLIs, nothing from a kit on the path.

Upstream ProGen2 IS a command line (`sample.py`, `likelihood.py`, run from the checkout with `./checkpoints/<name>` and `tokenizer.json`
in the cwd). One fresh process per call is the stock's only form: the package spawns
``python -s -m progen2_opt.stock_cli --script sample.py|likelihood.py -- <the script's own arguments>`` in the stock run directory
(stack.workdir) with the package's variables stripped from the environment (``clean_env``: the `PROGEN2_` names but the data paths); the child
proves its environment before anything else is imported (``env_proof``: no forbidden variable, no kit module loaded, no kit directory on
sys.path), prints ``[progen2-opt stock] ENV-CLEAN ok ...`` and then runs the stock script's own bytes as ``__main__`` (``runpy.run_path``: no
edit, no wrapper, the same argv the README documents). Two calling forms: ``run_passthrough`` (one call: the child's stdout / stderr are this
process's own, its exit code returned) and ``run_item`` (an item of a multi-item job: the streams captured, the block cut from the stdout —
outputs.block_of_stdout — and the stock's own ``loading parameters took <s>`` read for the ready line).

Run standalone (by any caller), from the stock run directory:
    python -s -m progen2_opt.stock_cli --script likelihood.py --env-absent PROGEN2_ -- --model progen2-small --context 1MK...

The interpreter of the child is ``$PROGEN2_PYTHON`` when set (STOCK.md 'Variables'), else the calling one (stack.python); one wall budget
per child (stack.PROCESS_TIMEOUT_S): a child past it is killed and its exit reported.
"""
from __future__ import annotations

import argparse
import os
import runpy
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

from . import outputs, stack

PREFIX = "[progen2-opt stock]"
KIT_MODULE_PREFIXES = ("engines.progen2", "progen2_decode", "sampler_exact", "oneread_loader", "kit_t1", "compare.bit_equal")
KIT_PATH_MARKERS = ("progen2_decode.py", os.path.join("engines", "progen2", "kits"))            # a sys.path entry holding these is a kit directory
LOAD_RE = "loading parameters took "


def env_absent_spec(pins: Optional[dict]) -> List[str]:
    """Entries ending in '_' are prefixes, others exact names: stock/PINS.json ``must_be_absent`` (the package's variable family; the kits read none of their own)."""
    env_rules = (pins or {}).get("stock_environment") or {}
    return list(env_rules.get("must_be_absent_prefixes") or (pins or {}).get("must_be_absent") or ("PROGEN2_",))


def env_allowed(pins: Optional[dict], data_env: Tuple[str, ...]) -> Tuple[str, ...]:
    """The data-path names a prefix rule must not flag: stock/PINS.json ``stock_environment.allowed_exceptions`` + the package's own."""
    env_rules = (pins or {}).get("stock_environment") or {}
    return tuple(dict.fromkeys(list(env_rules.get("allowed_exceptions") or []) + list(data_env)))


def _forbidden(environ, spec) -> list:
    hits = []
    for s in spec:
        for k in environ:
            if (k.startswith(s) if s.endswith("_") else k == s):
                hits.append(k)
    return sorted(set(hits))


CLEAN_RECORD: Dict[str, list] = {"stripped": [], "kept": []}       # what the last clean_env() removed / kept (the ENV-CLEAN line of the child names them)


def clean_env(environ: Dict[str, str], spec: List[str], allowed: Tuple[str, ...]) -> Dict[str, str]:
    """The stock process's environment: the caller's with every forbidden name stripped (the data-path names kept) and kit directories
    removed from PYTHONPATH — nothing added."""
    env = {k: v for k, v in environ.items() if k in allowed or not any((k.startswith(s) if s.endswith("_") else k == s) for s in spec)}
    CLEAN_RECORD["stripped"] = sorted(k for k in environ if k not in env)
    CLEAN_RECORD["kept"] = sorted(k for k in env if k in allowed)
    pp = [p for p in (env.get("PYTHONPATH") or "").split(os.pathsep) if p and not _is_kit_dir(p)]
    if pp:
        env["PYTHONPATH"] = os.pathsep.join(pp)
    else:
        env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _is_kit_dir(p: str) -> bool:
    return any(os.path.exists(os.path.join(p, m)) for m in KIT_PATH_MARKERS)


def env_proof(env_absent: List[str], environ=None, modules=None, path=None, allowed: Tuple[str, ...] = ()) -> dict:
    """The clean-environment proof of this process (``allowed``: the data-path names a prefix rule must not flag). Raises RuntimeError
    listing every violation."""
    environ = os.environ if environ is None else environ
    modules = sys.modules if modules is None else modules
    path = sys.path if path is None else path
    hits = [h for h in _forbidden(environ, env_absent) if h not in allowed]
    kit_mods = sorted(m for m in modules if m.startswith(KIT_MODULE_PREFIXES))
    kit_dirs = sorted(p for p in path if p and _is_kit_dir(p))
    torch_loaded = "torch" in modules
    proof = {"env_absent": list(env_absent), "env_allowed": list(allowed), "forbidden_present": hits, "kit_modules": kit_mods, "kit_dirs": kit_dirs,
             "torch_loaded_before_proof": torch_loaded, "no_user_site": bool(sys.flags.no_user_site), "cwd": os.getcwd(), "argv": list(sys.argv)}
    bad = []
    if hits:
        bad.append(f"forbidden variables present: {hits}")
    if kit_mods:
        bad.append(f"kit modules loaded: {kit_mods}")
    if kit_dirs:
        bad.append(f"kit directories on sys.path: {kit_dirs}")
    if bad:
        raise RuntimeError("stock process is not clean: " + "; ".join(bad))
    return proof


def proof_line(proof: dict) -> str:
    """The ENV-CLEAN line (the contract template): stripped = the names removed from the caller's environment, kept = the data-path names kept."""
    stripped = ",".join(proof.get("stripped") or []) or "none"
    kept = ",".join(proof.get("kept") or proof.get("env_allowed") or []) or "none"
    ENV_CLEAN_FMT = f"[progen2-opt stock] ENV-CLEAN ok stripped={stripped} kept={kept}"
    return ENV_CLEAN_FMT + f" absent={','.join(proof['env_absent'])} kit_modules=none kit_dirs=none no_user_site={proof['no_user_site']} cwd={proof['cwd']}"


def proc_line(route: str, upstream_name: str, pid: int, cwd: str) -> str:
    size = upstream_name[len("progen2-"):] if upstream_name.startswith("progen2-") else upstream_name
    STOCK_PROC_FMT = f"[progen2-opt stock] {route} progen2-{size} pid={pid} cwd={cwd}"
    return STOCK_PROC_FMT


def load_wall_of_stdout(stdout: str) -> Optional[float]:
    """The stock's own `loading parameters took <s>s` (print_time, sample.py/likelihood.py L31)."""
    for l in stdout.splitlines():
        if l.startswith(LOAD_RE):
            try:
                return float(l[len(LOAD_RE):].rstrip().rstrip("s"))
            except ValueError:
                return None
    return None


# ------------------------------------------------------------------------------------------------------ the calling-process side
def script_of(route: str) -> str:
    return {"sample": "sample.py", "score": "likelihood.py"}[route]


def item_argv(route: str, item: dict, upstream_name: str, job_argv: List[str]) -> List[str]:
    """The script's argv for one item of a job: --model, the job-level flags as given, then the item's own per-item flags."""
    if route == "sample":
        return ["--model", upstream_name] + list(job_argv) + outputs.sample_argv(item)
    return ["--model", upstream_name] + list(job_argv) + ["--context", item["context"]]


def child_command(route: str, script_argv: List[str], env_absent: List[str], allowed: Tuple[str, ...] = ()) -> List[str]:
    cmd = [stack.python(), "-s", "-m", "progen2_opt.stock_cli", "--script", script_of(route), "--env-absent", ",".join(env_absent)]
    if allowed:
        cmd += ["--env-allowed", ",".join(allowed)]
    if CLEAN_RECORD["stripped"] or CLEAN_RECORD["kept"]:
        cmd += ["--stripped", ",".join(CLEAN_RECORD["stripped"]), "--kept", ",".join(CLEAN_RECORD["kept"])]
    return cmd + ["--"] + list(script_argv)


def run_passthrough(route: str, script_argv: List[str], upstream_name: str, workdir: str, env: Dict[str, str], env_absent: List[str],
                    allowed: Tuple[str, ...] = ()) -> dict:
    """ONE stock process whose stdout and stderr are this process's own (nothing captured, nothing cut); returns {rc, wall_s, cmd, pid}."""
    cmd = child_command(route, script_argv, env_absent, allowed)
    t0 = time.monotonic()
    p = subprocess.Popen(cmd, cwd=workdir, env=env)
    print(proc_line(route, upstream_name, p.pid, workdir), file=sys.stderr, flush=True)
    try:
        rc = p.wait(timeout=stack.PROCESS_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        p.kill(); rc = p.wait()
        print(f"{PREFIX} TIMEOUT after {stack.PROCESS_TIMEOUT_S}s (killed)", file=sys.stderr, flush=True)
    return {"rc": rc, "wall_s": time.monotonic() - t0, "cmd": cmd, "pid": p.pid}


def run_item(route: str, item: dict, upstream_name: str, workdir: str, env: Dict[str, str], env_absent: List[str],
             allowed: Tuple[str, ...] = (), job_argv: Optional[List[str]] = None) -> dict:
    """One stock process for one item of a job (streams captured); returns {rc, wall_s, stdout, stderr, block, load_s, cmd, pid}."""
    cmd = child_command(route, item_argv(route, item, upstream_name, list(job_argv or [])), env_absent, allowed)
    t0 = time.monotonic()
    p = subprocess.Popen(cmd, cwd=workdir, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(proc_line(route, upstream_name, p.pid, workdir), file=sys.stderr, flush=True)
    try:
        out, err = p.communicate(timeout=stack.PROCESS_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        p.kill(); out, err = p.communicate()
        err = (err or "") + f"\n{PREFIX} TIMEOUT after {stack.PROCESS_TIMEOUT_S}s (killed)"
    wall = time.monotonic() - t0
    return {"rc": p.returncode, "wall_s": wall, "stdout": out, "stderr": err, "block": outputs.block_of_stdout(route, out),
            "load_s": load_wall_of_stdout(out), "cmd": cmd, "pid": p.pid}


# ------------------------------------------------------------------------------------------------------------- the child side
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -s -m progen2_opt.stock_cli", allow_abbrev=False)
    ap.add_argument("--script", required=True, choices=("sample.py", "likelihood.py"))
    ap.add_argument("--env-absent", default="PROGEN2_", help="comma list; entries ending in '_' are prefixes")
    ap.add_argument("--env-allowed", default="", help="comma list of data-path names a prefix rule must not flag")
    ap.add_argument("--stripped", default="", help="comma list: the names the calling process removed from its environment (for the ENV-CLEAN line)")
    ap.add_argument("--kept", default="", help="comma list: the data-path names the calling process kept (for the ENV-CLEAN line)")
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="-- then the script's own arguments")
    a = ap.parse_args(argv)
    rest = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest
    spec = [s for s in a.env_absent.split(",") if s]
    proof = env_proof(spec, allowed=tuple(s for s in a.env_allowed.split(",") if s))
    proof["stripped"] = [x for x in a.stripped.split(",") if x]
    proof["kept"] = [x for x in a.kept.split(",") if x]
    print(proof_line(proof), file=sys.stderr, flush=True)
    script = os.path.join(os.getcwd(), a.script)
    if not os.path.isfile(script):
        print(f"{PREFIX} ERROR: {script} not found (run from the stock run directory)", file=sys.stderr, flush=True)
        return 2
    sys.argv = [a.script] + list(rest)
    if "" not in sys.path and os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())                       # the stock's `models` package and tokenizer.json resolve against the cwd, as `python sample.py` does
    runpy.run_path(a.script, run_name="__main__")             # --- the stock call: the script's own bytes as __main__ (argv[0] == the script name, as `python sample.py`)
    return 0


if __name__ == "__main__":
    sys.exit(main())
