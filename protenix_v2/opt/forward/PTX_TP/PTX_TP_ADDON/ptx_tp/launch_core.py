"""ptx_tp.launch_core -- engine-agnostic torchrun launcher core (no protenix imports).

  python -m ptx_tp.launch_core --nproc P [--hook mod:func ...] --entry <module:func | path/to/script.py | console-script> -- <args...>

Parent side: builds and execs `torchrun --standalone --nproc_per_node=P -m ptx_tp.launch_core --rank-side ...` (P=1 -> plain python, no torchrun).
Rank side (every rank): rank_init() = ptx_tp.dist.init_from_env() + prints device name / capability / memory, torch / CUDA / NCCL versions and
every NCCL_* / CUDA_* / TORCH_* env knob (rank 0 prints versions, every rank prints its device line); then calls each --hook in order
(a hook is `module:function`, called with no arguments -- e.g. `ptx_tp:apply_from_env`), then runs the entry with sys.argv = [entry] + args.
API for engine launchers:  torchrun_cmd(nproc, rank_args) ; rank_init(verbose=True) -> (P, rank, device) ; run_entry(entry, args) ; main(argv)
"""
from __future__ import annotations

import argparse
import importlib
import os
import runpy
import shutil
import sys
from typing import List, Optional


def torchrun_cmd(nproc: int, rank_args: List[str], python: Optional[str] = None, extra: Optional[List[str]] = None) -> List[str]:
    python = python or sys.executable
    if int(nproc) <= 1:
        return [python, "-m", "ptx_tp.launch_core", "--rank-side"] + rank_args
    return [python, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={int(nproc)}"] + (extra or []) + \
           ["-m", "ptx_tp.launch_core", "--rank-side"] + rank_args


NCCL_TIMEOUT_S = 1800                                       # the process group's collective timeout (init_process_group timeout=), passed explicitly


def rank_init(verbose: bool = True, backend: Optional[str] = None):
    """Per-rank init: process group (NCCL on GPUs, collective timeout NCCL_TIMEOUT_S), device binding, version + device + NCCL env print.
    Returns (P, rank, device)."""
    import torch
    from ptx_tp.dist import init_from_env
    P, rank, device = init_from_env(backend=backend, timeout_s=NCCL_TIMEOUT_S)
    if verbose:
        if rank == 0:
            nccl = None
            try:
                nccl = ".".join(map(str, torch.cuda.nccl.version())) if torch.cuda.is_available() else None
            except Exception as e:
                nccl = f"n/a ({e!r})"
            knobs = {k: v for k, v in sorted(os.environ.items()) if k.startswith(("NCCL_", "TORCH_NCCL", "CUDA_", "CUBLAS_", "PTX_TP", "TORCH_DIST"))}
            print(f"[launch_core] P={P} torch={torch.__version__} cuda={torch.version.cuda} nccl={nccl} python={sys.version.split()[0]} env={knobs}",
                  file=sys.stderr, flush=True)
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(device)
            drv = _driver_version()
            print(f"[launch_core] rank {rank}/{P} device={device} name={p.name} cc={p.major}.{p.minor} mem={p.total_memory / 2**30:.1f} GiB driver={drv}",
                  file=sys.stderr, flush=True)
        else:
            print(f"[launch_core] rank {rank}/{P} device=cpu (no CUDA; gloo backend)", file=sys.stderr, flush=True)
    return P, rank, device


def _driver_version() -> str:
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip().splitlines()[0] if r.stdout.strip() else "?"
    except Exception:
        return "?"


def call_hook(spec: str):
    mod, _, fn = spec.partition(":")
    m = importlib.import_module(mod)
    return getattr(m, fn or "apply_from_env")()


def run_entry(entry: str, args: List[str]):
    """entry = 'module:func' (called, SystemExit propagated as return code) | 'module' (run as __main__) | 'x.py' path | console script name."""
    if entry.endswith(".py") and os.path.exists(entry):
        sys.argv = [entry] + list(args)
        runpy.run_path(entry, run_name="__main__")
        return 0
    if ":" in entry:
        mod, fn = entry.split(":", 1)
        sys.argv = [mod.split(".")[-1]] + list(args)
        f = getattr(importlib.import_module(mod), fn)
        try:
            rv = f()
            return int(rv) if isinstance(rv, int) else 0
        except SystemExit as e:  # click-style entry points exit
            return int(e.code) if isinstance(e.code, int) else (0 if e.code is None else 1)
    try:
        importlib.util.find_spec(entry)
        found = True
    except Exception:
        found = False
    if found and importlib.util.find_spec(entry) is not None:
        sys.argv = [entry] + list(args)
        runpy.run_module(entry, run_name="__main__", alter_sys=True)
        return 0
    exe = shutil.which(entry)
    if exe:  # console script: exec its python file in-process so hooks stay installed
        sys.argv = [exe] + list(args)
        runpy.run_path(exe, run_name="__main__")
        return 0
    raise SystemExit(f"[launch_core] cannot resolve entry {entry!r}")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args_after = []
    if "--" in argv:
        i = argv.index("--")
        argv, args_after = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nproc", type=int, default=int(os.environ.get("PTX_TP", "1") or 1))
    ap.add_argument("--entry", required=False, default=os.environ.get("PTX_TP_ENTRY", ""), help="module:func | module | script.py | console script")
    ap.add_argument("--hook", action="append", default=[], help="module:func called (no args) on every rank before the entry, in order")
    ap.add_argument("--rank-side", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--no-verbose", action="store_true")
    ap.add_argument("--torchrun-arg", action="append", default=[], help="extra torchrun argument (repeatable)")
    a = ap.parse_args(argv)
    if not a.entry:
        ap.error("--entry is required")
    if not a.rank_side:
        rank_args = ["--entry", a.entry] + sum([["--hook", h] for h in a.hook], []) + (["--no-verbose"] if a.no_verbose else []) + ["--"] + args_after
        cmd = torchrun_cmd(a.nproc, rank_args, extra=a.torchrun_arg)
        env = dict(os.environ)
        env.setdefault("PTX_TP", str(a.nproc))
        print(f"[launch_core] exec: {' '.join(cmd)}", file=sys.stderr, flush=True)
        os.execvpe(cmd[0], cmd, env)
    rank_init(verbose=not a.no_verbose)
    for h in a.hook:
        call_hook(h)
    rc = run_entry(a.entry, args_after)
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    except Exception:
        pass
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
