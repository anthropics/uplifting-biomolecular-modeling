"""ptx_tp.launch -- launcher for tensor-parallel STOCK Protenix inference: P torchrun ranks, every rank running the STOCK CLI entry
(runner.batch_inference:protenix_cli) with the unit's hooks installed in this order.

  python -m ptx_tp.launch --nproc P [--lazy-relp] [--lift-guard] [--label L] [--out DIR] -- <stock protenix CLI arguments>
  e.g.  python -m ptx_tp.launch --nproc 2 --lazy-relp --lift-guard --label smoke --out out/tp -- pred -i items/L.1052_1BTLx4.json -o out -n protenix-v2 --seeds 101

  1. --lazy-relp   == PTX_TP_RELP=lazy (the relative-position rows built lazily per row block)
  2. --lift-guard  == XL_LIFT_GUARD=1 semantics (hook ptx_tp.runner_hooks:install_guard_lift)
  3. ptx_tp.apply_from_env()  (PTX_TP=P; the diffusion / PairCore v3 / layout-block defaults below, each kept when already set in the environment)
  `--trimul_kernel torch` is appended to the stock arguments unless they name --trimul_kernel (the sharded TriMul is the torch statement).
  --out DIR holds the per-rank phase/memory jsonl (PTX_TP_PHASE_LOG) and runmeta_<label>.json.
Prints GPU model + driver (nvidia-smi --query-gpu=name,driver_version) and torch/CUDA/NCCL versions before launching."""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

from ptx_tp.launch_core import torchrun_cmd

ENTRY = "runner.batch_inference:protenix_cli"


def parse(argv):
    argv = list(argv)
    rest = []
    if "--" in argv:
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(prog="python -m ptx_tp.launch", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nproc", type=int, required=True, help="tensor-parallel degree P (torchrun ranks)")
    ap.add_argument("--label", default="job", help="job label: PTX_TP_JOB_LABEL and runmeta_<label>.json")
    ap.add_argument("--lazy-relp", action="store_true", help="PTX_TP_RELP=lazy")
    ap.add_argument("--lift-guard", action="store_true", help="XL_LIFT_GUARD=1 (hook ptx_tp.runner_hooks:install_guard_lift)")
    ap.add_argument("--out", default="out", help="job out dir (phase logs and runmeta land here)")
    a = ap.parse_args(argv)
    return a, rest


def gpu_banner():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        print("[launch] nvidia-smi: " + " | ".join(x.strip() for x in r.stdout.strip().splitlines()), flush=True)
    except Exception as e:
        print(f"[launch] nvidia-smi unavailable: {e!r}", flush=True)
    try:
        import torch
        nccl = ".".join(map(str, torch.cuda.nccl.version())) if torch.cuda.is_available() else None
        print(f"[launch] torch={torch.__version__} cuda={torch.version.cuda} nccl={nccl}", flush=True)
    except Exception as e:
        print(f"[launch] torch unavailable here: {e!r}", flush=True)
    try:
        import importlib.metadata as im
        print(f"[launch] protenix=={im.version('protenix')}", flush=True)
    except Exception:
        pass


def build(a, rest):
    """-> (cmd list, env dict, description)"""
    env = dict(os.environ)
    hooks = []
    if a.lazy_relp:
        env["PTX_TP_RELP"] = "lazy"
    if a.lift_guard:
        env["XL_LIFT_GUARD"] = "1"
        hooks.append("ptx_tp.runner_hooks:install_guard_lift")
    env["PTX_TP_JOB_LABEL"] = a.label
    env.setdefault("PTX_TP_PHASE_LOG", os.path.abspath(a.out))
    env["PTX_TP"] = str(a.nproc)
    # diffusion seam defaults: row-split attention with the SDPA query-padding guard, replicated denoiser below 3,841
    for k, v in (("PTX_TP_DIFF_REPLICATE_BELOW", "3841"), ("PTX_TP_DIFF_ATTN", "rowsplit"), ("PTX_TP_SDPA_QPAD", "1"), ("PTX_TP_DIFFCACHE_GB", "8"),
                 ("PTX_TP_BIAS_CHUNK", "128"), ("PTX_TP_F2_GATHER_BELOW", "3841"), ("PTX_TP_RNG_SYNC", "1"), ("PTX_TP_RELP", "lazy"), ("PTX_TP_ATOM_LOCAL", "1")):
        env.setdefault(k, v)
    # TriMul / TriAtt tiling defaults by input size: shard = N^2*512/P bytes; BCACHE=device while shard*2.5+25 GB < 170 GB else host
    # (+ pinned pool 64 GB reserved at init on every rank); ROWS_A 1024 (2048 with host stash), ROWS_B 256, ZCOPY lazy (host at N >= 15000). N is read from the input JSON.
    n_tok = _guess_n_tokens(rest)
    if n_tok:
        shard_gb = n_tok * n_tok * 512 / a.nproc / 1e9
        host_stash = (shard_gb * 2.5 + 25.0) >= 170.0
        rec = {"PTX_TP_TRIMUL_BCACHE": "host" if host_stash else "device", "PTX_TP_TRIMUL_ROWS_A": "2048" if host_stash else "1024",
               "PTX_TP_TRIMUL_ROWS_B": "256", "PTX_TP_TRIMUL_ZCOPY": "host" if n_tok >= 15000 else "lazy"}   # incoming zcopy=lazy costs roughly two extra shard-sized transients at large N => host
        if host_stash:
            rec["PTX_TP_PINNED_POOL_GB"] = "64"
        for k, v in rec.items():
            env.setdefault(k, v)
        user_env = set(os.environ)
        if "PTX_TP_B" not in user_env and n_tok:                     # largest layout block with NO empty rank
            try:
                from ptx_tp.dist import Layout as _L
                chosenB = None
                for _B in (128, 64, 32, 16):
                    Ls = [_L(int(n_tok), a.nproc, r, B=_B) for r in range(a.nproc)]
                    if Ls[0].replicated or all(L.R > 0 for L in Ls):
                        chosenB = _B
                        break
                if chosenB is not None and chosenB != 128:
                    env["PTX_TP_B"] = str(chosenB)
                print(f"[launch] layout block B = {chosenB or 128} for N~{n_tok}, P={a.nproc} (auto: largest of 128/64/32/16 with no empty rank"
                      f"{'; B != 128 => label the row B' + str(chosenB) if chosenB not in (None, 128) else ''})", flush=True)
            except Exception as _e:
                print(f"[launch] auto-B skipped: {_e!r}", flush=True)
        if n_tok >= 15000:
            if "PTX_TP_BIAS_CHUNK" not in user_env:
                env["PTX_TP_BIAS_CHUNK"] = "32"
            if "PTX_TP_TRIATT_HEADS_PER_CALL" not in user_env:
                env["PTX_TP_TRIATT_HEADS_PER_CALL"] = "1"
        env["PTX_TP_TILING_DESC"] = (f"N~{n_tok} P={a.nproc} shard={shard_gb:.1f}GB bcache={env['PTX_TP_TRIMUL_BCACHE']} rowsA={env['PTX_TP_TRIMUL_ROWS_A']} "
                                     f"rowsB={env['PTX_TP_TRIMUL_ROWS_B']} zcopy={env['PTX_TP_TRIMUL_ZCOPY']} pinned_pool_GB={env.get('PTX_TP_PINNED_POOL_GB', '0')} "
                                     f"bias_chunk={env.get('PTX_TP_BIAS_CHUNK', '128')} triatt_heads_per_call={env.get('PTX_TP_TRIATT_HEADS_PER_CALL', 'auto')}")
        print(f"[launch] TRIMUL/TRIATT env (PairCore v3 recommended): {env['PTX_TP_TILING_DESC']}", flush=True)
    if "--trimul_kernel" not in rest:
        rest = rest + ["--trimul_kernel", "torch"]      # the TP TriMul IS the torch statement; its single-card equivalent is stock --trimul_kernel torch
        print("[launch] TP run: appended --trimul_kernel torch (PairCore tp_trimul implements the torch TriMul statement; cuEq TriMul = TIER-2-by-kernel reference only)", flush=True)
    hooks.append("ptx_tp:apply_from_env")
    rank_args = ["--entry", ENTRY] + sum([["--hook", h] for h in hooks], []) + ["--"] + rest
    cmd = torchrun_cmd(a.nproc, rank_args)
    desc = f"TP P={a.nproc} (torchrun; hooks: {hooks})"
    return cmd, env, desc


def _guess_n_tokens(rest):
    """N_token estimate from the Protenix input JSON named after -i/--input (sum of sequence length x count over entities; ligands ~ atoms unknown -> 0)."""
    try:
        import json as _json
        path = None
        for i, t in enumerate(rest):
            if t in ("-i", "--input") and i + 1 < len(rest):
                path = rest[i + 1]
        if not path or not os.path.isfile(path):
            return 0
        d = _json.load(open(path))
        e = d[0] if isinstance(d, list) else d
        n = 0
        for ent in e.get("sequences", []):
            for k, v in ent.items():
                if isinstance(v, dict):
                    seq = v.get("sequence", "")
                    n += len(seq) * int(v.get("count", 1))
        return n
    except Exception:
        return 0


def main(argv=None) -> int:
    a, rest = parse(sys.argv[1:] if argv is None else argv)
    return run_one(a, rest, banner=True)


def run_one(a, rest, banner=False) -> int:
    if not rest:
        print("usage: python -m ptx_tp.launch --nproc P [--lazy-relp] [--lift-guard] [--label L] [--out DIR] -- <protenix CLI args, e.g. pred -i x.json -o out -n protenix-v2 ...>", file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)
    gpu_banner()
    cmd, env, desc = build(a, rest)
    knobs = {k: v for k, v in sorted(env.items()) if k.startswith(("PTX_", "XL_", "CUBLAS", "PROTENIX_DET", "LAYERNORM", "NCCL_")) or k == "PYTHONPATH"}
    print(f"[launch] {desc}\n[launch] env: {knobs}\n[launch] cmd: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
