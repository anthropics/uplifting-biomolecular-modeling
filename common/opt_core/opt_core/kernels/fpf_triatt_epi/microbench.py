#!/usr/bin/env python3
"""timing probe.py — epilogue kernel config sweep WITHOUT the model (synthetic o/g/W/z with real shapes & realistic magnitudes; bit-exact check vs the torch/cuBLAS
emulation of the stock math, which round-1b showed is itself bit-exact == stock _wrap_up). Engineering only; tables come from selftest.py on real dumps.
   python -m fpf_triatt_epi.microbench --sizes 356,705,813 --out out/"""
import os, sys, json, time, argparse, itertools
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpf_triatt_epi import epilogue as E

ap = argparse.ArgumentParser(); ap.add_argument("--sizes", default="356,705,813"); ap.add_argument("--out", default="out"); ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--c", type=int, default=256); ap.add_argument("--H", type=int, default=8); ap.add_argument("--D", type=int, default=32)
ap.add_argument("--cfgs", default="")   # "KV,BI,BJ,W,S;..." else default grid
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
dev = "cuda"; print("GPU", torch.cuda.get_device_name(), flush=True)


def evms(fn, reps=a.reps, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(reps): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / reps


if a.cfgs:
    CFGS = [tuple(int(v) for v in c.split(",")) for c in a.cfgs.split(";")]
else:
    CFGS = []
    for KV in (2, 1):
        for (BI, BJ) in ((8, 8), (16, 8), (8, 16), (16, 4), (4, 16), (16, 16), (32, 4), (4, 32), (32, 8)):
            for W in (4, 8):
                for S in (1, 2, 3):
                    if KV == 1 and S == 3: continue
                    CFGS.append((KV, BI, BJ, W, S))
rows = []
c, H, D = a.c, a.H, a.D
for N in [int(x) for x in a.sizes.split(",")]:
    torch.manual_seed(N)
    o = (torch.randn(N, H, N, D, device=dev) * 20).to(torch.bfloat16)               # cuEq output magnitude ~ LN'd values * attention -> O(1-30)
    g = (torch.randn(N, N, H * D, device=dev) * 4).to(torch.bfloat16)
    W = (torch.randn(c, H * D, device=dev) / 16).to(torch.bfloat16)
    z = (torch.randn(N, N, c, device=dev) * 30).to(torch.bfloat16)
    ref = E.torch_reference_epilogue(o, g, W)                                          # cuBLAS path
    zr = z.clone(); E.torch_reference_epilogue(o, g, W, zr, ending=True, residual=True)
    # stock-pieces timing (same as selftest, no module): sigmoid, mul, flatten, linear, add, transpose
    import torch.nn.functional as F
    zscr_stock = z.clone()                                                              # scratch for the stock-pieces timing (mutated in place; never compared)
    def stock_pieces(ending):
        gg = torch.sigmoid(g).view(N, N, H, D); om = o.transpose(-2, -3) * gg; of = om.reshape(N, N, H * D); u = F.linear(of, W); zz = zscr_stock; zz.add_(u)
        return zz.transpose(-2, -3).contiguous() if ending else zz
    t_stock_s = evms(lambda: stock_pieces(False)); t_stock_e = evms(lambda: stock_pieces(True))
    zbytes = N * N * c * 2
    print(f"N={N} stock pieces start {t_stock_s:.3f} ms end {t_stock_e:.3f} ms", flush=True)
    for (KV, BI, BJ, Wn, S) in CFGS:
        cfg = dict(KVER=KV, BI=BI, BJ=BJ, num_warps=Wn, num_stages=S, EXP="libdevice")
        try:
            u = E.triatt_epilogue(o, g, W, cfg=cfg); ok = bool(torch.equal(u.reshape(ref.shape), ref))
            zz = z.clone(); E.triatt_epilogue(o, g, W, zz, ending=True, residual=True, cfg=cfg); okz = bool(torch.equal(zz, zr))
            t_op = evms(lambda: E.triatt_epilogue(o, g, W, cfg=cfg)); zs = z.clone()
            t_bs = evms(lambda: E.triatt_epilogue(o, g, W, zs, ending=False, residual=True, cfg=cfg))
            t_be = evms(lambda: E.triatt_epilogue(o, g, W, zs, ending=True, residual=True, cfg=cfg))
            row = {"N": N, "cfg": [KV, BI, BJ, Wn, S], "ms_op": round(t_op, 4), "ms_blk_start": round(t_bs, 4), "ms_blk_end": round(t_be, 4), "GBs_blk_end": round(4 * zbytes / t_be / 1e6), "bitwise_op": ok, "bitwise_blk_end": okz,
                   "stock_start": round(t_stock_s, 4), "stock_end": round(t_stock_e, 4)}
        except Exception as ex:
            row = {"N": N, "cfg": [KV, BI, BJ, Wn, S], "error": repr(ex)[:160]}
        rows.append(row); print("MB", json.dumps(row), flush=True)
    del o, g, W, z, ref, zr; torch.cuda.empty_cache()
json.dump(rows, open(os.path.join(a.out, "MICROBENCH.json"), "w"), indent=1)
best = {}
for r in rows:
    if "ms_blk_end" in r and r["bitwise_op"] and r["bitwise_blk_end"]:
        k = r["N"]
        if k not in best or r["ms_blk_end"] < best[k]["ms_blk_end"]: best[k] = r
print("BEST", json.dumps(best), flush=True)
