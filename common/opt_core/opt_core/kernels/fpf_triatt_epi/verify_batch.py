#!/usr/bin/env python3
"""verify_batch.py — v3 delta check ONLY: (1) one bit-exact cell per size (registry fn vs stock forward, real dump pf_c1_b0, un-batched x as in bench.py);
(2) leading batch dim B in {1,2,5}: fn(x_B) vs stock(x_B) bit-exact (cand == stock), plus per-element equality of each vs the un-batched result (documents the
stock `[0]` batch-squeeze semantics that the kernel reproduces).   python -m fpf_triatt_epi.verify_batch --dumps $FPF_DUMPS --sizes 356,546,705,813 --out out"""
import os, sys, json, time, argparse
import torch, torch.nn.functional as F
ap = argparse.ArgumentParser(); ap.add_argument("--dumps", default=os.environ.get("FPF_DUMPS", "dumps")); ap.add_argument("--sizes", default="356,546,705,813")
ap.add_argument("--out", default="out"); ap.add_argument("--spec", default=os.environ.get("FPF_SPEC_DIR", "")); ap.add_argument("--bmax-n", type=int, default=705)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
if a.spec: sys.path.insert(0, a.spec)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fpf, fpf.reference as R
from fpf_triatt_epi import epilogue as E
print("GPU", torch.cuda.get_device_name(), "cfg_sha256", E.config_sha256()[:16], flush=True)
runner = R.build_runner(); model = runner.model.eval(); pb = model.pairformer_stack.blocks[0]
MODS = {"triatt_start": pb.tri_att_start, "triatt_end": pb.tri_att_end}
kw = dict(mask=None, triangle_attention="cuequivariance", inplace_safe=True, chunk_size=None)
rows = []
for N in [int(v) for v in a.sizes.split(",")]:
    d = torch.load(os.path.join(a.dumps, f"acts_{N}tok.pt"), map_location="cuda")
    z = d["pf_c1_b0"]["z"].cuda().contiguous(); z10 = d["pf_c10_b47"]["z"].cuda().contiguous(); del d
    for opname, m in MODS.items():
        with torch.no_grad(), R.amp():
            st = R.stock_call(m, "forward", z, **kw); mine = E.fn(m, z, **kw)
            row = {"N": N, "op": opname, "test": "bitwise_unbatched", "bitwise": bool(torch.equal(mine, st)), "n_diff": int((mine != st).sum()), "shape": list(mine.shape)}
            rows.append(row); print(json.dumps(row), flush=True)
            if N <= a.bmax_n:
                for B in (1, 2, 5):
                    if B == 5 and N > 546: continue
                    xs = [z, z10, z.flip(0).contiguous(), z10.flip(1).contiguous(), (z.float() * 0.5).to(torch.bfloat16)][:B]
                    xB = torch.stack(xs, 0).contiguous()
                    try:
                        stB = R.stock_call(m, "forward", xB, **kw); okS = True
                    except Exception as ex:
                        stB = None; okS = repr(ex)[:120]
                    try:
                        myB = E.fn(m, xB, **kw); okM = True
                    except Exception as ex:
                        myB = None; okM = repr(ex)[:120]
                    row = {"N": N, "op": opname, "test": f"batch_B{B}", "stock_runs": okS, "cand_runs": okM}
                    if stB is not None and myB is not None:
                        row["cand_eq_stock_bitwise"] = bool(torch.equal(myB, stB)); row["n_diff"] = int((myB != stB).sum()); row["shape"] = list(myB.shape)
                        singles = [R.stock_call(m, "forward", xb, **kw) for xb in xs]
                        row["stock_elem_eq_unbatched"] = [bool(torch.equal(stB[b], singles[b])) for b in range(B)]
                        row["cand_elem_eq_unbatched"] = [bool(torch.equal(myB[b], singles[b])) for b in range(B)]
                        del singles
                    rows.append(row); print(json.dumps(row), flush=True)
                    del xB, stB, myB; torch.cuda.empty_cache()
            del st, mine
    del z, z10; torch.cuda.empty_cache()
json.dump({"gpu": torch.cuda.get_device_name(), "config_sha256": E.config_sha256(), "rows": rows, "stats": dict(E._STATS)}, open(os.path.join(a.out, "VERIFY_BATCH.json"), "w"), indent=1)
print("SUMMARY unbatched bitwise all:", all(r["bitwise"] for r in rows if r["test"] == "bitwise_unbatched"),
      "| B=1 cand==stock all:", all(r.get("cand_eq_stock_bitwise") for r in rows if r["test"] == "batch_B1"),
      "| batched cand[b]==unbatched stock(x[b]) all:", all(all(r.get("cand_elem_eq_unbatched", [False])) for r in rows if r["test"].startswith("batch_")),
      "| stock[b]==unbatched (documents stock [0]-squeeze):", [r.get("stock_elem_eq_unbatched") for r in rows if r["test"] in ("batch_B2", "batch_B5")][:4], flush=True)
