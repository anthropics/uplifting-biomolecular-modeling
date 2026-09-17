"""loadcheck.py — deterministic test vectors for the protenix_fpf_msa kernels; see protenix_fpf_apb.loadcheck for the scheme.
`python -m protenix_fpf_msa.loadcheck` (read-only comparison; the recorder is workbench tooling, not part of this package)."""
from __future__ import annotations
import json, os, sys
import torch
from opt_core.kernels.apb.fpf_apb.loadcheck import det, sha, key          # shared generator / hashing / stack key (protenix_opt 0.3.51: the shared core's carried copy)

HERE = os.path.dirname(os.path.abspath(__file__))
VEC = os.path.join(HERE, "VECTORS.json")


def cases():
    from opt_core.ops.msa_fused.msa_triton import ln_linear, opm_out, pwa_ln_vg, pwa_out2
    out = {}
    S, N, C = 77, 52, 128
    m = det((S, N, C), 3.0, 31, dtype=torch.bfloat16); lnw = det((C,), 0.5, 32) + 1.0; lnb = det((C,), 0.2, 33)
    w1 = det((32, C), 0.1, 34); w2 = det((32, C), 0.1, 35)
    a, b = ln_linear(m, lnw, lnb, (w1, w2)); out["ln_linear_a"] = a; out["ln_linear_b"] = b
    outer = torch.einsum("...bac,...dae->...bdce", a.transpose(-2, -3), b.transpose(-2, -3))          # the stock statement (cuBLAS) — hashed too, informational
    wout = det((256, 1024), 0.05, 36, dtype=torch.bfloat16); bias = det((256,), 0.1, 37, dtype=torch.bfloat16)
    norm = torch.full((N, N), float(S) + 1e-3, device="cuda", dtype=torch.bfloat16); o = torch.empty(N, N, 256, device="cuda", dtype=torch.bfloat16)
    out["opm_out"] = opm_out(outer, wout.t().contiguous(), bias, norm, o).clone()
    M = 61
    m2 = det((M, N, C), 3.0, 41, dtype=torch.bfloat16); wmv = det((64, C), 0.1, 42); wmg = det((64, C), 0.1, 43)
    v_hm, g = pwa_ln_vg(m2, lnw, lnb, wmv, wmg); out["pwa_ln_v"] = v_hm; out["pwa_ln_g"] = g
    wv = det((8, N, M, 8), 1.0, 44, dtype=torch.bfloat16); wt = det((64, 128), 0.1, 45, dtype=torch.bfloat16)
    out["pwa_out2"] = pwa_out2(g, wv, wt)
    torch.cuda.synchronize()
    return out


def compute():
    """{case: {sha256, shape, dtype, finite}} for this stack (used by main() and by the workbench recorder)."""
    res = cases()
    return {name: {"sha256": sha(t), "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""), "finite": bool(torch.isfinite(t.float()).all())} for name, t in res.items()}


def main(argv):
    got = compute()
    k = key()
    db = json.load(open(VEC)) if os.path.exists(VEC) else {"schema": "fpf_vectors/v1", "generator": "protenix_fpf_apb.loadcheck.det", "expected": {}}
    exp = db["expected"].get(k)
    if exp is None:
        print(f"loadcheck: no expected vectors for {k} (keys: {sorted(db['expected'])}); outputs finite={all(v['finite'] for v in got.values())}"); return 2
    bad = [n for n in exp if got.get(n, {}).get("sha256") != exp[n]["sha256"]]
    for n in sorted(got): print(("OK   " if n not in bad else "DIFF ") + f"{n:14s} {got[n]['dtype']:8s} {got[n]['shape']} {got[n]['sha256'][:16]}")
    print("loadcheck:", "PASS" if not bad else f"FAIL {bad}"); return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
