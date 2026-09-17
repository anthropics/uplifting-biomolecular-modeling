#!/usr/bin/env python3
"""esmfold2_trimul v6.1 — replay the sealed v6 test vectors on this device with the FIXED k3v6 (same arithmetic => same expected bytes), plus an
optional back-to-back determinism stress of the K3 CuTe kernel (`--stress N`: N launches per (case N=200, direction, mask); every launch must equal launch 0).
vectors.pt is read from ./vectors/ if present, else from the sibling v6 package (../v6/tests/vectors/vectors.pt; vectors.json here records its sha256).

Cases: N in {36, 44} (stored) and N = 200 (regenerated deterministically from z_N44, sha256 checked) x direction x mask; kernels per case:
v5 / v6 whole route, K3 alone (k3cute fastsig 0|1 with lnfold=1 — the LNFOLD=0 builds are withdrawn —, k3v5) on the route's own planes; one case also carries STORED
planes so the CuTe K3 is tested independently of K1 / cuBLAS reproducibility.  Verdicts per (case, kernel):
    BITWISE      sha256 == recorded for this device class (sm90)
    TOLERANCE    bytes differ but within MAX_ULP bf16 ulps of the stored expected tensor (v6 / k3cute_fs1_lf1 of the stored cases), or — where
                 no tensor is stored — max|d| vs the fp32 reference within 2x the recorded value (numerics class = tolerance / fast tier)
    FAIL / N/A   beyond that / k3cute on a non-sm_90 device
Run: python tests/test_vectors.py (pytest collects test_all).  Needs torch, triton (tensor descriptors: 3.6+), cuda.bindings; sm_90 for k3cute."""
import hashlib, importlib.util, json, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE)
MAX_ULP = 4
WKEYS = ("norm_in_w", "norm_in_b", "p_in_w", "g_in_w", "norm_out_w", "norm_out_b", "p_out_w", "g_out_w")


def _face():
    spec = importlib.util.spec_from_file_location("esmfold2_trimul_face", os.path.join(PKG, "python", "face.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def tbytes(t): return t.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()
def sha(t): return hashlib.sha256(tbytes(t)).hexdigest()


def tiled_pair(N, base):
    n = base.shape[0]; i = torch.arange(N); off = (((i[:, None] * 131 + i[None, :]) % 61) - 30).float() / 32.0
    zz = base.float()[i[:, None] % n, i[None, :] % n] + off[:, :, None]
    return zz.to(torch.bfloat16).contiguous()


def ulp_stats(o, e):
    o = o.float().cpu(); e = e.float().cpu(); d = (o - e).abs()
    ulp = torch.exp2(torch.floor(torch.log2(e.abs().clamp_min(2.0 ** -120))) - 7)
    return dict(max_abs=float(d.max()), mean_abs=float(d.mean()), max_ulp=float((d / ulp).max()), frac_diff=float((d > 0).float().mean()))


def main(verbose=True):
    F = _face(); meta = json.load(open(os.path.join(HERE, "vectors", "vectors.json")))
    pt = os.path.join(HERE, "vectors", "vectors.pt")
    if not os.path.exists(pt): pt = os.path.join(os.path.dirname(PKG), "v6", "tests", "vectors", "vectors.pt")
    if hashlib.sha256(open(pt, "rb").read()).hexdigest() != meta["pt_sha256"]:
        print(f"FAIL vectors.pt at {pt}: sha256 != vectors.json pt_sha256"); return 1
    T = torch.load(pt, map_location="cpu")
    if not torch.cuda.is_available():
        print("esmfold2_trimul vectors: no CUDA device — nothing to run"); return 2
    cc = torch.cuda.get_device_capability(); CC = f"sm{cc[0]}{cc[1]}"; cute_ok = cc == (9, 0)
    fails, lines = 0, []
    X = dict(T)
    for k, ent in meta["inputs"].items():
        if k not in X and k.startswith("z_N"):
            X[k] = tiled_pair(int(k[3:]), T["z_N44"])
        if k not in X and k.startswith("mask_N"):
            X[k] = (tiled_pair(int(k[6:]), T["z_N44"])[:, :, 0].float() > -0.9).float()
        if sha(X[k]) != ent["sha256"]:
            lines.append(f"FAIL input {k}: sha256 differs from vectors.json (corrupt file or generator drift)"); fails += 1
    W = {k: X[k].cuda() for k in WKEYS}; w = F.pack_weights(**W)
    F.load(prebuilt=None)
    with torch.inference_mode():
        for case in meta["cases"]:
            N, direction, use_mask = case["N"], case["direction"], case["mask"]
            z = X[f"z_N{N}"].cuda(); mk = X[f"mask_N{N}"].cuda() if use_mask else None
            outs = {}
            try:
                outs["v5"] = F.forward(z, direction, mk, w, kernel="v5")
                if cute_ok: outs["v6"] = F.forward(z, direction, mk, w, kernel="v6")
                x, n_, Np = F.planes(z, direction, mk, w)
                if "planes" in case:
                    xs = T[case["planes"]["stored"]].cuda()
                    lines.append(f"{'BITWISE' if torch.equal(xs, x) else 'differs':<9} {case['name']:>22} {'planes(K1+bmm)':>15}: this device's planes vs the stored planes (informational: cuBLAS / Triton order)"); x = xs
                for fs in (0, 1):                                   # lnfold=1 builds only: the LNFOLD=0 builds are withdrawn (vectors.json 'withdrawn')
                    if cute_ok: outs[f"k3cute_fs{fs}_lf1"] = F.k3_forward(x, z, w, kernel="k3cute", fastsig=fs, lnfold=True)
                outs["k3v5_fs1"] = F.k3_forward(x, z, w, kernel="k3v5", fastsig=True); outs["k3v5_fs0"] = F.k3_forward(x, z, w, kernel="k3v5", fastsig=False)
                torch.cuda.synchronize()
            except Exception as e:
                lines.append(f"FAIL {case['name']}: {type(e).__name__}: {str(e)[:300]}"); fails += 1; continue
            ref = None
            for kern, exp in case["expected"].items():
                if kern not in outs:
                    lines.append(f"N/A       {case['name']:>22} {kern:>15}: sm_90a kernel, device is {CC}"); continue
                o = outs[kern]; h = sha(o); rec = exp.get(CC); stored = next((v["stored"] for v in exp.values() if "stored" in v), None)
                if rec and h == rec["sha256"]:
                    lines.append(f"BITWISE   {case['name']:>22} {kern:>15}: sha256 {h[:16]} == recorded ({CC})"); continue
                if stored is not None:
                    st = ulp_stats(o, T[stored]); verdict = "TOLERANCE" if st["max_ulp"] <= MAX_ULP else "FAIL"; fails += verdict == "FAIL"
                    lines.append(f"{verdict:<9} {case['name']:>22} {kern:>15}: sha256 {h[:16]} != {'recorded ' + rec['sha256'][:16] if rec else 'no record for ' + CC}; vs stored: max|d| {st['max_abs']:.4g} mean {st['mean_abs']:.3g} max_ulp {st['max_ulp']:.2f} frac {st['frac_diff']:.4f}")
                else:
                    if ref is None: ref = F.reference_fp32(z, direction, mk, **{k: v.float() for k, v in W.items()})
                    d = (o.float() - ref).abs(); recn = case["numerics_vs_fp32"].get(kern, {}); ok = float(d.max()) <= 2.0 * recn.get("max_abs", 1e9) + 1e-3; fails += not ok
                    lines.append(f"{'TOLERANCE' if ok else 'FAIL':<9} {case['name']:>22} {kern:>15}: sha256 {h[:16]} {'!= recorded ' + rec['sha256'][:16] if rec else '(no record for ' + CC + ')'}; vs fp32: max|d| {float(d.max()):.4g} (recorded {recn.get('max_abs')}) mean {float(d.mean()):.3g} (recorded {recn.get('mean_abs')})")
    if verbose:
        d = F.describe(); print(f"esmfold2_trimul v6.1 vectors on {torch.cuda.get_device_name()} ({CC}), torch {torch.__version__}: prebuilt {d.get('prebuilt')} cell {d.get('cell')}")
        print("\n".join(lines)); print("RESULT:", "PASS" if fails == 0 else f"FAIL ({fails})")
    return 1 if fails else 0


def stress(n_launches=2000, N=200, verbose=True):
    """Back-to-back determinism stress of the K3 CuTe kernel of record (fastsig 1, lnfold 1): for (direction, mask) at N tokens, launch 0 is
    checked against the fp32 reference (max|d| < 0.25) and every later launch must be byte-identical to launch 0.  Returns the number of
    differing launches (0 = PASS).  The v6 builds without the release-ordering fix fail this at ~1e-4..1e-2 per launch on H100."""
    F = _face(); meta = json.load(open(os.path.join(HERE, "vectors", "vectors.json")))
    pt = os.path.join(HERE, "vectors", "vectors.pt")
    if not os.path.exists(pt): pt = os.path.join(os.path.dirname(PKG), "v6", "tests", "vectors", "vectors.pt")
    T = torch.load(pt, map_location="cpu")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        print("esmfold2_trimul stress: needs an sm_90 device"); return 2
    W = {k: T[k].cuda() for k in WKEYS}; w = F.pack_weights(**W); F.load(prebuilt=None)
    z = tiled_pair(N, T["z_N44"]).cuda(); bad_total = 0; lines = []
    with torch.inference_mode():
        for direction in ("outgoing", "incoming"):
            for use_mask in (False, True):
                mk = (z[:, :, 0].float() > -0.9).float() if use_mask else None
                ref = F.reference_fp32(z, direction, mk, **{k: v.float() for k, v in W.items()})
                x, n_, Np = F.planes(z, direction, mk, w); o0 = F.k3_forward(x, z, w, kernel="k3cute"); torch.cuda.synchronize()
                e0 = float((o0.float() - ref).abs().max()); o = torch.empty_like(z); nbad = torch.zeros((), dtype=torch.int64, device="cuda")
                for r in range(n_launches):
                    F.k3_forward(x, z, w, kernel="k3cute", out=o); nbad += (o != o0).any().to(torch.int64)
                torch.cuda.synchronize(); nb = int(nbad.item()) + int(e0 > 0.25); bad_total += nb
                lines.append(f"{'PASS' if nb == 0 else 'FAIL'}  stress N={N} {direction:9s} mask={int(use_mask)}: {int(nbad.item())}/{n_launches} launches differ from launch 0 (launch-0 max|d| vs fp32 {e0:.4f})")
    if verbose:
        print("\n".join(lines)); print("STRESS RESULT:", "PASS" if bad_total == 0 else f"FAIL ({bad_total})")
    return bad_total


def test_all():
    assert main() == 0


def test_stress():
    assert stress(500) == 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--stress":
        sys.exit(1 if stress(int(sys.argv[2]) if len(sys.argv) > 2 else 2000) else 0)
    rc = main(); rs = stress(300)
    sys.exit(1 if (rc or rs) else 0)
