"""loadcheck.py — deterministic test vectors for the protenix_fpf_apb kernels: inputs are generated from an integer hash
(bit-reproducible on any machine, no RNG / library version dependence), each shipped cell runs once, and the sha256 of the output BYTES is compared
with VECTORS.json (keyed by compute capability + triton version). `python -m protenix_fpf_apb.loadcheck` (read-only comparison; the recorder that
writes new stack keys is workbench tooling and is not part of this package)."""
from __future__ import annotations
import hashlib, json, math, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
VEC = os.path.join(HERE, "VECTORS.json")


def det(shape, scale=1.0, salt=0, dtype=torch.float32, device="cuda"):
    """x[i] = ((i * 2654435761 + salt * 40503) mod 2^32) / 2^32 - 0.5, times scale: exact integer arithmetic + one exact int->fp32 conversion."""
    n = 1
    for s in shape: n *= s
    i = torch.arange(n, dtype=torch.int64, device=device)
    u = (i * 2654435761 + salt * 40503) % (1 << 32)
    x = (u.to(torch.float64) / float(1 << 32) - 0.5) * (2.0 * scale)          # fp64 exact for 32-bit ints; then one rounding to dtype
    return x.to(dtype).reshape(shape)


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes() if t.dtype != torch.bfloat16 else
                          t.detach().contiguous().cpu().view(torch.int16).numpy().tobytes()).hexdigest()


def cases():
    from .apb_triton import apb_views, precast16
    from .atom_triton import atom_apb
    from .pf_triton import pf_bias
    out = {}
    # DiT token site (S=5, H=16, D=48), N=72 (ragged vs the 64 tile): cells tf32x3 (dit_attn) and fp16 precast (dit_attn_fp16), plus bf16-in (Sampler feed)
    S, N, H, D = 5, 72, 16, 48
    q = det((S, N, H, D), 2.0, 1); k = det((S, N, H, D), 2.0, 2); v = det((S, N, H, D), 2.0, 3); g = det((S, N, H, D), 4.0, 4); b = det((H, N, N), 3.0, 5)
    out["dit_tf32x3_N72"] = apb_views(q, k, v, b, g, scale=1 / math.sqrt(D), out_dtype=torch.float32, opd="tf32x3")
    q16, k16, v16 = precast16(q, k, v)
    out["dit_fp16precast_N72"] = apb_views(q16, k16, v16, b, g, scale=1 / math.sqrt(D), out_dtype=torch.float32)
    out["dit_bf16in_N72"] = apb_views(q.bfloat16(), k.bfloat16(), v.bfloat16(), b, g.bfloat16(), scale=1 / math.sqrt(D), out_dtype=torch.bfloat16)
    # atom site (S=2, H=4, D=32), N_atom=333 (11 trunks, ragged), bias [1,4,T,32,128]: cell tf32rn
    S, N, H, D = 2, 333, 4, 32
    T = math.ceil(N / 32)
    q = det((S, N, H, D), 2.0, 11); k = det((S, N, H, D), 2.0, 12); v = det((S, N, H, D), 2.0, 13); g = det((S, N, H, D), 4.0, 14); b = det((1, H, T, 32, 128), 3.0, 15)
    out["atom_tf32rn_N333"] = atom_apb(q, k, v, b, g, out_dtype=torch.float32, opd="tf32rn")
    # pf producer: z [N,N,256] bf16, LN weight/bias, W [16,256]; N=52 (pitch-8 padding 52->56)
    N, C, Hh = 52, 256, 16
    z = det((N, N, C), 3.0, 21, dtype=torch.bfloat16); lnw = det((C,), 0.5, 22) + 1.0; lnb = det((C,), 0.2, 23); Wz = det((Hh, C), 0.1, 24)
    out["pf_bias_N52"] = pf_bias(z, lnw, lnb, Wz, 1e-5, out_dtype=torch.float32)[:, :, :N].contiguous()
    # pf core: q,k,v,g [1,N,16,24] bf16 + that bias
    q = det((1, N, Hh, 24), 2.0, 25, dtype=torch.bfloat16); k = det((1, N, Hh, 24), 2.0, 26, dtype=torch.bfloat16); v = det((1, N, Hh, 24), 2.0, 27, dtype=torch.bfloat16); g = det((1, N, Hh, 24), 4.0, 28, dtype=torch.bfloat16)
    out["pf_core_bf16_N52"] = apb_views(q, k, v, out["pf_bias_N52"], g, scale=1 / math.sqrt(24), out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    return out


def key():
    import triton
    return f"cc{'%d%d' % torch.cuda.get_device_capability()}|triton{triton.__version__}|torch{torch.__version__.split('+')[0]}"


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
        print(f"loadcheck: no expected vectors for {k} (keys: {sorted(db['expected'])}) — outputs computed, finite={all(v['finite'] for v in got.values())}; a measured stack's expected values are added with the workbench recorder (not shipped)"); return 2
    bad = [n for n in exp if got.get(n, {}).get("sha256") != exp[n]["sha256"]]
    for n in sorted(got): print(("OK   " if n not in bad else "DIFF ") + f"{n:24s} {got[n]['dtype']:8s} {got[n]['shape']} {got[n]['sha256'][:16]}")
    print("loadcheck:", "PASS" if not bad else f"FAIL {bad}"); return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
