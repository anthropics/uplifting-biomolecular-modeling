"""vectors.py — byte-level cases for the cell this unit serves (pair Transition, c_in 256, hidden 1024, bf16, sm_90).

Inputs (rows z, the three projection weights, LayerNorm weight/bias) are generated from fixed integer seeds with numpy's frozen RandomState stream, so no
input bytes are stored; vectors.json holds, per case, the sha256 of the expected outputs (residual form `z + transition(z)` and update form
`transition(z)`) and, for the small cases, the expected residual-form output as raw little-endian bf16 bytes (expected_M<rows>.bf16).  The expected
outputs ARE the stock Protenix bf16-autocast chain (FusedLayerNorm fast_layernorm kernel + cuBLAS GEMMs + F.silu) on H100 / torch 2.13.0+cu130, which this
unit reproduces bit for bit; a different GPU class or GEMM library may legitimately produce different stock bytes.

  python vectors.py              run every case through the unit (both LayerNorm routes) and compare with vectors.json -> prints OK / MISMATCH per case
  python vectors.py --write      recompute the expected outputs with the STOCK modules and rewrite vectors.json + expected_*.bf16 (maintainers)

Requires the kit environment (protenix importable, LAYERNORM_TYPE=fast_layernorm, a CUDA sm_90 device)."""
import os, sys, json, hashlib, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = [{"name": "M8", "rows": 8, "seed": 11, "store_bytes": True}, {"name": "M333", "rows": 333, "seed": 12, "store_bytes": True},
         {"name": "M4096", "rows": 4096, "seed": 13, "store_bytes": False}]
C_IN, HIDDEN = 256, 1024


def make_case(case, device):
    """deterministic inputs: numpy RandomState(seed) float32 normals -> bf16 (round to nearest even) tensors on `device`."""
    rs = np.random.RandomState(case["seed"])
    def t(shape, scale):
        return torch.from_numpy((rs.standard_normal(shape) * scale).astype(np.float32)).to(torch.bfloat16).to(device)
    z = t((case["rows"], C_IN), 4.0)                       # pair activations have O(1..10) magnitudes deep in the stack
    wa = t((HIDDEN, C_IN), C_IN ** -0.5); wb = t((HIDDEN, C_IN), C_IN ** -0.5); wo = t((C_IN, HIDDEN), HIDDEN ** -0.5)
    ln_w = (1.0 + 0.1 * torch.from_numpy(rs.standard_normal(C_IN).astype(np.float32))).to(device)
    ln_b = (0.05 * torch.from_numpy(rs.standard_normal(C_IN).astype(np.float32))).to(device)
    return z, wa, wb, wo, ln_w, ln_b


def build_module(wa, wb, wo, ln_w, ln_b, device):
    from protenix.model.modules.primitives import Transition
    mod = Transition(c_in=C_IN, n=HIDDEN // C_IN).to(device).eval()
    with torch.no_grad():
        mod.linear_no_bias_a.weight.copy_(wa.float()); mod.linear_no_bias_b.weight.copy_(wb.float()); mod.linear_no_bias.weight.copy_(wo.float())
        mod.layernorm1.weight.copy_(ln_w); mod.layernorm1.bias.copy_(ln_b)
    return mod


def digest(t):
    b = t.detach().contiguous().view(torch.int16).cpu().numpy().tobytes()
    return hashlib.sha256(b).hexdigest(), b


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); args = ap.parse_args()
    device = torch.device("cuda")
    vpath = os.path.join(HERE, "vectors.json")
    if args.write:
        out = {"cell": {"c_in": C_IN, "hidden": HIDDEN, "dtype": "bf16"}, "generator": "numpy.random.RandomState(seed).standard_normal, float32, scaled, cast to bf16 (see make_case)",
               "reference": {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)}, "cases": []}
        for case in CASES:
            z, wa, wb, wo, ln_w, ln_b = make_case(case, device); mod = build_module(wa, wb, wo, ln_w, ln_b, device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                u = mod(z); r = z + u                                              # the stock statement under bf16 autocast
            hr, br = digest(r); hu, _ = digest(u)
            rec = dict(case, sha256_residual=hr, sha256_update=hu, layernorm=type(mod.layernorm1).__name__)
            if case["store_bytes"]:
                fn = f"expected_{case['name']}.bf16"; open(os.path.join(HERE, fn), "wb").write(br); rec["residual_bytes_file"] = fn
            out["cases"].append(rec); print(f"[vectors] wrote {case['name']}: residual {hr[:16]} update {hu[:16]}", flush=True)
        json.dump(out, open(vpath, "w"), indent=1); return 0
    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))                  # third_party/ -> the unit is importable
    import protenix_fpf_flash_transition as U
    ref = json.load(open(vpath)); bad = 0
    for ln in (False, True):
        U._STATE["installed"] = False; st = U.install(verbose=False, ln=ln)
        for case in ref["cases"]:
            z, wa, wb, wo, ln_w, ln_b = make_case(case, device); mod = build_module(wa, wb, wo, ln_w, ln_b, device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                r = U.fn_residual(mod, z); u = U.fn(mod, z)
            hr, br = digest(r); hu, _ = digest(u)
            ok = hr == case["sha256_residual"] and hu == case["sha256_update"]
            if case.get("residual_bytes_file"):
                ok = ok and open(os.path.join(HERE, case["residual_bytes_file"]), "rb").read() == br
            bad += 0 if ok else 1
            print(f"[vectors] ln={'fused' if ln else 'module'} {case['name']:>6} rows={case['rows']:<5} {'OK' if ok else 'MISMATCH'}  residual {hr[:16]} (expected {case['sha256_residual'][:16]})  update {hu[:16]} (expected {case['sha256_update'][:16]})", flush=True)
    print(f"[vectors] {'ALL OK' if bad == 0 else str(bad) + ' MISMATCH(ES)'}; unit routes {U.report()['routes']}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
