"""The descriptor cells (K1 impl 'tma2', K3 impl 'tma': kernels/kdesc.py) write the SAME BYTES as the cells they replace (K1 'tma' = kernels._k1t,
K3 pointer = kernels._k3c, the same tile numbers) — GPU, cc >= 9.0 with the tensor-descriptor API.
    python -m opt_core.kernels.fpf_trimul_v4.tests.test_cells_eq [--sizes 128,200,257,384,512,1023] [--acts 'GLOB'] [--large]
Cases: seeded synthetic z [B, N, N, 128] bf16 with per-channel offsets (B in {1, 2} up to N 384), masks {none, 7 % right padding, random p=0.9, shared [N, N]},
both directions, residual on / off, pad 16; ``--acts``: recorded activation files (torch.save'd dicts holding 'z' [1, N, N, 128] and optionally 'pair_mask'),
every file x {its mask, no mask} x both directions x residual; ``--large`` adds N 2000.  Prints one line per case and RESULT EQUAL | DIFFERENT (exit 1)."""
import argparse, glob, sys
import torch

from .. import kernels as K, kdesc as KD, generic as G

SERVED_GATE = KD.N_MIN_DESC                                     # the plane-extent gate in service; main() compares the KERNELS from their own minimum (K1's BM = 128)

NEW = {"k1": {"impl": "tma2", "BM": 128, "BN": 32, "num_warps": 4, "num_stages": 3}, "k3": {"impl": "tma", "BM": 64, "BN": 64, "num_warps": 4, "num_stages": 1}, "overrides": {}}
OLD = ({"k1": {"impl": "tma", "BM": 128, "BN": 32, "num_warps": 4, "num_stages": 3}, "k3": {"BM": 64, "BN": 64, "num_warps": 4, "num_stages": 1}, "overrides": {}},   # the 9.0|3.6 cells replaced
       {"k1": {"impl": "tma", "BM": 64, "BN": 32, "num_warps": 4, "num_stages": 3}, "k3": {"BM": 64, "BN": 64, "num_warps": 4, "num_stages": 1}, "overrides": {}})    # the 9.0|3.7 K1 cell replaced


def weights(dev, C=128, D=128, seed=0):
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    r = lambda *s, scale=1.0: (torch.randn(*s, generator=g) * scale).to(dev)
    return G.pack_weights(ln_in_w=1 + 0.1 * r(C), ln_in_b=0.1 * r(C), ln_out_w=1 + 0.1 * r(D), ln_out_b=0.1 * r(D),
                          w_ag=r(D, C, scale=C ** -0.5), w_ap=r(D, C, scale=C ** -0.5), w_bg=r(D, C, scale=C ** -0.5), w_bp=r(D, C, scale=C ** -0.5),
                          w_o=r(C, D, scale=D ** -0.5), w_og=r(C, C, scale=C ** -0.5))


def synthetic(dev, N, B, mask_kind, seed, C=128):
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    z = (torch.randn(B, N, N, C, generator=g) * (1 + torch.rand(C, generator=g)) + 2 * torch.randn(C, generator=g)).to(dev).to(torch.bfloat16)
    if mask_kind == "none":
        return z, None
    if mask_kind == "pad":
        m = torch.ones(B, N, N); k = max(1, int(0.07 * N)); m[:, -k:, :] = 0; m[:, :, -k:] = 0
    elif mask_kind == "rand":
        m = (torch.rand(B, N, N, generator=g) < 0.9).float()
    else:                                                        # 'shared': one [N, N] mask for every b
        m = (torch.rand(N, N, generator=g) < 0.9).float()
    return z, m.to(dev)


def one(tag, z, mask, w, fails):
    for outgoing in (True, False):
        for residual in (False, True):
            a = K.trimul_v4_forward(z, outgoing, mask, w, NEW, residual=residual)
            outs = [K.trimul_v4_forward(z, outgoing, mask, w, old, residual=residual) for old in OLD]
            torch.cuda.synchronize()
            for k, b in enumerate(outs):
                eq = torch.equal(a, b)
                line = "%s %s %s residual=%d vs old%d" % ("EQUAL" if eq else "DIFF ", tag, "out" if outgoing else "in ", int(residual), k)
                if not eq:
                    line += "  n_diff=%d max|diff|=%.3g" % (int((a != b).sum()), float((a.float() - b.float()).abs().max())); fails.append(line)
                print(line, flush=True)
            del a, outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="128,200,257,384,512,1023"); ap.add_argument("--acts", default=None); ap.add_argument("--large", action="store_true")
    a = ap.parse_args()
    KD.N_MIN_DESC = 128                                         # every size below the served gate still exercises the descriptor kernels in this test
    dev = torch.device("cuda")
    import triton
    print("env torch %s triton %s device %s cc %s HAS_TD %s" % (torch.__version__, triton.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_capability(), KD.HAS_TD), flush=True)
    if not (KD.HAS_TD and K.HAS_DESC and torch.cuda.get_device_capability()[0] >= 9):
        print("RESULT SKIPPED (the descriptor cells do not build on this stack)"); sys.exit(0)
    w = weights(dev); fails = []
    sizes = [int(s) for s in a.sizes.split(",")] + ([2000] if a.large else [])
    for N in sizes:
        for B in ((1, 2) if N <= 384 else (1,)):
            for mk in ("none", "pad", "rand", "shared"):
                z, m = synthetic(dev, N, B, mk, seed=N + B)
                one("synthetic:N%d:B%d:%s" % (N, B, mk), z, m, w, fails)
                del z, m
        torch.cuda.empty_cache()
    for p in sorted(glob.glob(a.acts)) if a.acts else ():
        d = torch.load(p, map_location="cpu", weights_only=False)
        z = d["z"].to(dev).to(torch.bfloat16).reshape(-1, *d["z"].shape[-3:]).contiguous()
        N = z.shape[1]
        mk = [k for k in d if k != "z" and k != "meta" and torch.is_tensor(d[k]) and tuple(d[k].shape[-2:]) == (N, N)]
        m = d[mk[0]].to(dev).float().reshape(-1, N, N).contiguous() if mk else None
        for tagm, mm in (("mask", m), ("nomask", None)):
            one("acts:%s:%s" % (p.rsplit("/", 1)[-1][:60], tagm), z, mm, w, fails)
        del d, z, m; torch.cuda.empty_cache()
    print("served gate: plane extent Np >= %d take the descriptor cells (kdesc.N_MIN_DESC); compared here from Np >= 128" % SERVED_GATE, flush=True)
    print("RESULT", "EQUAL" if not fails else "DIFFERENT (%d cases)" % len(fails), flush=True)
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
