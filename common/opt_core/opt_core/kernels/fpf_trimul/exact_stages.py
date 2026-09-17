#!/usr/bin/env python3
"""exact_stages.py — are OUR stages bit-exact to the STOCK cuEq stages when fed the stock inputs?  Real dumps + real block weights.
Stages: S_in (our _k_ln_rows vs layer_norm_transpose bijd->bijd), A' (our _k_proj6 on STOCK x_in vs fused_sigmoid_gated_dual_gemm transpose_out),
        B (cuBLAS bmm on XS=N planes vs stock einsum), S_out (our _k_ln_cols_T on STOCK X vs layer_norm_transpose dbij->bijd), C' (our _k_out6 on STOCK x_in/x_out vs dual_x GEMM),
        full EXACT-mode pipeline (pad 0, cuBLAS) vs stock full call.  Also times the pad-0 / pad-8 / pad-64 full pipelines."""
import os, sys, math, torch, triton
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpf_trimul import kernels as K
from cuequivariance_ops_torch.fused_layer_norm_torch import layer_norm_transpose
from cuequivariance_ops_torch.gated_gemm_torch import fused_sigmoid_gated_dual_gemm, fused_sigmoid_gated_dual_gemm_dual_x
from cuequivariance_torch import triangle_multiplicative_update as CUEQ
dumps = os.environ.get("FPF_DUMPS", "dumps"); wdir = sys.argv[1] if len(sys.argv) > 1 else "out"
sizes = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "356,705").split(",")]
dev = "cuda"; C = 256; DT = torch.bfloat16
def md(a, b): return float((a.double() - b.double()).abs().max())
def bw(a, b): return bool(torch.equal(a, b))
def evms(fn, reps=20, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True); s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    t = torch.tensor(ts); return float(t.mean()), float(t.std())
for N in sizes:
    d = torch.load(f"{dumps}/acts_{N}tok.pt", map_location="cpu")
    for key, blk in (("pf_c1_b0", 0), ("pf_c10_b47", 47)):
        z = d[key]["z"].to(dev).to(DT).contiguous(); M = N * N
        for direction in ("outgoing", "incoming"):
            W = torch.load(f"{wdir}/ptx_trimul_{'out' if direction == 'outgoing' else 'in'}_b{blk}_bf16.pt", map_location=dev)
            WP = K.pack_weights(W["ln_in_w"], W["ln_in_b"], W["p_in"], W["g_in"], W["ln_out_w"], W["ln_out_b"], W["p_out"], W["g_out"])
            mask = torch.ones(N, N, device=dev, dtype=DT)
            with torch.no_grad():
                x4 = z[None]
                xin_s = layer_norm_transpose(x4, W["ln_in_w"], W["ln_in_b"], eps=1e-5, layout="bijd->bijd")            # [1,N,N,C]
                ab_s = fused_sigmoid_gated_dual_gemm(xin_s, W["g_in"], W["p_in"], mask[None], transpose_out=True)          # [2C,1,N,N]
                a_s, b_s = torch.chunk(ab_s, 2, dim=0)
                x_s = torch.einsum("dbik,dbjk->dbij", a_s, b_s) if direction == "outgoing" else torch.einsum("dbki,dbkj->dbij", a_s, b_s)
                xo_s = layer_norm_transpose(x_s, W["ln_out_w"], W["ln_out_b"], eps=1e-5, layout="dbij->bijd")              # [1,N,N,C]
                out_s = fused_sigmoid_gated_dual_gemm_dual_x(xin_s, xo_s, W["g_out"], W["p_out"])[0]                       # [N,N,C]
                full_s = CUEQ(x4, direction=direction, mask=mask[None], norm_in_weight=W["ln_in_w"], norm_in_bias=W["ln_in_b"], p_in_weight=W["p_in"], g_in_weight=W["g_in"],
                              norm_out_weight=W["ln_out_w"], norm_out_bias=W["ln_out_b"], p_out_weight=W["p_out"], g_out_weight=W["g_out"], eps=1e-5)[0]
                res_s = full_s + z
                # ---- ours, stage by stage on STOCK inputs (XS = N planes)
                xin_o = torch.empty(M, C, dtype=DT, device=dev)
                K._k_ln_rows[(triton.cdiv(M, 64),)](z, WP["ln_in_w"], WP["ln_in_b"], xin_o, M, C, 1e-5, BM=64, CK=C, num_warps=4)
                r_sin = (bw(xin_o, xin_s.reshape(M, C)), md(xin_o, xin_s.reshape(M, C)))
                ab_o = torch.zeros(2 * C, N, N, dtype=DT, device=dev)
                nkt = triton.cdiv(N, 128)
                K._k_proj6[((2 * C) // 64, N * nkt)](xin_s.reshape(M, C).contiguous(), WP["p_in"], WP["g_in"], mask.reshape(-1), ab_o, N, N, C, nkt, HAS_MASK=True, PREC=0, BM=128, BN=64, BK=64, NK=4, num_warps=4, num_stages=3)
                r_a = (bw(ab_o, ab_s[:, 0]), md(ab_o, ab_s[:, 0]))
                x_o = torch.empty(C, N, N, dtype=DT, device=dev)
                if direction == "outgoing": torch.matmul(a_s[:, 0], b_s[:, 0].transpose(-1, -2), out=x_o)
                else: torch.matmul(a_s[:, 0].transpose(-1, -2), b_s[:, 0], out=x_o)
                r_b = (bw(x_o, x_s[:, 0]), md(x_o, x_s[:, 0]))
                xo_o = torch.empty(M, C, dtype=DT, device=dev)
                K._k_ln_cols_T[(N * triton.cdiv(N, 64),)](x_s[:, 0].contiguous(), WP["ln_out_w"], WP["ln_out_b"], xo_o, N, N, C, triton.cdiv(N, 64), 1e-5, BM=64, CK=C, num_warps=4)
                r_sout = (bw(xo_o, xo_s.reshape(M, C)), md(xo_o, xo_s.reshape(M, C)))
                out_o = torch.empty(N, N, C, dtype=DT, device=dev)
                K._k_out6[(C // 128, triton.cdiv(M, 64))](xo_s.reshape(M, C).contiguous(), xin_s.reshape(M, C).contiguous(), z, K._wT(WP, "p_out"), K._wT(WP, "g_out"), out_o, M, C, RESIDUAL=False, PREC=0, BM=64, BN=128, BK=64, NK=4, num_warps=4, num_stages=3)
                r_c = (bw(out_o, out_s), md(out_o, out_s))
                # ---- full pipelines: pad 0 (EXACT candidate), pad 8, pad 64; cuBLAS contraction
                cfg0 = dict(A=dict(v=6, BM=128, BN=64, BK=64, num_warps=4, num_stages=3), B=dict(BM=128, BN=128, BK=64, num_warps=8, num_stages=3, DMAJOR=True), C=dict(v=6, BM=64, BN=128, BK=64, num_warps=4, num_stages=3), LN=dict(BM=64, num_warps=4), LNO=dict(BM=64, num_warps=4))
                outs = {}
                for pad in (1, 8, 64):
                    cfg = dict(cfg0, pad=pad, contract="cublas")
                    f = lambda: K.trimul_forward(z, direction, mask, WP, eps=1e-5, residual=True, cfg=cfg, contract="cublas", key=f"ex{pad}")
                    o = f(); torch.cuda.synchronize(); ms, sd = evms(f)
                    outs[pad] = (o, ms, sd)
                fs = lambda: (CUEQ(x4, direction=direction, mask=mask[None], norm_in_weight=W["ln_in_w"], norm_in_bias=W["ln_in_b"], p_in_weight=W["p_in"], g_in_weight=W["g_in"], norm_out_weight=W["ln_out_w"], norm_out_bias=W["ln_out_b"], p_out_weight=W["p_out"], g_out_weight=W["g_out"], eps=1e-5)[0] + z)
                ms_s, sd_s = evms(fs)
                print(f"EXACT N={N} {key} {direction}: S_in bitwise={r_sin[0]} (max|d| {r_sin[1]:.3g}) | A' bitwise={r_a[0]} ({r_a[1]:.3g}) | B(cuBLAS XS=N) bitwise={r_b[0]} ({r_b[1]:.3g}) | S_out bitwise={r_sout[0]} ({r_sout[1]:.3g}) | C' bitwise={r_c[0]} ({r_c[1]:.3g})")
                print(f"FULL  N={N} {key} {direction}: stock {ms_s:.4f}+-{sd_s:.4f} ms | pad0 {outs[1][1]:.4f} ms bitwise_vs_stock={bw(outs[1][0], res_s)} max|d| {md(outs[1][0], res_s):.3g} | pad8 {outs[8][1]:.4f} ms bitwise={bw(outs[8][0], res_s)} max|d| {md(outs[8][0], res_s):.3g} | pad64 {outs[64][1]:.4f} ms bitwise={bw(outs[64][0], res_s)} max|d| {md(outs[64][0], res_s):.3g}", flush=True)
            K._BUF.clear(); torch.cuda.empty_cache()
print("EXACT DONE")
