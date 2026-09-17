"""opt_core.ops.msa_pwa2 — the MSA PairWeightedAveraging (c_m=64, c_z=128, c_h=32, 8 heads; attribute schema norm_m / proj_m / proj_g / norm_z / proj_z / proj_o /
inf / num_heads / c_h) as a fused EXACT cell (an exact-tier lever: bitwise = the stock statements) — and the same kernels can serve the fast class.

Stock, per call (chunk_heads path, N > 384): LN(m), LN(z); then PER HEAD h: v_h = m @ Wm_h.T (bf16), b_h = z @ Wz_h.T, + (1-mask)*-inf, softmax (fp32),
g_h = sigmoid(m @ Wg_h.T) (bf16), o_h = einsum('bhij,bhsjd->bhsid', w_h, v_h) (cuBLAS bmm, K = N, bf16 out), og = g_h*o_h (bf16),
o_out (+)= og @ Wo_h.T (bf16 chain) — ~25 passes over [S,N,*] tensors and 8 bmm calls whose B operand is re-laid-out by einsum each time.

This cell: the two LayerNorms and the per-head b/softmax stay the STOCK STATEMENTS (torch; cheap, and torch's LN/softmax are not reproduced in-kernel);
everything on the [S,N,*] side is two Triton kernels:
  _vg_kernel    one pass over LN(m): autocast's RNE cast, v = x16 @ WmT and g = sigmoid(bf16(x16 @ WgT)) for all heads (tl.dot columns are independent of
                the N tiling, so the full-width dot equals the per-head slice GEMMs column for column; each equals cuBLAS's slice GEMM at every M —
                the GEMM bit-equality note), sigmoid = bf16(1.0f/(1.0f+expf(-x))) (exhaustively == ATen); V, G stored [S*N, 256].
  _wv_kernel    per (i-tile, s-tile, head) CTA: the K = N contraction o_h[i,(s,c)] = sum_j w16_h[i,j] V_h[j,(s,c)] with fp32 accumulation in
                the k-GROUPING cuBLAS uses for this class (KMODE 0/1: ascending k16 groups after a residue-first partial tile of width R = 0 | N mod 32 |
                N mod 64 — nvjet and cutlass_80 s16816 align2 kernels; KMODE 2: k8 emulation — each k16 MMA issued twice with the low/high 8 k-lanes zeroed —
                after a residue of N mod 32: cutlass_75 s1688 align1 kernels; dev tool probe_korder.py), -> bf16 -> x g (bf16) -> OG [S*N, 256].
  _po_kernel    the proj_o: per head the slice dot (K = 32) -> bf16 -> the bf16 running sum across heads (EPI 0 = the chunked path's `o_out += ...` chain)
                or one fp32 accumulator over all heads = the unchunked path's single K = 256 proj_o GEMM (EPI 1) -> one store of o_out.
Which KMODE/R serves a (N, S, path) class is decided empirically by the engine adapter (a run-time bit-compare lock): at the class's first eager encounter the
stock statements run once and the candidates (candidates(N)) are compared torch.equal; the match is locked for the process, no match = the class is served
by the stock forward BY NAME. Nothing here is bitwise by assumption.

VOUCH ENVELOPE (measured; the adapter's exact-tier binding follows it BY NAME): stack torch 2.12 / CUDA 13.0 / triton 3.7, bf16 autocast, eval.
  cc 9.0 (H100): every (N, S, path) class exercised locks a candidate (unit test classes N 61..1292, S 64..2048, both paths, bf16 and fp32 m;
                 in-model 400 / 800 / 1200-token inputs: every class proven, every output file of the exact mode byte-identical to the stock run).
  cc 8.0 (A100): the candidate set is cuBLAS's sm_90 summation structures; of three in-model classes ONE proves (800 tokens x S 7382, chunked);
                 400 x 7311 and 1200 x 7410 (chunked) have NO equal candidate -> the lock serves the stock statements by name (correct, idle).
                 An exact-tier adapter therefore pins this cell to cc 9.0 (VOUCHED_CC) or relies on the lock, which never returns unequal bytes.
  other cards:   unmeasured -> the adapter's pin keeps the stock statements by name.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import triton
import triton.language as tl

try:
    from triton.language.extra import libdevice as _ld
except Exception:                                             # older layouts
    from triton.language.extra.cuda import libdevice as _ld

C_M, C_Z, C_H, HEADS, HC = 64, 128, 32, 8, 256
VOUCHED_CC = ((9, 0),)                                        # compute capabilities on which every measured class locked a candidate (the envelope above)
VOUCH_STACK = {"torch": "2.12.", "cuda": "13.0"}              # the torch / CUDA line the envelope was measured on (prefix match, as an adapter pins it)


@triton.jit(do_not_specialize=["M"])
def _vg_kernel(X, WMT, WGT, V, G, M, C: tl.constexpr, HCc: tl.constexpr, BM: tl.constexpr):
    # X: [M, C] fp32 (LN output; bf16 also loads); WMT/WGT: [C, HCc] bf16 (= proj_m.weight.T / proj_g.weight.T); V, G: [M, HCc] bf16
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BM + tl.arange(0, BM); rm = rows < M
    kk = tl.arange(0, C); nn = tl.arange(0, HCc)
    x16 = tl.load(X + rows[:, None] * C + kk[None, :], mask=rm[:, None], other=0.0).to(tl.bfloat16)   # autocast's cached_cast: RNE
    wm = tl.load(WMT + kk[:, None] * HCc + nn[None, :])
    v = tl.dot(x16, wm).to(tl.bfloat16)
    tl.store(V + rows[:, None] * HCc + nn[None, :], v, mask=rm[:, None])
    wg = tl.load(WGT + kk[:, None] * HCc + nn[None, :])
    g16 = tl.dot(x16, wg).to(tl.bfloat16).to(tl.float32)                     # g = m @ Wg.T is a bf16 tensor; ATen sigmoid computes in fp32 from it
    g = (1.0 / (1.0 + _ld.exp(-g16))).to(tl.bfloat16)                         # == ATen bf16 sigmoid, all 65,536 inputs (div.rn.f32 + libdevice expf)
    tl.store(G + rows[:, None] * HCc + nn[None, :], g, mask=rm[:, None])


@triton.jit(do_not_specialize=["N", "S", "R"])
def _wv_kernel(W16, V, G, OG, N, S, R,
               CH: tl.constexpr, HCc: tl.constexpr, BI: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, KMODE: tl.constexpr):
    """Head h = program_id(2): OG[(s,i), h*CH+c] = bf16( g * bf16( sum_j w16[h,i,j] * V[(s,j), h*CH+c] ) ) for an (i-tile, s-tile); the K = N sum in
    structure KMODE/R (0: residue-first partial tile of width R then ascending full tiles — R = 0 is plain ascending; 2: the same with every k16 MMA
    issued as two k8 halves)."""
    pid_i = tl.program_id(0); pid_s = tl.program_id(1); h = tl.program_id(2)
    i0 = pid_i * BI; s0 = pid_s * BS
    ii = i0 + tl.arange(0, BI); im = ii < N
    q = tl.arange(0, BS * CH); s_q = s0 + q // CH; c_q = q % CH; qm = s_q < S
    kk = tl.arange(0, BK)
    acc = tl.zeros((BI, BS * CH), dtype=tl.float32)
    wrow = W16 + (h * N + ii).to(tl.int64)[:, None] * N                       # w16[h, i, :]  (row pitch N)
    vcol = V + (s_q.to(tl.int64)[None, :] * N) * HCc + h * CH + c_q[None, :]   # + j*HCc per k row: V[(s*N + j), h*CH + c]
    for k0 in range(0, R, BK):                                                  # residue-first partial tile(s): k in [0, R)
        rk = k0 + kk
        a = tl.load(wrow + rk[None, :], mask=im[:, None] & (rk[None, :] < R), other=0.0)
        b = tl.load(vcol + rk.to(tl.int64)[:, None] * HCc, mask=(rk[:, None] < R) & qm[None, :], other=0.0)
        if KMODE == 2:
            lo = kk < 8
            acc = tl.dot(tl.where(lo[None, :], a, 0.0), b, acc)
            acc = tl.dot(tl.where(lo[None, :], 0.0, a), b, acc)
        else:
            acc = tl.dot(a, b, acc)
    for k0 in range(R, N, BK):                                                  # full tiles from R (k16 groups aligned at R), ascending
        rk = k0 + kk
        a = tl.load(wrow + rk[None, :], mask=im[:, None] & (rk[None, :] < N), other=0.0)
        b = tl.load(vcol + rk.to(tl.int64)[:, None] * HCc, mask=(rk[:, None] < N) & qm[None, :], other=0.0)
        if KMODE == 2:
            lo = kk < 8
            acc = tl.dot(tl.where(lo[None, :], a, 0.0), b, acc)
            acc = tl.dot(tl.where(lo[None, :], 0.0, a), b, acc)
        else:
            acc = tl.dot(a, b, acc)
    o = acc.to(tl.bfloat16).to(tl.float32)                                     # the einsum output is a bf16 tensor
    goff = (s_q.to(tl.int64)[None, :] * N + ii[:, None]) * HCc + h * CH + c_q[None, :]
    gm = im[:, None] & qm[None, :]
    g = tl.load(G + goff, mask=gm, other=0.0).to(tl.float32)
    tl.store(OG + goff, (g * o).to(tl.bfloat16), mask=gm)                     # o_chunks = g * o  (bf16 * bf16 -> bf16)


@triton.jit(do_not_specialize=["M"])
def _po_kernel(OG, WOT, OUT, M, H: tl.constexpr, CH: tl.constexpr, HCc: tl.constexpr, CM: tl.constexpr, BM: tl.constexpr, EPI: tl.constexpr):
    """OUT[r, :] = the proj_o of OG[r, :]: EPI 0 = the chunked path's per-head bf16 chain `o_out (+)= og_h @ Wo_h.T`; EPI 1 = the unchunked path's one
    K=256 GEMM (fp32 accumulation over heads in order, one rounding)."""
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BM + tl.arange(0, BM); rm = rows < M
    cc = tl.arange(0, CH); ee = tl.arange(0, CM)
    og = tl.load(OG + rows[:, None] * HCc + cc[None, :], mask=rm[:, None], other=0.0)
    wo = tl.load(WOT + cc[:, None] * CM + ee[None, :])
    if EPI == 0:
        o16 = tl.dot(og, wo).to(tl.bfloat16)
        for h in range(1, H):
            og = tl.load(OG + rows[:, None] * HCc + h * CH + cc[None, :], mask=rm[:, None], other=0.0)
            wo = tl.load(WOT + (h * CH + cc)[:, None] * CM + ee[None, :])
            o16 = (o16.to(tl.float32) + tl.dot(og, wo).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    else:
        acc = tl.dot(og, wo)
        for h in range(1, H):
            og = tl.load(OG + rows[:, None] * HCc + h * CH + cc[None, :], mask=rm[:, None], other=0.0)
            wo = tl.load(WOT + (h * CH + cc)[:, None] * CM + ee[None, :])
            acc = tl.dot(og, wo, acc)
        o16 = acc.to(tl.bfloat16)
    tl.store(OUT + rows[:, None] * CM + ee[None, :], o16, mask=rm[:, None])


def _weights(mod) -> Dict[str, torch.Tensor]:
    key = tuple((p.data_ptr(), p._version) for p in (mod.proj_m.weight, mod.proj_g.weight, mod.proj_o.weight))
    w = getattr(mod, "_msa2_pwa2_w", None)
    if w is None or w["key"] != key:
        with torch.no_grad():
            w = {"key": key,
                 "wmT": mod.proj_m.weight.detach().to(torch.bfloat16).t().contiguous(),     # [64, 256]  (autocast casts the weight slice: same bf16 values)
                 "wgT": mod.proj_g.weight.detach().to(torch.bfloat16).t().contiguous(),
                 "woT": mod.proj_o.weight.detach().to(torch.bfloat16).t().contiguous()}     # [256, 64]
        mod._msa2_pwa2_w = w
    return w


def supported(mod, m, z, mask, chunk_heads) -> str | None:
    """None if this cell can serve the call, else the fallback word."""
    if getattr(mod, "num_heads", None) != HEADS or getattr(mod, "c_h", None) != C_H or mod.proj_m.weight.shape != (HC, C_M) or mod.proj_z.weight.shape != (HEADS, C_Z) \
            or mod.proj_o.weight.shape != (C_M, HC):
        return "dims"
    if m.dim() != 4 or z.dim() != 4 or m.shape[0] != 1 or z.shape[0] != 1:
        return "batch"
    if m.shape[-1] != C_M or z.shape[-1] != C_Z or m.shape[2] != z.shape[1] or z.shape[1] != z.shape[2] or tuple(mask.shape) != (1, z.shape[1], z.shape[2]):
        return "shape"
    if not (m.is_cuda and z.is_cuda):
        return "device"
    if m.dtype not in (torch.bfloat16, torch.float32) or z.dtype not in (torch.bfloat16, torch.float32):
        return "dtype"
    if m.shape[1] < 1 or m.shape[2] < 16:
        return "tiny"
    return None


def candidates(N: int) -> List[Tuple[int, int]]:
    """(KMODE, R) summation structures cuBLAS may use for the K = N einsum, most likely first (dev tools probe_korder / probe_einsum_bands, cc 9.0 torch 2.12)."""
    if N % 8 == 0:
        c = [(0, 0), (1, N % 64), (1, N % 32)]                 # nvjet (16-B aligned rows): ascending full k16 groups
    elif N % 2 == 0:
        c = [(1, N % 32), (1, N % 64), (0, 0)]                 # cutlass_80 s16816 align2: residue-first, BK 32 or 64 by kernel
    else:
        c = [(2, N % 32), (2, N % 64), (0, 0)]                 # cutlass_75 s1688 align1: k8 MMAs, residue-first (BK 32)
    out: List[Tuple[int, int]] = []
    for k in c:
        k = (0, 0) if (k[0] == 1 and k[1] == 0) else k
        if k not in out:
            out.append(k)
    return out


def signature(kmode: int, R: int):
    """The arithmetic identity of a candidate = its k-grouping: (granule g, residue r). kmode 0/1 use k16 MMA groups (g = 16), kmode 2 k8 groups (g = 8);
    a residue-first partial tile of width R changes the grouping only when R is not a multiple of g (R = 48 under k16 groups is the ascending order:
    the same k16 boundaries, zero products add exactly 0)."""
    g = 8 if kmode == 2 else 16
    return (g, 0 if R % g == 0 else int(R))


def distinct_candidates(N: int):
    """candidates(N) de-duplicated by signature(), order kept (the first spelling of each grouping)."""
    seen, out = set(), []
    for km, R in candidates(N):
        sg = signature(km, R)
        if sg not in seen:
            seen.add(sg); out.append((km, R))
    return out


def pwa_forward(mod, m: torch.Tensor, z: torch.Tensor, mask: torch.Tensor, chunk_heads: bool, kmode: int = 0, R: int = 0,
                BI: int = 64, BS: int = 8, num_warps: int = 4, num_stages: int = 3) -> torch.Tensor:
    """PairWeightedAveraging.forward(m, z, mask, chunk_heads) in eval under CUDA autocast-bf16 -> bf16 [1, S, N, 64], with the K=N contraction in
    summation structure (kmode, R). Bitwise = stock when (kmode, R) is the class's bit-compared candidate (the adapter decides)."""
    m = mod.norm_m(m)                                          # stock statements (fp32 under autocast)
    z = mod.norm_z(z)
    B, S, N, _ = m.shape
    W = _weights(mod)
    x = m.reshape(S * N, C_M)
    if not x.is_contiguous():
        x = x.contiguous()
    Mrows = S * N
    V = torch.empty((Mrows, HC), device=m.device, dtype=torch.bfloat16); Gt = torch.empty_like(V)
    BMv = 64
    _vg_kernel[(triton.cdiv(Mrows, BMv),)](x, W["wmT"], W["wgT"], V, Gt, Mrows, C=C_M, HCc=HC, BM=BMv, num_warps=8, num_stages=2)
    z16 = z.to(torch.bfloat16)                                 # autocast's cast of LN(z), once (stock re-casts it per head: same values)
    negm = (1 - mask[:, None]) * -mod.inf                      # stock expression, [1,1,N,N] fp32
    w16 = torch.empty((HEADS, N, N), device=m.device, dtype=torch.bfloat16)
    unchunked = not (chunk_heads and not mod.training)
    if not unchunked:
        for h in range(HEADS):
            b = z16 @ mod.proj_z.weight[h:h + 1, :].T           # = `z @ sliced_weight_proj_z.T` under autocast (bf16 [1,N,N,1])
            b = b.permute(0, 3, 1, 2)
            b = b + negm
            w = torch.softmax(b, dim=-1)                       # fp32 (autocast)
            w16[h].copy_(w[0, 0])                              # einsum's autocast cast of w: RNE
    else:
        b = mod.proj_z(z16)                                    # = self.proj_z(z) under autocast
        b = b.permute(0, 3, 1, 2)
        b = b + negm
        w = torch.softmax(b, dim=-1)
        w16.copy_(w[0])
    OG = torch.empty((Mrows, HC), device=m.device, dtype=torch.bfloat16)
    BK = 16 if kmode == 2 else 64
    _wv_kernel[(triton.cdiv(N, BI), triton.cdiv(S, BS), HEADS)](w16, V, Gt, OG, N, S, int(R), CH=C_H, HCc=HC, BI=BI, BS=BS, BK=BK, KMODE=2 if kmode == 2 else 0,
                                                                num_warps=num_warps, num_stages=num_stages)
    del V, Gt
    out = torch.empty((B, S, N, C_M), device=m.device, dtype=torch.bfloat16)
    BMo = 128
    _po_kernel[(triton.cdiv(Mrows, BMo),)](OG, W["woT"], out, Mrows, H=HEADS, CH=C_H, HCc=HC, CM=C_M, BM=BMo, EPI=1 if unchunked else 0, num_warps=4, num_stages=2)
    return out
