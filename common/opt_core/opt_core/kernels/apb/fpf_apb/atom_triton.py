"""atom_triton.py — local-window attention with pair bias for the AtomTransformer (Protenix v2 atom encoder/decoder), one launch per call.

Stock (primitives._local_attention / rearrange_to_dense_trunk): query trunk t = rows 32t..32t+31, keys j in [32t - 48, 32t + 80), out-of-range
keys masked (-inf), bias `trunked_attn_bias` [H, n_trunks, 32, 128] added, fp32 softmax, output rows >= N dropped. Here the windows are
addressed in-kernel from N_atom (no padded / unfolded copies of q, k, v, no mask tensor), the bias is read once per (trunk, head) program,
q/k/v/g are [S, N, H, D]-addressable strided views (unit stride on D), o [S, N, H, D]; optional fused gate o *= sigmoid(g).
Operands: fp32 in -> "tf32rn" (operands rounded to nearest TF32 then a TF32 dot: the cuBLAS TF32 class, default) | "tf32x3" (fp32-faithful) | "tf32" (truncating) | in-kernel "bf16"; 16-bit inputs use their dtype. fp32 softmax and
accumulation always. Numerics class: TOLERANCE (stock's QK^T / PV run as TF32 cuBLAS GEMMs; the reduction order differs).
"""
from __future__ import annotations
import math
import torch
import triton
import triton.language as tl
from .apb_triton import _opd_code, _cast_opd, LAST_LAUNCH

_LOG2E = 1.4426950408889634


@triton.jit(do_not_specialize=["N", "c_qk", "c_b"])
def _atom_fwd(Q, K, V, G, B, O,
              sqs, sqn, sqh, sks, skn, skh, svs, svn, svh, sgs, sgn, sgh,
              sbh, sbt, sbq,
              sos, son, soh,
              N, c_qk, c_b,
              D: tl.constexpr, NQ: tl.constexpr, NK: tl.constexpr, PADL: tl.constexpr,
              GATE: tl.constexpr, OUT_F32: tl.constexpr, PREC: tl.constexpr, OPD: tl.constexpr):
    t = tl.program_id(0)
    s_idx = tl.program_id(1)
    h = tl.program_id(2)
    offs_q = t * NQ + tl.arange(0, NQ)
    offs_k = t * NQ - PADL + tl.arange(0, NK)
    offs_d = tl.arange(0, D)
    qmask = offs_q < N
    kmask = (offs_k >= 0) & (offs_k < N)
    q = tl.load(Q + s_idx * sqs + h * sqh + offs_q[:, None] * sqn + offs_d[None, :], mask=qmask[:, None], other=0.0)
    k = tl.load(K + s_idx * sks + h * skh + offs_k[:, None] * skn + offs_d[None, :], mask=kmask[:, None], other=0.0)
    v = tl.load(V + s_idx * svs + h * svh + offs_k[:, None] * svn + offs_d[None, :], mask=kmask[:, None], other=0.0)
    q = _cast_opd(q, OPD); k = _cast_opd(k, OPD); v = _cast_opd(v, OPD)
    bias = tl.load(B + h * sbh + t * sbt + tl.arange(0, NQ)[:, None] * sbq + tl.arange(0, NK)[None, :]).to(tl.float32)
    sc = tl.dot(q, tl.trans(k), input_precision=PREC) * c_qk + bias * c_b
    sc = tl.where(kmask[None, :], sc, -float("inf"))
    m = tl.max(sc, 1)
    p = tl.exp2(sc - m[:, None])
    l = tl.sum(p, 1)
    o = tl.dot(_cast_opd(p.to(v.dtype), OPD), v, input_precision=PREC) * (1.0 / l)[:, None]
    if GATE:
        g = tl.load(G + s_idx * sgs + h * sgh + offs_q[:, None] * sgn + offs_d[None, :], mask=qmask[:, None], other=0.0).to(tl.float32)
        o = o * (1.0 / (1.0 + tl.exp2(-g * 1.4426950408889634)))
    optr = O + s_idx * sos + h * soh + offs_q[:, None] * son + offs_d[None, :]
    if OUT_F32:
        tl.store(optr, o, mask=qmask[:, None])
    else:
        tl.store(optr, o.to(O.dtype.element_ty), mask=qmask[:, None])


def atom_apb(q, k, v, bias, g=None, *, n_queries=32, n_keys=128, scale=None, out=None, out_dtype=None, opd: str | None = None, num_warps=4):
    """q,k,v(,g): [S, N, H, D] strided views (unit stride on D); bias: [H, n_trunks, n_queries, n_keys] (or with leading 1s), unit stride last.
    Returns o [S, N, H, D] (out_dtype, default q.dtype): stock's local_cross_attention result (gated if g is given)."""
    assert q.dim() == 4 and k.shape == q.shape and v.shape == q.shape
    S, N, H, D = q.shape
    while bias.dim() > 4:
        assert bias.shape[0] == 1, bias.shape
        bias = bias[0]
    T = triton.cdiv(N, n_queries)
    assert bias.shape == (H, T, n_queries, n_keys) and bias.stride(3) == 1, (bias.shape, bias.stride(), (H, T, n_queries, n_keys))
    for t_ in (q, k, v) + ((g,) if g is not None else ()):
        assert t_.stride(3) == 1 and t_.dtype == q.dtype
    opd_code, opd_name, prec = _opd_code(q.dtype, opd or ("tf32rn" if q.dtype == torch.float32 else None))
    out_dtype = out_dtype or q.dtype
    if out is None:
        out = torch.empty((S, N, H, D), device=q.device, dtype=out_dtype)
    scale = (1.0 / math.sqrt(D)) if scale is None else float(scale)
    gg = g if g is not None else q
    grid = (T, S, H)
    _atom_fwd[grid](q, k, v, gg, bias, out,
                    q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2),
                    gg.stride(0), gg.stride(1), gg.stride(2),
                    bias.stride(0), bias.stride(1), bias.stride(2),
                    out.stride(0), out.stride(1), out.stride(2),
                    N, scale * _LOG2E, _LOG2E,
                    D=D, NQ=n_queries, NK=n_keys, PADL=(n_keys - n_queries) // 2,
                    GATE=g is not None, OUT_F32=out_dtype == torch.float32, PREC=prec, OPD=opd_code, num_warps=num_warps, num_stages=1)
    LAST_LAUNCH.update(kernel="_atom_fwd", grid=grid, in_dtype=str(q.dtype), opd=opd_name, prec=prec, bias_dtype=str(bias.dtype))
    return out
