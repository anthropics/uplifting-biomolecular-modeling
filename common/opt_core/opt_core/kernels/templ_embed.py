"""TEMPL_EMBED v1.0 — the template embedder of the AlphaFold3-architecture engines (Algorithm 16: TemplateEmbedder) around its pair stack,
as two Triton kernels: the per-template FEATURE EMBEDDING that produces the pair stack's input, and the AGGREGATION TAIL that reduces the
stack's output over templates.

Stock statements (the OF3 family's TemplatePairEmbedderAllAtom / TemplateEmbedderAllAtom under bf16 autocast, T templates, N tokens, c_t = 64):

    embed   a[t,i,j] = W_dgram distogram[t,i,j,0:39] + w_pb (pb[t,i] pb[t,j] same_chain[i,j]) + W_aa1 restype[t,i] + W_aa2 restype[t,j]
                       + w_x ux[t,i,j] + w_y uy[t,i,j] + w_z uz[t,i,j] + w_bb (bb[t,i] bb[t,j] same_chain[i,j])         (8 bias-free Linears:
                       8 [T,N,N,64] bf16 tensors written and 7 bf16 adds); v[t,i,j] = zp[i,j] + a[t,i,j] with zp = linear_z(LayerNorm(z))
    tail    u[i,j] = relu( (sum_t s[t,i,j]) / T ) ; out[i,j] = W_t u[i,j]      (s = the template pair stack's output, bias-free linear_t)

Served:
    embed(dg, uv, pbm, bbm, asym, ri, rj, zp, P)  -> v [T, N, N, 64] bf16
        ONE kernel per call: per (template t, row i, block of BJ columns) program the 39-wide distogram tile goes through one MMA with W_dgram^T
        (bf16 operands as autocast rounds them, fp32 accumulate), the two pair masks are built in-kernel from the per-token masks and the chain
        ids, the three unit-vector components are rank-1 FMAs with their weight columns, the two restype terms arrive as per-token tables
        (ri = W_aa1 restype [T,N,64], rj = W_aa2 restype [T,N,64] — 2 T N rows instead of 2 T N^2), zp [N,N,64] is read once per template;
        everything is summed in fp32 and rounded to bf16 ONCE (stock rounds nine times — the same numerics class, more accurate, not bitwise).
    tail(s, wt)  -> out [N, N, c_z] bf16
        ONE kernel: per (row i, block of BJ columns) the T template rows are summed in fp32 in template order (a stride-0 template axis — an
        expanded [1,N,N,64] -> [T,N,N,64] view — is served in place and read once per address), rounded to bf16, divided by T and rounded
        (stock's bf16 sum then bf16 / int), relu, one MMA with W_t^T (bf16 operands, fp32 accumulate) -> bf16.  For T a power of two the sum
        and the division are exact, so the result equals the stock statements bit for bit wherever the MMA and cuBLAS reduce the K = 64 product
        alike (observed on H100 at the tested shapes); within one bf16 rounding in general.
    pack_weights(...)  -> P: the eight embedding weights and W_t in the kernels' layouts (bf16, as autocast rounds Linear weights).

Domain: T >= 1, N >= 1 (edge tiles masked), any chain layout and masks, distinct or identical templates, c_t = 64, 39 distogram bins,
c_z in {64, 128, 256} (W_t^T held in registers), CUDA bf16.  Offsets: the row / column products are int64 in-kernel and the template axis
of `tail` advances a 64-bit pointer per template — T N^2 64 exceeds 2^31 at N >= 2897 for T = 4 (tests/gpu/test_templ_embed_gpu.py holds the
high-offset region of a T=4, N=3400 problem bitwise to the same kernels run on token crops).  `check_embed` / `check_tail` raise Unsupported
BEFORE any launch for operands outside the domain; `reference_embed` / `reference_tail` are the materialised stock statements the kernels
are held to (fp32 or the autocast bf16 chain).
"""
from typing import Dict, Optional

import torch
import triton
import triton.language as tl

__version__ = "1.0"
C_T, C_DG, C_AA = 64, 39, 32                                   # template pair width, distogram bins, restype classes (AlphaFold3 Algorithm 16)
KDP = 64                                                       # the distogram reduction padded to one MMA K tile (rows 39..63 of W_dgram^T are zero)
SERVED_CZ = (64, 128, 256)
BJ_EMBED, BJ_TAIL = 64, 64
NUM_WARPS, NUM_STAGES = 4, 2


class Unsupported(Exception):
    """An operand outside the kernels' served domain, refused by name before any work: `.event` is one census word, `.detail` a sentence."""

    def __init__(self, event: str, detail: str = ""):
        super().__init__(f"{event}: {detail}" if detail else event)
        self.event, self.detail = event, detail


# ----------------------------------------------------------------------------------------------------------------------------------- kernels
@triton.jit
def _embed_kernel(DG, UV, PBM, BBM, ASYM, RI, RJ, ZP, OUT, WD, WPB, WX, WY, WZ, WBB,
                  N, s_dg_t, s_dg_i, s_dg_j, s_uv_t, s_uv_i, s_uv_j, s_m_t, s_r_t, s_r_n, s_zp_i, s_zp_j, s_o_t, s_o_i, s_o_j,
                  C: tl.constexpr, KD: tl.constexpr, KDPAD: tl.constexpr, BJ: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)                      # one (template t, row i) pair per program on axis 0 — int64: every product below is 64-bit
    jb = tl.program_id(1).to(tl.int64)
    t = pid // N
    i = pid % N
    j = jb * BJ + tl.arange(0, BJ).to(tl.int64)              # [BJ] int64 column indices
    jm = j < N
    c = tl.arange(0, C)
    k = tl.arange(0, KDPAD)
    # distogram tile [BJ, KDPAD] (fp32 one-hot features; the bf16 operand autocast would hand the Linear) @ W_dgram^T [KDPAD, C]
    dg = tl.load(DG + t * s_dg_t + i * s_dg_i + j[:, None] * s_dg_j + k[None, :], mask=jm[:, None] & (k[None, :] < KD), other=0.0)
    wd = tl.load(WD + k[:, None] * C + c[None, :])           # [KDPAD, C] bf16, rows >= KD are zero
    acc = tl.dot(dg.to(tl.bfloat16), wd)                     # [BJ, C] fp32
    # scalar features: the two pair masks from the per-token masks x same-chain, the unit-vector components (bf16-rounded operands, like autocast)
    a_i = tl.load(ASYM + i)
    a_j = tl.load(ASYM + j, mask=jm, other=0)
    same = (a_j == a_i).to(tl.float32)
    pb = tl.load(PBM + t * s_m_t + i).to(tl.float32) * tl.load(PBM + t * s_m_t + j, mask=jm, other=0.0).to(tl.float32) * same
    bb = tl.load(BBM + t * s_m_t + i).to(tl.float32) * tl.load(BBM + t * s_m_t + j, mask=jm, other=0.0).to(tl.float32) * same
    uv = UV + t * s_uv_t + i * s_uv_i + j * s_uv_j
    ux = tl.load(uv + 0, mask=jm, other=0.0).to(tl.bfloat16).to(tl.float32)
    uy = tl.load(uv + 1, mask=jm, other=0.0).to(tl.bfloat16).to(tl.float32)
    uz = tl.load(uv + 2, mask=jm, other=0.0).to(tl.bfloat16).to(tl.float32)
    wpb = tl.load(WPB + c).to(tl.float32)
    wbb = tl.load(WBB + c).to(tl.float32)
    wx = tl.load(WX + c).to(tl.float32)
    wy = tl.load(WY + c).to(tl.float32)
    wz = tl.load(WZ + c).to(tl.float32)
    acc += pb.to(tl.bfloat16).to(tl.float32)[:, None] * wpb[None, :] + bb.to(tl.bfloat16).to(tl.float32)[:, None] * wbb[None, :]
    acc += ux[:, None] * wx[None, :] + uy[:, None] * wy[None, :] + uz[:, None] * wz[None, :]
    # per-token restype terms (W_aa1 restype of token i, W_aa2 restype of token j) and zp[i, j] = linear_z(LN(z))
    ri = tl.load(RI + t * s_r_t + i * s_r_n + c).to(tl.float32)
    rj = tl.load(RJ + t * s_r_t + j[:, None] * s_r_n + c[None, :], mask=jm[:, None], other=0.0).to(tl.float32)
    zp = tl.load(ZP + i * s_zp_i + j[:, None] * s_zp_j + c[None, :], mask=jm[:, None], other=0.0).to(tl.float32)
    acc += ri[None, :] + rj + zp
    tl.store(OUT + t * s_o_t + i * s_o_i + j[:, None] * s_o_j + c[None, :], acc.to(tl.bfloat16), mask=jm[:, None])


@triton.jit
def _tail_kernel(TS, WT, OUT, N, T, s_t, s_i, s_j, s_o_i, s_o_j,
                 C: tl.constexpr, CZ: tl.constexpr, BJ: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)                        # int64 row index: i * s_i never wraps
    jb = tl.program_id(1).to(tl.int64)
    j = jb * BJ + tl.arange(0, BJ).to(tl.int64)              # [BJ] int64 column indices
    jm = j < N
    c = tl.arange(0, C)
    acc = tl.zeros((BJ, C), dtype=tl.float32)
    ptrs = TS + i * s_i + j[:, None] * s_j + c[None, :]
    for tt in range(0, T):                                    # template order, fp32 accumulation (torch.sum over a bf16 dim)
        acc += tl.load(ptrs, mask=jm[:, None], other=0.0).to(tl.float32)
        ptrs += s_t                                           # a 64-bit POINTER advance per template — never the int32 product tt * s_t, which
                                                              # wraps once tt * N^2 * 64 >= 2^31 (T = 4 distinct templates at N >= 2897)
    u = acc.to(tl.bfloat16).to(tl.float32) / T                # stock: the bf16 sum, then bf16 / int (an fp32 op with a bf16 result)
    u = tl.maximum(u.to(tl.bfloat16).to(tl.float32), 0.0)
    cz = tl.arange(0, CZ)
    wt = tl.load(WT + c[:, None] * CZ + cz[None, :])          # [C, CZ] bf16 (W_t^T as autocast rounds it)
    o = tl.dot(u.to(tl.bfloat16), wt)                        # [BJ, CZ] fp32
    tl.store(OUT + i * s_o_i + j[:, None] * s_o_j + cz[None, :], o.to(tl.bfloat16), mask=jm[:, None])


# --------------------------------------------------------------------------------------------------------------------------------- host side
def pack_weights(*, w_dgram: torch.Tensor, w_pb: torch.Tensor, w_x: torch.Tensor, w_y: torch.Tensor, w_z: torch.Tensor, w_bb: torch.Tensor,
                 w_t: torch.Tensor) -> Dict[str, torch.Tensor]:
    """The kernels' weight pack (bf16, the rounding autocast applies to Linear weights): W_dgram [64,39] -> `wd` [KDP=64, 64] (W^T, zero rows past
    39); the four scalar-feature columns [64,1] -> `wpb` / `wbb` / `wx` / `wy` / `wz` [64]; W_t [c_z,64] -> `wt` [64, c_z] (W_t^T).  The restype
    Linears and linear_z stay the caller's statements (their outputs are kernel operands).  Raises Unsupported for shapes outside the domain."""
    if tuple(w_dgram.shape) != (C_T, C_DG):
        raise Unsupported("dims:w_dgram", f"dgram weight {tuple(w_dgram.shape)} is not ({C_T}, {C_DG})")
    for nm, w in (("w_pb", w_pb), ("w_x", w_x), ("w_y", w_y), ("w_z", w_z), ("w_bb", w_bb)):
        if tuple(w.shape) != (C_T, 1):
            raise Unsupported(f"dims:{nm}", f"{nm} {tuple(w.shape)} is not ({C_T}, 1)")
    if w_t.dim() != 2 or int(w_t.shape[1]) != C_T or int(w_t.shape[0]) not in SERVED_CZ:
        raise Unsupported("dims:w_t", f"linear_t weight {tuple(w_t.shape)}: served c_t {C_T}, c_z in {SERVED_CZ}")
    dev, bf = w_dgram.device, torch.bfloat16
    wd = torch.zeros(KDP, C_T, device=dev, dtype=bf)
    wd[:C_DG].copy_(w_dgram.detach().t().to(bf))
    col = lambda w: w.detach()[:, 0].to(bf).contiguous()     # noqa: E731
    return {"wd": wd.contiguous(), "wpb": col(w_pb), "wbb": col(w_bb), "wx": col(w_x), "wy": col(w_y), "wz": col(w_z),
            "wt": w_t.detach().t().to(bf).contiguous()}


def check_embed(dg, uv, pbm, bbm, asym, ri, rj, zp) -> None:
    """Raises Unsupported unless: dg [T,N,N,39] / uv [T,N,N,3] floating CUDA tensors with unit last stride, pbm / bbm [T,N], asym [N],
    ri / rj [T,N,64] bf16, zp [N,N,64] bf16 with unit last stride, all on one device."""
    if not dg.is_cuda:
        raise Unsupported("device:cpu", "the kernels serve CUDA tensors")
    if dg.dim() != 4 or int(dg.shape[-1]) != C_DG or dg.shape[1] != dg.shape[2]:
        raise Unsupported("shape:dgram", f"distogram {tuple(dg.shape)} is not [T, N, N, {C_DG}]")
    T, N = int(dg.shape[0]), int(dg.shape[1])
    if T < 1 or N < 1:
        raise Unsupported("shape:empty", f"T={T} N={N}")
    want = {"uv": (uv, (T, N, N, 3)), "pbm": (pbm, (T, N)), "bbm": (bbm, (T, N)), "asym": (asym, (N,)), "ri": (ri, (T, N, C_T)), "rj": (rj, (T, N, C_T)), "zp": (zp, (N, N, C_T))}
    for nm, (x, shp) in want.items():
        if tuple(x.shape) != shp:
            raise Unsupported(f"shape:{nm}", f"{nm} {tuple(x.shape)} is not {shp}")
        if x.device != dg.device:
            raise Unsupported(f"device:{nm}", f"{nm} on {x.device}, distogram on {dg.device}")
    for nm, x in (("dgram", dg), ("uv", uv), ("ri", ri), ("rj", rj), ("zp", zp)):
        if x.stride(-1) != 1:
            raise Unsupported(f"stride:{nm}", f"{nm} needs unit stride on its last dim (strides {tuple(x.stride())})")
    if not (pbm.is_contiguous() and bbm.is_contiguous() and asym.is_contiguous()):
        raise Unsupported("stride:masks", "pbm / bbm / asym must be contiguous")
    if pbm.stride(0) != bbm.stride(0) or ri.stride() != rj.stride():
        raise Unsupported("stride:pairs", "pbm/bbm and ri/rj must share strides")
    if not (dg.dtype.is_floating_point and uv.dtype.is_floating_point):
        raise Unsupported("dtype:features", f"distogram {dg.dtype} / unit vector {uv.dtype} must be floating")
    if ri.dtype != torch.bfloat16 or rj.dtype != torch.bfloat16 or zp.dtype != torch.bfloat16:
        raise Unsupported("dtype:tables", f"ri {ri.dtype} / rj {rj.dtype} / zp {zp.dtype} must be bf16 (the autocast line's activations)")


def embed(dg: torch.Tensor, uv: torch.Tensor, pbm: torch.Tensor, bbm: torch.Tensor, asym: torch.Tensor, ri: torch.Tensor, rj: torch.Tensor,
          zp: torch.Tensor, P: Dict[str, torch.Tensor], out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """v [T, N, N, 64] bf16 = zp[None] + the fused feature embedding (see the module docstring).  Operands (validated by check_embed):
    dg [T,N,N,39] and uv [T,N,N,3] (fp32 features, any strides with unit last stride), pbm / bbm [T,N] float masks, asym [N] chain ids (any
    dtype comparable for equality), ri / rj [T,N,64] bf16 per-token restype terms, zp [N,N,64] bf16, P = pack_weights(...); `out`, when
    given, is a [T,N,N,64] bf16 tensor with unit last stride to write into."""
    check_embed(dg, uv, pbm, bbm, asym, ri, rj, zp)
    T, N = int(dg.shape[0]), int(dg.shape[1])
    if out is None:
        out = torch.empty((T, N, N, C_T), device=dg.device, dtype=torch.bfloat16)
    elif tuple(out.shape) != (T, N, N, C_T) or out.dtype != torch.bfloat16 or out.stride(-1) != 1:
        raise Unsupported("shape:out", f"out {tuple(out.shape)} {out.dtype} is not a [T,N,N,{C_T}] bf16 tensor with unit last stride")
    grid = (T * N, triton.cdiv(N, BJ_EMBED))
    _embed_kernel[grid](dg, uv, pbm, bbm, asym, ri, rj, zp, out, P["wd"], P["wpb"], P["wx"], P["wy"], P["wz"], P["wbb"],
                        N, dg.stride(0), dg.stride(1), dg.stride(2), uv.stride(0), uv.stride(1), uv.stride(2), pbm.stride(0), ri.stride(0), ri.stride(1),
                        zp.stride(0), zp.stride(1), out.stride(0), out.stride(1), out.stride(2),
                        C=C_T, KD=C_DG, KDPAD=KDP, BJ=BJ_EMBED, num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    return out


def check_tail(s: torch.Tensor, wt: torch.Tensor) -> None:
    if not s.is_cuda:
        raise Unsupported("device:cpu", "the kernels serve CUDA tensors")
    if s.dim() != 4 or int(s.shape[-1]) != C_T or s.shape[1] != s.shape[2] or int(s.shape[0]) < 1 or int(s.shape[1]) < 1:
        raise Unsupported("shape:stack", f"stack output {tuple(s.shape)} is not [T, N, N, {C_T}] with T, N >= 1")
    if s.stride(-1) != 1:
        raise Unsupported("stride:stack", f"stack output needs unit stride on its last dim (strides {tuple(s.stride())})")
    if s.dtype != torch.bfloat16:
        raise Unsupported("dtype:stack", f"stack output {s.dtype} must be bf16 (the autocast line's activations)")
    if wt.dim() != 2 or int(wt.shape[0]) != C_T or int(wt.shape[1]) not in SERVED_CZ or wt.dtype != torch.bfloat16 or not wt.is_contiguous():
        raise Unsupported("dims:w_t", f"wt {tuple(wt.shape)} {wt.dtype}: served [{C_T}, c_z in {SERVED_CZ}] bf16 contiguous (pack_weights)")


def tail(s: torch.Tensor, wt: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """out [N, N, c_z] bf16 = relu( bf16(sum_t s[t]) / T ) @ W_t^T for s [T, N, N, 64] bf16 with ANY template stride (0 = one plane expanded
    over T, read in place) and unit last stride; wt = pack_weights(...)["wt"]."""
    check_tail(s, wt)
    T, N = int(s.shape[0]), int(s.shape[1])
    CZ = int(wt.shape[1])
    if out is None:
        out = torch.empty((N, N, CZ), device=s.device, dtype=torch.bfloat16)
    elif tuple(out.shape) != (N, N, CZ) or out.dtype != torch.bfloat16 or out.stride(-1) != 1:
        raise Unsupported("shape:out", f"out {tuple(out.shape)} {out.dtype} is not a [N,N,{CZ}] bf16 tensor with unit last stride")
    grid = (N, triton.cdiv(N, BJ_TAIL))
    _tail_kernel[grid](s, wt, out, N, T, s.stride(0), s.stride(1), s.stride(2), out.stride(0), out.stride(1),
                       C=C_T, CZ=CZ, BJ=BJ_TAIL, num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    return out


# -------------------------------------------------------------------------------------------------------------------------------- references
def reference_embed(dg, uv, pbm, bbm, asym, restype, zp, *, w_dgram, w_pb, w_aa1, w_aa2, w_x, w_y, w_z, w_bb, dtype=torch.float32):
    """The stock statements materialised at `dtype` (float32: an accurate reference; bfloat16: the autocast chain's roundings — every Linear
    output and every add rounded): a [T,N,N,64] + zp[None]."""
    same = (asym[None, :, None] == asym[None, None, :]).to(dtype)                       # [1, N, N]
    lin = lambda x, w: (x.to(dtype) @ w.to(dtype).t()).to(dtype)                          # noqa: E731
    pbp = (pbm[:, :, None] * pbm[:, None, :]).to(dtype) * same
    bbp = (bbm[:, :, None] * bbm[:, None, :]).to(dtype) * same
    a = lin(dg, w_dgram) + lin(pbp[..., None], w_pb)
    a = a + lin(restype, w_aa1)[:, :, None, :]
    a = a + lin(restype, w_aa2)[:, None, :, :]
    a = a + lin(uv[..., 0:1], w_x) + lin(uv[..., 1:2], w_y) + lin(uv[..., 2:3], w_z)
    a = a + lin(bbp[..., None], w_bb)
    return (zp.to(dtype)[None] + a).to(dtype)


def reference_tail(s, w_t, dtype=torch.float32):
    """The stock statements at `dtype`: relu(sum_t(s) / T) @ W_t^T (bfloat16 = the autocast chain: bf16 sum, bf16 / T, bf16 matmul output)."""
    T = int(s.shape[0])
    u = torch.relu((s.to(dtype).sum(0)).to(dtype) / T).to(dtype)
    return (u @ w_t.to(dtype).t()).to(dtype)
