"""protenix_fpf_atom_attn_exact.kernel — lever `atom_attn_exact`: EXACT-BITWISE fused local attention for the Protenix-v2 atom transformer.

Replaces protenix.model.modules.primitives._local_attention for fp32 q/k/v [.., H, N, 32] with n_queries=32, n_keys=128, attn_bias=None,
trunked_attn_bias [.., H, nb, 32, 128], use_efficient_implementation=True (stock inference).  ONE Triton launch per call reproduces, bit for bit,
what stock computes through rearrange_to_dense_trunk (zero padding, 128-key windows at stride 32, pad mask 0 / -inf(=1e10)) + chunk_layer(256) +
F.scaled_dot_product_attention MATH path (scale=1.0): S = q k^T by cuBLAS (cutlass s1688 TF32: operands rounded to tf32 round-to-nearest-even,
sequential-K fp32 accumulation from +0 — or, when cuBLAS's heuristic picks it for the remainder chunk, the FFMA fp32 kernel: acc = fma(q_d, k_d, acc)
d ascending), S += (mask + bias), P = softmax_warp_forward(S) (row max; libdevice expf(x - max); per-lane sums of the 4 lane-strided elements from 0;
xor-butterfly 16,8,4,2,1; IEEE division), O = P v (TF32 as above, K=128 sequential).  Which GEMM numerics cuBLAS uses for a given chunk batch count is
determined per process at install (torch.matmul on the exact shapes/layouts compared with both variants); a batch count matching neither refuses by name.
Output is written directly in the [.., N, H, 32] layout the caller transposes to (pure layout; no arithmetic).
"""
import math, os, sys
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

NQ, NK, D = 32, 128, 32
HEADS = 4                       # atom transformer heads (pinned topology); route table covers multiples of HEADS
CHUNK = 256                     # stock chunk_layer chunk size at this site (infer_setting chunk_size); read from the call's chunk_size argument
STATE = {"routes_qk": {}, "routes_pv": {}, "routes_done": False, "calls": {"kernel": 0, "original": 0}, "why_original": {}}


def _rne_tf32(x):
    b = x.view(torch.int32) if x.is_contiguous() else x.contiguous().view(torch.int32)
    exp_all1 = (b & 0x7f800000) == 0x7f800000
    inc = ((b & (1 << 12)) != 0) & (((b & 0xFFF) != 0) | ((b & (1 << 13)) != 0))
    return (torch.where(inc & ~exp_all1, b + (1 << 13), b) & -8192).view(torch.float32)


@triton.jit
def _rne(x):
    """fp32 -> tf32-representable fp32, round to nearest even (CUTLASS NumericConverter<tfloat32_t, float>)."""
    b = x.to(tl.int32, bitcast=True)
    not_special = (b & 0x7f800000) != 0x7f800000
    rnd = (b & 4096) != 0
    sticky_or_odd = ((b & 4095) != 0) | ((b & 8192) != 0)
    inc = rnd & sticky_or_odd & not_special
    b = tl.where(inc, b + 8192, b) & -8192
    return b.to(tl.float32, bitcast=True)


@triton.jit
def _fadd(a, b):
    # a + b as ONE IEEE fp32 add the compiler cannot fold into the preceding dot accumulator (Triton rewrites dot(a,b)+c into dot(a,b,acc=c),
    # which starts the K accumulation from c instead of adding c after it: different rounding than stock bmm-then-add).
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _qk_block(Qp, Kp, sq_n, sk_n, q_row0, k_row0, NA, FFMA: tl.constexpr):
    """S[32, 32] = q[32 rows, 32 d] @ k[32 keys, 32 d]^T for one 32-key block; rows/keys outside [0, NA) are zero (stock zero padding)."""
    i = tl.arange(0, 32)
    qr = q_row0 + i; kr = k_row0 + i
    qm = (qr >= 0) & (qr < NA); km = (kr >= 0) & (kr < NA)
    if FFMA:
        acc = tl.zeros((32, 32), dtype=tl.float32)
        for d in tl.static_range(0, 32):
            qa = tl.load(Qp + qr * sq_n + d, mask=qm, other=0.0)
            kb = tl.load(Kp + kr * sk_n + d, mask=km, other=0.0)
            acc = tl.math.fma(qa[:, None], kb[None, :], acc)
        return acc
    else:
        dd = tl.arange(0, 32)
        qt = tl.load(Qp + qr[:, None] * sq_n + dd[None, :], mask=qm[:, None], other=0.0)          # [32 q, 32 d]
        kt = tl.load(Kp + kr[None, :] * sk_n + dd[:, None], mask=km[None, :], other=0.0)          # [32 d, 32 keys]
        return tl.dot(_rne(qt), _rne(kt), input_precision="tf32")


@triton.jit
def _atom_attn_exact_kernel(Q, K, V, B, O, NA, NB, H, F_FULL,
                            sq_s, sq_h, sq_n, sk_s, sk_h, sk_n, sv_s, sv_h, sv_n,
                            sb_s, sb_h, sb_j, sb_q, so_s, so_n, so_h,
                            REM_QK_FFMA: tl.constexpr):
    f = tl.program_id(0)                      # flattened (sample, head, trunk) index in stock's chunk order
    j = f % NB; sh = f // NB; h = sh % H; s = sh // H
    Qp = Q + s * sq_s + h * sq_h; Kp = K + s * sk_s + h * sk_h; Vp = V + s * sv_s + h * sv_h
    Bp = B + s * sb_s + h * sb_h + j * sb_j
    q_row0 = j * 32; k_base = j * 32 - 48
    i = tl.arange(0, 32); c = tl.arange(0, 32)
    if REM_QK_FFMA:
        use_ffma = f >= F_FULL
    else:
        use_ffma = False
    if use_ffma:
        s0 = _qk_block(Qp, Kp, sq_n, sk_n, q_row0, k_base + 0, NA, True)
        s1 = _qk_block(Qp, Kp, sq_n, sk_n, q_row0, k_base + 32, NA, True)
        s2 = _qk_block(Qp, Kp, sq_n, sk_n, q_row0, k_base + 64, NA, True)
        s3 = _qk_block(Qp, Kp, sq_n, sk_n, q_row0, k_base + 96, NA, True)
    else:
        s0 = _qk_block(Qp, Kp, sq_n, sk_n, q_row0, k_base + 0, NA, False)
        s1 = _qk_block(Qp, Kp, sq_n, sk_n, q_row0, k_base + 32, NA, False)
        s2 = _qk_block(Qp, Kp, sq_n, sk_n, q_row0, k_base + 64, NA, False)
        s3 = _qk_block(Qp, Kp, sq_n, sk_n, q_row0, k_base + 96, NA, False)
    # + (pad mask + pair bias): the combined additive tensor, [.., nb, 32, 128]
    bp = Bp + i[:, None] * sb_q + c[None, :]
    s0 = _fadd(s0, tl.load(bp)); s1 = _fadd(s1, tl.load(bp + 32)); s2 = _fadd(s2, tl.load(bp + 64)); s3 = _fadd(s3, tl.load(bp + 96))
    # softmax_warp_forward statement (lane l holds columns l, l+32, l+64, l+96)
    m = tl.max(tl.maximum(tl.maximum(s0, s1), tl.maximum(s2, s3)), axis=1)
    p0 = libdevice.exp(s0 - m[:, None]); p1 = libdevice.exp(s1 - m[:, None]); p2 = libdevice.exp(s2 - m[:, None]); p3 = libdevice.exp(s3 - m[:, None])
    ls = ((p0 + p1) + p2) + p3                                            # [32 rows, 32 lanes]
    t = tl.sum(tl.reshape(ls, (32, 2, 16)), axis=1)                       # pairs (l, l^16)
    t = tl.sum(tl.reshape(t, (32, 2, 8)), axis=1)                         # (.., ^8)
    t = tl.sum(tl.reshape(t, (32, 2, 4)), axis=1)
    t = tl.sum(tl.reshape(t, (32, 2, 2)), axis=1)
    tot = tl.sum(t, axis=1)                                               # [32]
    p0 = tl.div_rn(p0, tot[:, None]); p1 = tl.div_rn(p1, tot[:, None]); p2 = tl.div_rn(p2, tot[:, None]); p3 = tl.div_rn(p3, tot[:, None])
    # O = P @ V, K = 128 sequential over the four 32-key blocks (TF32 route)
    dd = tl.arange(0, 32)
    kr = k_base + i
    v0 = tl.load(Vp + (kr + 0)[:, None] * sv_n + dd[None, :], mask=((kr + 0) >= 0)[:, None] & ((kr + 0) < NA)[:, None], other=0.0)
    acc = tl.dot(_rne(p0), _rne(v0), input_precision="tf32")
    v1 = tl.load(Vp + (kr + 32)[:, None] * sv_n + dd[None, :], mask=((kr + 32) >= 0)[:, None] & ((kr + 32) < NA)[:, None], other=0.0)
    acc = tl.dot(_rne(p1), _rne(v1), acc, input_precision="tf32")
    v2 = tl.load(Vp + (kr + 64)[:, None] * sv_n + dd[None, :], mask=((kr + 64) >= 0)[:, None] & ((kr + 64) < NA)[:, None], other=0.0)
    acc = tl.dot(_rne(p2), _rne(v2), acc, input_precision="tf32")
    v3 = tl.load(Vp + (kr + 96)[:, None] * sv_n + dd[None, :], mask=((kr + 96) >= 0)[:, None] & ((kr + 96) < NA)[:, None], other=0.0)
    acc = tl.dot(_rne(p3), _rne(v3), acc, input_precision="tf32")
    orow = q_row0 + i
    tl.store(O + s * so_s + orow[:, None] * so_n + h * so_h + dd[None, :], acc, mask=(orow < NA)[:, None])


# ---------------------------------------------------------------- host side
_MASK_CACHE = {}


def pad_mask(n, device, inf):
    """[nb, 32, 128] fp32: 0 where stock's rearrange_to_dense_trunk mask is 0, -inf(=-1e10) where it pads (rows >= n, keys outside [0, n))."""
    key = (n, str(device), inf)
    m = _MASK_CACHE.get(key)
    if m is None:
        nb = (n + NQ - 1) // NQ
        r = torch.arange(nb * NQ, device=device).view(nb, NQ, 1)
        cidx = (torch.arange(nb, device=device) * NQ - (NK - NQ) // 2).view(nb, 1, 1) + torch.arange(NK, device=device).view(1, 1, NK)
        bad = (r >= n) | (cidx < 0) | (cidx >= n)
        m = torch.zeros(nb, NQ, NK, dtype=torch.float32, device=device).masked_fill_(bad, -inf)
        if len(_MASK_CACHE) > 16:
            _MASK_CACHE.pop(next(iter(_MASK_CACHE)))
        _MASK_CACHE[key] = m
    return m


def _variant_qk(Q, KT, ffma):
    """torch-side variants used only by the route determination: Q [b,32,32], KT [b,32,128] logical."""
    if ffma:
        acc = torch.zeros(Q.shape[0], 32, 128, device=Q.device)
        for d in range(32):
            acc = torch.addcmul(acc, Q[:, :, d:d + 1], KT[:, d:d + 1, :])      # fused multiply-add per element (equals the fma chain)
        return acc
    out = torch.empty(Q.shape[0], 32, 128, device=Q.device)
    _calib_dot[(Q.shape[0],)](_rne_tf32(Q.contiguous()), _rne_tf32(KT.contiguous()), out, KDIM=32, N=128, num_warps=4)
    return out


@triton.jit
def _calib_dot(A, Bm, C, KDIM: tl.constexpr, N: tl.constexpr):
    b = tl.program_id(0)
    i = tl.arange(0, 32); kk = tl.arange(0, KDIM); n = tl.arange(0, N)
    a = tl.load(A + b * 32 * KDIM + i[:, None] * KDIM + kk[None, :])
    bt = tl.load(Bm + b * KDIM * N + kk[:, None] * N + n[None, :])
    tl.store(C + b * 32 * N + i[:, None] * N + n[None, :], tl.dot(a, bt, input_precision="tf32"))


def _variant_pv(P, V):
    out = torch.empty(P.shape[0], 32, 32, device=P.device)
    _calib_dot[(P.shape[0],)](_rne_tf32(P.contiguous()), _rne_tf32(V.contiguous()), out, KDIM=128, N=32, num_warps=4)
    return out


def determine_routes(device, batches=None, seed=1234):
    """Per process, never cached to disk: which numerics cuBLAS uses for each chunk batch count at the two GEMM call shapes/layouts of this site."""
    if batches is None:
        batches = range(HEADS, CHUNK + 1, HEADS)          # reachable remainder batch counts: S * HEADS * n_trunks mod CHUNK is a multiple of HEADS
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(CHUNK, NQ, D, generator=g).to(device); k = torch.randn(CHUNK, NK, D, generator=g).to(device); v = torch.randn(CHUNK, NK, D, generator=g).to(device)
    logits = torch.randn(CHUNK, NQ, NK, generator=g).to(device) * 3
    bad = []
    for b in batches:
        Q, Kb, Vb = q[:b].contiguous(), k[:b].contiguous(), v[:b].contiguous()
        ref = torch.matmul(Q, Kb.transpose(-2, -1))                          # the math-SDPA call's layout: k contiguous [b,128,32], transposed view
        KT = Kb.transpose(-2, -1)
        if torch.equal(_variant_qk(Q, KT, False), ref):
            STATE["routes_qk"][b] = "tf32"
        elif torch.equal(_variant_qk(Q, KT, True), ref):
            STATE["routes_qk"][b] = "ffma"
        else:
            STATE["routes_qk"][b] = None; bad.append(("qk", b))
        P = torch.softmax(logits[:b], -1).contiguous()
        refpv = torch.matmul(P, Vb)
        STATE["routes_pv"][b] = "tf32" if torch.equal(_variant_pv(P, Vb), refpv) else None
        if STATE["routes_pv"][b] is None:
            bad.append(("pv", b))
    STATE["routes_done"] = True
    return bad


def route_summary():
    def spans(d):
        out, cur = [], None
        for b in sorted(d):
            if cur and cur[0] == d[b] and cur[2] == b - HEADS:
                cur[2] = b
            else:
                cur = [d[b], b, b]; out.append(cur)
        return ",".join(f"{r}[{a}..{z}]" if a != z else f"{r}[{a}]" for r, a, z in out)
    return f"qk:{spans(STATE['routes_qk'])} pv:{spans(STATE['routes_pv'])}"


class Refused(RuntimeError):
    pass


def fused_local_attention(q, k, v, bias_comb, chunk):
    """q,k,v [S.., H, NA, 32] fp32 (last stride 1); bias_comb [Sb, H, nb, 32, 128] fp32 contiguous (Sb in {1, S}); returns [.., H, NA, 32] as a
    transpose view of a contiguous [.., NA, H, 32] buffer."""
    lead = q.shape[:-3]; H, NA = q.shape[-3], q.shape[-2]
    S = 1
    for x in lead: S *= int(x)
    q3, k3, v3 = q.reshape(S, H, NA, D), k.reshape(S, H, NA, D), v.reshape(S, H, NA, D)     # views for the standard [S,NA,H*D]->transpose layout
    for t in (q3, k3, v3):
        assert t.stride(-1) == 1
    nb = (NA + NQ - 1) // NQ
    b5 = bias_comb.reshape(-1, H, nb, NQ, NK)
    assert b5.is_contiguous() and b5.shape[0] in (1, S)
    F_total = S * H * nb
    n_full = (F_total // chunk) * chunk; rem = F_total - n_full
    rq = "tf32"; 
    if rem:
        rq = STATE["routes_qk"].get(rem); rp = STATE["routes_pv"].get(rem)
        if rq is None or rp != "tf32":
            raise Refused(f"atom_attn_exact: cuBLAS numerics for remainder chunk batch {rem} not reproducible (qk={rq}, pv={rp})")
    if n_full and (STATE["routes_qk"].get(chunk) != "tf32" or STATE["routes_pv"].get(chunk) != "tf32"):
        raise Refused(f"atom_attn_exact: cuBLAS numerics for full chunks ({chunk}) not tf32-sequential (qk={STATE['routes_qk'].get(chunk)}, pv={STATE['routes_pv'].get(chunk)})")
    out = torch.empty(*lead, NA, H, D, dtype=torch.float32, device=q.device) if lead else torch.empty(NA, H, D, dtype=torch.float32, device=q.device)
    o3 = out.reshape(S, NA, H, D)
    _atom_attn_exact_kernel[(F_total,)](
        q3, k3, v3, b5, o3, NA, nb, H, n_full,
        q3.stride(0), q3.stride(1), q3.stride(2), k3.stride(0), k3.stride(1), k3.stride(2), v3.stride(0), v3.stride(1), v3.stride(2),
        (b5.stride(0) if b5.shape[0] > 1 else 0), b5.stride(1), b5.stride(2), b5.stride(3), o3.stride(0), o3.stride(1), o3.stride(2),
        REM_QK_FFMA=(rq == "ffma"), num_warps=4)
    return out.transpose(-2, -3)


def make_local_attention(original):
    """Drop-in for primitives._local_attention (same signature); envelope misses take the original statement (counted)."""
    def _local_attention(q, k, v, n_queries, n_keys, attn_bias=None, trunked_attn_bias=None, inf=1e10, use_efficient_implementation=True, inplace_safe=False, chunk_size=None):
        why = None
        if not (q.dtype == k.dtype == v.dtype == torch.float32): why = "dtype"
        elif not (n_queries == NQ and n_keys == NK and q.shape[-1] == D and q.shape == k.shape == v.shape and q.dim() >= 3 and q.shape[-3] == HEADS): why = "shape"
        elif attn_bias is not None or trunked_attn_bias is None or trunked_attn_bias.dtype != torch.float32: why = "bias form"
        elif not use_efficient_implementation: why = "inefficient path requested"
        elif not q.is_cuda: why = "device"
        elif min(q.stride(-1), k.stride(-1), v.stride(-1)) != 1: why = "strides"
        if why is not None:
            STATE["calls"]["original"] += 1; STATE["why_original"][why] = STATE["why_original"].get(why, 0) + 1
            return original(q, k, v, n_queries, n_keys, attn_bias=attn_bias, trunked_attn_bias=trunked_attn_bias, inf=inf,
                            use_efficient_implementation=use_efficient_implementation, inplace_safe=inplace_safe, chunk_size=chunk_size)
        NA = q.shape[-2]; H = q.shape[-3]
        mask = pad_mask(NA, q.device, inf)                                      # [nb,32,128], values {0, -inf}
        comb = mask + trunked_attn_bias                                        # stock: attn_bias_trunked + trunked_attn_bias (fp32 add; same op)
        lead = q.shape[:-3]
        chunk = chunk_size if chunk_size is not None else (int(torch.tensor(lead).prod()) if len(lead) else 1) * H * mask.shape[0]   # no chunking == one chunk of everything
        STATE["calls"]["kernel"] += 1
        return fused_local_attention(q, k, v, comb.contiguous(), chunk)
    _local_attention._atom_attn_exact = True
    return _local_attention
