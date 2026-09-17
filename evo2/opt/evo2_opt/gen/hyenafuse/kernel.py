"""One Triton kernel = one hyena block's whole decode-step filter path (vtx 1.1.0 `HyenaCascade.sequential_forward`):
outer short FIR (3 taps, state of 2) on the three projection rows feeding each channel, then the inner FIR (hcs 7 taps / hcm 128 taps,
state of N-1) or the modal IIR (hcl, 16 poles), the gating products, the bf16 casts, and the state updates. States are read from one
buffer and written, physically rolled exactly as `engine.step_fir` leaves them, to a second buffer (ping-pong): a Triton program is a
tile program executed by many threads, so an in-place roll (thread A stores slot k while thread B still has to load slot k+1) and the
replicated 1-D layouts of the outer rows would race; separate in/out buffers make every load independent of every store.

Bitwise contract (why the arithmetic is spelled out instead of using tl.sum / FMAs):
  * every product and every sum is a separately rounded fp32 op (launch with enable_fp_fusion=False); bf16 products (x1*v, D*x1v, y*x2)
    are exact fp32 products of bf16 values rounded once to bf16 (== torch's bf16 mul);
  * torch.sum over the tap dimension (N contiguous fp32 values, N <= 128) is reproduced in ATen Reduce.cuh order: L = min(lastpow2(N), 32)
    threads, thread t owns elements t, t+L, t+2L, t+3L in four accumulators that start at +0.0f (hence the `+ 0.0` on every product: it is the
    `0.0f + x` of the first accumulate, which canonicalises -0.0), folded ((acc0+acc1)+acc2)+acc3, then the shfl-down pairwise tree over lanes
    (adjacent pairs first). `_lane_tree` is that tree.
`__init__` builds the constants (projection-row map, fp32 taps, poles) from the live parameters with vortex's own functions and launches this kernel with enable_fp_fusion=False; the in/out state buffers it passes never alias."""
import triton
import triton.language as tl


@triton.jit
def _bf16r(x):
    """fp32 -> bf16 (RNE) -> fp32: the value torch's `.to(torch.bfloat16)` produces, held in fp32."""
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _lane_tree(v, BLOCK_C: tl.constexpr, L: tl.constexpr):
    """[BLOCK_C, L] -> [BLOCK_C]: pairwise tree over adjacent lanes ((v0+v1)+(v2+v3))+..., L a power of two <= 32."""
    if L >= 2:
        a, b = tl.split(tl.reshape(v, [BLOCK_C, L // 2, 2]))
        v = a + b
    if L >= 4:
        a, b = tl.split(tl.reshape(v, [BLOCK_C, L // 4, 2]))
        v = a + b
    if L >= 8:
        a, b = tl.split(tl.reshape(v, [BLOCK_C, L // 8, 2]))
        v = a + b
    if L >= 16:
        a, b = tl.split(tl.reshape(v, [BLOCK_C, L // 16, 2]))
        v = a + b
    if L >= 32:
        a, b = tl.split(tl.reshape(v, [BLOCK_C, L // 32, 2]))
        v = a + b
    return tl.reshape(v, [BLOCK_C])


@triton.jit
def _outer_fir(Z, z_sb, z_sc, WS, SS, SSO, ss_sb, ss_sc, ss_sk, b, j, cm):
    """engine.step_fir for filter length 3 / state 2 on rows j: y = h0*u + (s0*w0 + s1*w1) in fp32; new state [s1, u] written to SSO.
    Returns the bf16-rounded output held in fp32."""
    u = tl.load(Z + b * z_sb + j * z_sc, mask=cm, other=0.0).to(tl.float32)
    w0 = tl.load(WS + j * 3 + 0, mask=cm, other=0.0)
    w1 = tl.load(WS + j * 3 + 1, mask=cm, other=0.0)
    h0 = tl.load(WS + j * 3 + 2, mask=cm, other=0.0)
    off = b * ss_sb + j * ss_sc
    s0 = tl.load(SS + off, mask=cm, other=0.0)
    s1 = tl.load(SS + off + ss_sk, mask=cm, other=0.0)
    a0 = (s0 * w0) + 0.0
    a1 = (s1 * w1) + 0.0
    y = (h0 * u) + (a0 + a1)
    tl.store(SSO + off, s1, mask=cm)
    tl.store(SSO + off + ss_sk, u, mask=cm)
    return _bf16r(y)


@triton.jit
def _hyena_decode_step(
    Z, z_sb, z_sc,                      # projections output u[:, -1]: [B, 3H] bf16
    MAP,                                # int32 [3, H]: row of Z feeding (x2, x1, v) of channel c
    WS,                                 # fp32 [3H, 3]: outer taps (w0, w1 on state[0], state[1]; h0 on u)
    SS, SSO, ss_sb, ss_sc, ss_sk,       # fp32 [B, 3H, 2] outer state: read SS, write SSO (same strides)
    WI, BI,                             # fp32 [G, NI+1] inner taps (tap k on state[k], index NI = h0); fp32 [H] gated bias (hcm)
    SI, SIO, si_sb, si_sc, si_sk,       # fp32 [B, H, NI] inner FIR state: read SI, write SIO
    PO, RE, DD,                         # fp32 [H, NS] poles (exp(log_poles) precomputed), fp32 [H, NS] residues, bf16 [H] D
    SR, SRO, sr_sb, sr_sc, sr_sk,       # fp32 [B, H, NS] IIR state: read SR, write SRO
    Y, y_sb, y_sc,                      # bf16 [B, H] output
    H, CPG,                             # channels, channels per filter group
    MODE: tl.constexpr,                 # 0 = inner FIR (hcs / hcm), 1 = modal IIR (hcl)
    NI: tl.constexpr,                   # inner state length (FIR: filter_len - 1; IIR: state size)
    L: tl.constexpr,                    # reduction lanes = min(lastpow2(NI), 32)
    GATED_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    b = tl.program_id(1)
    c = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = c < H

    j2 = tl.load(MAP + c, mask=cm, other=0)
    j1 = tl.load(MAP + H + c, mask=cm, other=0)
    jv = tl.load(MAP + 2 * H + c, mask=cm, other=0)
    x2 = _outer_fir(Z, z_sb, z_sc, WS, SS, SSO, ss_sb, ss_sc, ss_sk, b, j2, cm)
    x1 = _outer_fir(Z, z_sb, z_sc, WS, SS, SSO, ss_sb, ss_sc, ss_sk, b, j1, cm)
    v = _outer_fir(Z, z_sb, z_sc, WS, SS, SSO, ss_sb, ss_sc, ss_sk, b, jv, cm)

    x1v = _bf16r(x1 * v)                # bf16 product

    if MODE == 0:
        g = c // CPG
        t = tl.arange(0, L)
        row = SI + b * si_sb + c[:, None] * si_sc
        wrow = WI + g[:, None] * (NI + 1)
        # four accumulators per thread, element n = i*L + t (NI <= 4L); masked slots are untouched +0.0 accumulators
        n0 = t[None, :]
        m0 = cm[:, None] & (n0 < NI)
        vs = (tl.load(row + n0 * si_sk, mask=m0, other=0.0) * tl.load(wrow + n0, mask=m0, other=0.0)) + 0.0
        n1 = L + t[None, :]
        m1 = cm[:, None] & (n1 < NI)
        vs = vs + ((tl.load(row + n1 * si_sk, mask=m1, other=0.0) * tl.load(wrow + n1, mask=m1, other=0.0)) + 0.0)
        n2 = 2 * L + t[None, :]
        m2 = cm[:, None] & (n2 < NI)
        vs = vs + ((tl.load(row + n2 * si_sk, mask=m2, other=0.0) * tl.load(wrow + n2, mask=m2, other=0.0)) + 0.0)
        n3 = 3 * L + t[None, :]
        m3 = cm[:, None] & (n3 < NI)
        vs = vs + ((tl.load(row + n3 * si_sk, mask=m3, other=0.0) * tl.load(wrow + n3, mask=m3, other=0.0)) + 0.0)
        s = _lane_tree(vs, BLOCK_C, L)
        h0 = tl.load(WI + g * (NI + 1) + NI, mask=cm, other=0.0)
        y = (h0 * x1v) + s
        if GATED_BIAS:
            bi = tl.load(BI + c, mask=cm, other=0.0)
            y = y + (bi * x1v)
        out = _bf16r(_bf16r(y) * x2)     # step_fir output cast to bf16, then the bf16 gate y * x2
        # new state = old rolled left by one with u (= x1v) appended, written to the other buffer
        k = tl.arange(0, 4 * L)[None, :]
        nxt = tl.load(row + (k + 1) * si_sk, mask=cm[:, None] & ((k + 1) < NI), other=0.0)
        newst = tl.where(k == (NI - 1), x1v[:, None], nxt)
        tl.store(SIO + b * si_sb + c[:, None] * si_sc + k * si_sk, newst, mask=cm[:, None] & (k < NI))
    else:
        t = tl.arange(0, L)[None, :]    # L == NI == state size
        m = cm[:, None] & (t < NI)
        soff = b * sr_sb + c[:, None] * sr_sc + t * sr_sk
        st = tl.load(SR + soff, mask=m, other=0.0)
        po = tl.load(PO + c[:, None] * NI + t, mask=m, other=0.0)
        re = tl.load(RE + c[:, None] * NI + t, mask=m, other=0.0)
        st = (po * st) + x1v[:, None]
        tl.store(SRO + soff, st, mask=m)
        res = _lane_tree((re * st) + 0.0, BLOCK_C, L)
        d = tl.load(DD + c, mask=cm, other=0.0).to(tl.float32)
        out = x2 * (res + _bf16r(d * x1v))
    tl.store(Y + b * y_sb + c * y_sc, out.to(tl.bfloat16), mask=cm)
