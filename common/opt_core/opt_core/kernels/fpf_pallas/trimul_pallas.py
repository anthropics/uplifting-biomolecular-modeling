"""trimul_pallas.py - fused Pallas (Triton lowering) GPU kernels for the pairformer TriangleMultiplication (inference only).

Stock module (the model's modules.py::TriangleMultiplication, bf16 activations, C = 128):
    x   = LayerNorm_in(act)                                  (N,N,C)   f32 statistics, output cast to bf16
    prj = GLU(x) = (x @ Wproj) * sigmoid(x @ Wgate)          (N,N,2C)  tokamax pallas kernel
    prj = transpose(prj,(2,0,1)) * mask ; a = prj[0::2], b = prj[1::2]   (C,N,N) each   <- one full read+write pass in XLA
    t   = einsum('cik,cjk->cij', a, b)   (outgoing)  |  einsum('ckj,cki->cij', a, b)  (incoming)     cuBLAS batched GEMM
    y   = LayerNorm_center(t over c) -> transpose (1,2,0)     (N,N,C)
    out = (y @ Wout) * sigmoid(x @ Wgating)                   (N,N,C)
This file replaces it with 3 launches:
    K1  prologue : LN_in + GLU + mask, written DIRECTLY in channel-major planes P,Q (C,N,N)  (for 'incoming' the planes are written transposed,
                   so the contraction is always out[c,i,j] = sum_k P[c,i,k] Q[c,j,k])
    EIN          : the cubic contraction, either XLA/cuBLAS (jnp.einsum 'cik,cjk->cij') or a Pallas batched-matmul kernel (config 'ein')
    K2  epilogue : LN_center + transpose + output projection + gating (LN_in recomputed from act instead of re-read) -> (N,N,C)
Numerics: LN and sigmoid in f32, all matmuls bf16 x bf16 -> f32 accumulate, ONE final rounding to bf16 per kernel (stock rounds after every op).
Not bit-identical to XLA; error vs an fp64 reference is <= stock's (measured). Requires N % block == 0 (the model's token buckets are multiples of 256).
"""
import functools
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

_CP = getattr(plgpu, "CompilerParams", None) or getattr(plgpu, "TritonCompilerParams")
F32, BF16 = jnp.float32, jnp.bfloat16
LN_EPS = 1e-5


def _ln(x, scale, offset):
    """haiku LayerNorm(use_fast_variance=True) over the last axis of a [T, C] f32 tile; scale/offset [C] f32."""
    mean = jnp.mean(x, axis=1, keepdims=True)
    var = jnp.mean(x * x, axis=1, keepdims=True) - mean * mean
    inv = jax.lax.rsqrt(var + LN_EPS) * scale[None, :]
    return (x - mean) * inv + offset[None, :]


def _dot(a, b):
    return jnp.dot(a, b, preferred_element_type=F32)


# ----------------------------------------------------------------------------------------------------------- K1 prologue
def _k1_kernel(x_ref, m_ref, s_ref, o_ref, wpa_ref, wga_ref, wpb_ref, wgb_ref, a_ref, b_ref):
    x = x_ref[...].astype(F32)                                   # [T, C]  one row-segment (outgoing) or column-segment (incoming) of pixels
    xn = _ln(x, s_ref[...], o_ref[...]).astype(BF16)             # LN output is cast back to bf16 in the stock module
    m = m_ref[...].astype(F32)[:, None]                          # [T, 1]
    # plane a then plane b, each stored before the next pair of MMAs: at most two f32 accumulators live at a time
    # (on sm_100 Triton keeps tl.dot accumulators in tensor memory, 512 columns per SM; four live [T,128] accumulators overflow it)
    a = _dot(xn, wpa_ref[...]) * jax.nn.sigmoid(_dot(xn, wga_ref[...])) * m      # [T, C] f32
    a_ref[...] = a.T.astype(a_ref.dtype)                         # [C, T]  channel-major store (T contiguous elements per channel)
    b = _dot(xn, wpb_ref[...]) * jax.nn.sigmoid(_dot(xn, wgb_ref[...])) * m
    b_ref[...] = b.T.astype(b_ref.dtype)


def prologue(act, mask, ln_scale, ln_offset, wpa, wga, wpb, wgb, *, transpose_out, t=64, num_warps=4, num_stages=2):
    """act (N,N,C) bf16, mask (N,N) -> planes A,B (C,N,N) bf16.  transpose_out=False: A[c,i,j] = a(pixel i,j);  True: A[c,j,i] = a(pixel i,j)."""
    N, N2, C = act.shape
    assert N == N2 and N % t == 0, (act.shape, t)
    if not transpose_out:   # program (i, jb): pixels (i, jb*t : jb*t+t)  -> A[:, i, jb*t:...]
        x_spec = pl.BlockSpec((None, t, C), lambda i, jb: (i, jb, 0))
        m_spec = pl.BlockSpec((None, t), lambda i, jb: (i, jb))
        o_spec = pl.BlockSpec((C, None, t), lambda i, jb: (0, i, jb))
    else:                   # program (j, kb): pixels (kb*t : kb*t+t, j) -> A[:, j, kb*t:...]
        x_spec = pl.BlockSpec((t, None, C), lambda j, kb: (kb, j, 0))
        m_spec = pl.BlockSpec((t, None), lambda j, kb: (kb, j))
        o_spec = pl.BlockSpec((C, None, t), lambda j, kb: (0, j, kb))
    vec = pl.BlockSpec((C,), lambda i, jb: (0,))
    mat = pl.BlockSpec((C, C), lambda i, jb: (0, 0))
    return pl.pallas_call(
        _k1_kernel, grid=(N, N // t),
        in_specs=[x_spec, m_spec, vec, vec, mat, mat, mat, mat],
        out_specs=[o_spec, o_spec],
        out_shape=[jax.ShapeDtypeStruct((C, N, N), act.dtype), jax.ShapeDtypeStruct((C, N, N), act.dtype)],
        compiler_params=_CP(num_warps=num_warps, num_stages=num_stages),
        name="trimul_prologue",
    )(act, mask, ln_scale, ln_offset, wpa, wga, wpb, wgb)


# ----------------------------------------------------------------------------------------------------------- EIN cubic contraction
def _ein_kernel(p_ref, q_ref, o_ref, *, tk, nk):
    tm, tn = o_ref.shape

    def body(kb, acc):
        ks = pl.ds(kb * tk, tk)
        p = p_ref[:, ks]                                           # [tm, tk]
        q = q_ref[:, ks]                                           # [tn, tk]
        return acc + _dot(p, q.T)

    acc = jax.lax.fori_loop(0, nk, body, jnp.zeros((tm, tn), F32))
    o_ref[...] = acc.astype(o_ref.dtype)


def einsum_pallas(P, Q, *, tm=128, tn=128, tk=64, num_warps=8, num_stages=3):
    """out[c,i,j] = sum_k P[c,i,k] Q[c,j,k]  (== jnp.einsum('cik,cjk->cij'));  bf16 in, f32 accumulate, bf16 out."""
    C, N, K = P.shape
    assert Q.shape == (C, N, K) and N % tm == 0 and N % tn == 0 and K % tk == 0, (P.shape, tm, tn, tk)
    return pl.pallas_call(
        functools.partial(_ein_kernel, tk=tk, nk=K // tk), grid=(C, N // tm, N // tn),
        in_specs=[pl.BlockSpec((None, tm, K), lambda c, i, j: (c, i, 0)),
                  pl.BlockSpec((None, tn, K), lambda c, i, j: (c, j, 0))],
        out_specs=pl.BlockSpec((None, tm, tn), lambda c, i, j: (c, i, j)),
        out_shape=jax.ShapeDtypeStruct((C, N, N), P.dtype),
        compiler_params=_CP(num_warps=num_warps, num_stages=num_stages),
        name="trimul_einsum",
    )(P, Q)


# ----------------------------------------------------------------------------------------------------------- K2 epilogue
def _k2_kernel(t_ref, x_ref, si_ref, oi_ref, sc_ref, oc_ref, wo_ref, wg_ref, out_ref):
    tt = t_ref[...].astype(F32).T                                # [C, T] -> [T, C]
    y = _ln(tt, sc_ref[...], oc_ref[...]).astype(BF16)           # center_norm (over channels), cast to bf16 as haiku does
    z = _dot(y, wo_ref[...])                                     # output_projection (no bias)
    xn = _ln(x_ref[...].astype(F32), si_ref[...], oi_ref[...]).astype(BF16)   # recompute left_norm_input(act) rather than re-reading it
    g = jax.nn.sigmoid(_dot(xn, wg_ref[...]))                    # gating_linear (no bias)
    out_ref[...] = (z * g).astype(out_ref.dtype)


def epilogue(tri, act, ln_in_scale, ln_in_offset, ln_c_scale, ln_c_offset, w_out, w_gate, *, t=64, num_warps=4, num_stages=2):
    """tri (C,N,N) bf16 contraction result [c,i,j]; act (N,N,C) module input -> module output (N,N,C) bf16."""
    N, _, C = act.shape
    assert tri.shape == (C, N, N) and N % t == 0
    vec = pl.BlockSpec((C,), lambda i, jb: (0,))
    mat = pl.BlockSpec((C, C), lambda i, jb: (0, 0))
    return pl.pallas_call(
        _k2_kernel, grid=(N, N // t),
        in_specs=[pl.BlockSpec((C, None, t), lambda i, jb: (0, i, jb)),
                  pl.BlockSpec((None, t, C), lambda i, jb: (i, jb, 0)),
                  vec, vec, vec, vec, mat, mat],
        out_specs=pl.BlockSpec((None, t, C), lambda i, jb: (i, jb, 0)),
        out_shape=jax.ShapeDtypeStruct((N, N, C), act.dtype),
        compiler_params=_CP(num_warps=num_warps, num_stages=num_stages),
        name="trimul_epilogue",
    )(tri, act, ln_in_scale, ln_in_offset, ln_c_scale, ln_c_offset, w_out, w_gate)


# ----------------------------------------------------------------------------------------------------------- module
def split_glu_weights(w_proj, w_gate):
    """stock: projection (N,N,2C) -> transpose -> reshape (C,2,N,N): plane a = output channels 0::2, plane b = 1::2."""
    return w_proj[:, 0::2], w_gate[:, 0::2], w_proj[:, 1::2], w_gate[:, 1::2]


DEFAULT_CFG = dict(t1=64, w1=4, s1=2, t2=64, w2=4, s2=2, ein="xla", tm=128, tn=128, tk=64, we=8, se=3)


def triangle_multiplication_fused(act, mask, p, *, equation, cfg=None):
    """act (N,N,C) bf16; mask (N,N); p = dict(ln_in_scale, ln_in_offset, w_proj (C,2C), w_gate (C,2C), ln_c_scale, ln_c_offset, w_out (C,C), w_gl (C,C)).
    equation in {'ikc,jkc->ijc' (outgoing), 'kjc,kic->ijc' (incoming)}.  Returns (N,N,C) in act.dtype."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    assert equation in ("ikc,jkc->ijc", "kjc,kic->ijc"), equation
    incoming = equation == "kjc,kic->ijc"
    dt = act.dtype
    wpa, wga, wpb, wgb = [w.astype(dt) for w in split_glu_weights(p["w_proj"], p["w_gate"])]
    A, B = prologue(act, mask.astype(dt), p["ln_in_scale"].astype(F32), p["ln_in_offset"].astype(F32), wpa, wga, wpb, wgb,
                    transpose_out=incoming, t=cfg["t1"], num_warps=cfg["w1"], num_stages=cfg["s1"])
    P, Q = (B, A) if incoming else (A, B)          # incoming: out[c,i,j] = sum_k b(k,i) a(k,j) = sum_k B'[c,i,k] A'[c,j,k]
    if cfg["ein"] == "pallas":
        tri = einsum_pallas(P, Q, tm=cfg["tm"], tn=cfg["tn"], tk=cfg["tk"], num_warps=cfg["we"], num_stages=cfg["se"])
    elif cfg.get("f32_tri"):
        tri = jnp.einsum("cik,cjk->cij", P, Q, preferred_element_type=F32)      # 'hi' precision: contraction result handed to the epilogue in f32
    else:
        tri = jnp.einsum("cik,cjk->cij", P, Q)
    return epilogue(tri, act, p["ln_in_scale"].astype(F32), p["ln_in_offset"].astype(F32), p["ln_c_scale"].astype(F32), p["ln_c_offset"].astype(F32),
                    p["w_out"].astype(dt), p["w_gl"].astype(dt), t=cfg["t2"], num_warps=cfg["w2"], num_stages=cfg["s2"])


# ----------------------------------------------------------------------------------------------------------- references (pure jnp, no kernels)
def triangle_multiplication_reference(act, mask, p, *, equation, compute_dtype=jnp.float32, precision=None):
    """Op-by-op replica of the stock module math. compute_dtype=bf16 mimics the stock rounding points (LN in f32, everything else rounded to bf16);
    compute_dtype=f32/f64 with precision=HIGHEST is the reference. Weights are first rounded to act.dtype (that IS the model), then upcast."""
    dt = act.dtype
    cd = compute_dtype
    hi = dict(precision=precision, preferred_element_type=cd if cd != BF16 else None)
    def ln(x, s, o, axis=-1):
        x32 = x.astype(jnp.promote_types(cd, F32))
        mean = jnp.mean(x32, axis=axis, keepdims=True); var = jnp.mean(jnp.square(x32), axis=axis, keepdims=True) - jnp.square(mean)
        shp = [1] * x.ndim; shp[axis] = x.shape[axis]
        y = (x32 - mean) * jax.lax.rsqrt(var + LN_EPS) * s.astype(x32.dtype).reshape(shp) + o.astype(x32.dtype).reshape(shp)
        return y.astype(cd)
    w = {k: p[k].astype(dt).astype(cd) for k in ("w_proj", "w_gate", "w_out", "w_gl")}
    x = ln(act, p["ln_in_scale"], p["ln_in_offset"])
    proj = jnp.einsum("ijc,cd->ijd", x, w["w_proj"], **hi).astype(cd)
    gate = jnp.einsum("ijc,cd->ijd", x, w["w_gate"], **hi).astype(cd)
    prj = (proj * jax.nn.sigmoid(gate)).astype(cd)
    prj = jnp.transpose(prj, (2, 0, 1)) * mask.astype(cd)[None]
    C = act.shape[-1]
    prj = prj.reshape(C, 2, *prj.shape[1:]); a, b = prj[:, 0], prj[:, 1]
    eq = {"ikc,jkc->ijc": "cik,cjk->cij", "kjc,kic->ijc": "ckj,cki->cij"}[equation]
    tri = jnp.einsum(eq, a, b, **hi).astype(cd)
    y = ln(tri, p["ln_c_scale"], p["ln_c_offset"], axis=0)
    y = jnp.transpose(y, (1, 2, 0))
    z = jnp.einsum("ijc,cd->ijd", y, w["w_out"], **hi).astype(cd)
    g = jax.nn.sigmoid(jnp.einsum("ijc,cd->ijd", x, w["w_gl"], **hi).astype(cd))
    return (z * g).astype(cd)


# =========================================================================================================== v2 kernels
# K1 v2: one GLU plane per launch (2 dots instead of 4 -> half the weight/regs per program), and a 2-D pixel tile (ti x t) per program processed as an
# unrolled loop over ti row-segments so the (C,C) weight tiles are reused ti times per program.
def _k1v2_kernel(x_ref, m_ref, s_ref, o_ref, wp_ref, wg_ref, out_ref, *, ti, transposed):
    s = s_ref[...]; o = o_ref[...]
    for r in range(ti):
        if transposed:      # x block (t, ti, C): pixels (k-segment, j0+r)  -> out block (C, ti, t): [c, j0+r, k-segment]
            x = x_ref[:, r, :].astype(F32); m = m_ref[:, r].astype(F32)[:, None]
        else:               # x block (ti, t, C): pixels (i0+r, j-segment)  -> out block (C, ti, t): [c, i0+r, j-segment]
            x = x_ref[r].astype(F32); m = m_ref[r].astype(F32)[:, None]
        xn = _ln(x, s, o).astype(BF16)
        v = _dot(xn, wp_ref[...]) * jax.nn.sigmoid(_dot(xn, wg_ref[...])) * m          # [t, C]
        out_ref[:, r, :] = v.T.astype(out_ref.dtype)                                     # [C, t]


def prologue_v2(act, mask, ln_scale, ln_offset, wpa, wga, wpb, wgb, *, transpose_out, t=64, ti=4, num_warps=4, num_stages=2):
    N, _, C = act.shape
    assert N % t == 0 and N % ti == 0, (N, t, ti)
    if not transpose_out:   # grid (ib, jb): pixel rows i0..i0+ti, cols j-segment
        x_spec = pl.BlockSpec((ti, t, C), lambda ib, jb: (ib, jb, 0)); m_spec = pl.BlockSpec((ti, t), lambda ib, jb: (ib, jb))
        o_spec = pl.BlockSpec((C, ti, t), lambda ib, jb: (0, ib, jb)); grid = (N // ti, N // t)
    else:                   # grid (jb, kb): pixel rows k-segment, cols j0..j0+ti ; out[c, j, k]
        x_spec = pl.BlockSpec((t, ti, C), lambda jb, kb: (kb, jb, 0)); m_spec = pl.BlockSpec((t, ti), lambda jb, kb: (kb, jb))
        o_spec = pl.BlockSpec((C, ti, t), lambda jb, kb: (0, jb, kb)); grid = (N // ti, N // t)
    vec = pl.BlockSpec((C,), lambda a, b: (0,)); mat = pl.BlockSpec((C, C), lambda a, b: (0, 0))
    outs = []
    for wp, wg, nm in ((wpa, wga, "a"), (wpb, wgb, "b")):
        outs.append(pl.pallas_call(
            functools.partial(_k1v2_kernel, ti=ti, transposed=transpose_out), grid=grid,
            in_specs=[x_spec, m_spec, vec, vec, mat, mat], out_specs=o_spec,
            out_shape=jax.ShapeDtypeStruct((C, N, N), act.dtype),
            compiler_params=_CP(num_warps=num_warps, num_stages=num_stages), name=f"trimul_prologue_v2_{nm}")(act, mask, ln_scale, ln_offset, wp, wg))
    return outs[0], outs[1]


def _k2v2_kernel(t_ref, x_ref, si_ref, oi_ref, sc_ref, oc_ref, wo_ref, wg_ref, out_ref, *, ti):
    si = si_ref[...]; oi = oi_ref[...]; sc = sc_ref[...]; oc = oc_ref[...]
    for r in range(ti):
        tt = t_ref[:, r, :].astype(F32).T                                                # [C, t] -> [t, C]
        y = _ln(tt, sc, oc).astype(BF16)
        z = _dot(y, wo_ref[...])
        xn = _ln(x_ref[r].astype(F32), si, oi).astype(BF16)
        g = jax.nn.sigmoid(_dot(xn, wg_ref[...]))
        out_ref[r] = (z * g).astype(out_ref.dtype)


def epilogue_v2(tri, act, ln_in_scale, ln_in_offset, ln_c_scale, ln_c_offset, w_out, w_gate, *, t=64, ti=4, num_warps=4, num_stages=2):
    N, _, C = act.shape
    assert tri.shape == (C, N, N) and N % t == 0 and N % ti == 0
    vec = pl.BlockSpec((C,), lambda a, b: (0,)); mat = pl.BlockSpec((C, C), lambda a, b: (0, 0))
    return pl.pallas_call(
        functools.partial(_k2v2_kernel, ti=ti), grid=(N // ti, N // t),
        in_specs=[pl.BlockSpec((C, ti, t), lambda ib, jb: (0, ib, jb)), pl.BlockSpec((ti, t, C), lambda ib, jb: (ib, jb, 0)), vec, vec, vec, vec, mat, mat],
        out_specs=pl.BlockSpec((ti, t, C), lambda ib, jb: (ib, jb, 0)),
        out_shape=jax.ShapeDtypeStruct((N, N, C), act.dtype),
        compiler_params=_CP(num_warps=num_warps, num_stages=num_stages), name="trimul_epilogue_v2",
    )(tri, act, ln_in_scale, ln_in_offset, ln_c_scale, ln_c_offset, w_out, w_gate)


def triangle_multiplication_fused_v2(act, mask, p, *, equation, cfg=None):
    """Same contract as triangle_multiplication_fused; cfg keys ti1/ti2 > 0 select the v2 (pixel-tiled) prologue/epilogue, 0 selects v1."""
    cfg = {**DEFAULT_CFG, "ti1": 4, "ti2": 4, **(cfg or {})}
    incoming = equation == "kjc,kic->ijc"; dt = act.dtype
    wpa, wga, wpb, wgb = [w.astype(dt) for w in split_glu_weights(p["w_proj"], p["w_gate"])]
    lni = (p["ln_in_scale"].astype(F32), p["ln_in_offset"].astype(F32)); lnc = (p["ln_c_scale"].astype(F32), p["ln_c_offset"].astype(F32))
    if cfg["ti1"] > 0:
        A, B = prologue_v2(act, mask.astype(dt), *lni, wpa, wga, wpb, wgb, transpose_out=incoming, t=cfg["t1"], ti=cfg["ti1"], num_warps=cfg["w1"], num_stages=cfg["s1"])
    else:
        A, B = prologue(act, mask.astype(dt), *lni, wpa, wga, wpb, wgb, transpose_out=incoming, t=cfg["t1"], num_warps=cfg["w1"], num_stages=cfg["s1"])
    P, Q = (B, A) if incoming else (A, B)
    tri = einsum_pallas(P, Q, tm=cfg["tm"], tn=cfg["tn"], tk=cfg["tk"], num_warps=cfg["we"], num_stages=cfg["se"]) if cfg["ein"] == "pallas" else jnp.einsum("cik,cjk->cij", P, Q)
    if cfg["ti2"] > 0:
        return epilogue_v2(tri, act, *lni, *lnc, p["w_out"].astype(dt), p["w_gl"].astype(dt), t=cfg["t2"], ti=cfg["ti2"], num_warps=cfg["w2"], num_stages=cfg["s2"])
    return epilogue(tri, act, *lni, *lnc, p["w_out"].astype(dt), p["w_gl"].astype(dt), t=cfg["t2"], num_warps=cfg["w2"], num_stages=cfg["s2"])
