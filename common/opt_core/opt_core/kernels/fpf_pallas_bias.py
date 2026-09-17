"""``+bias`` variants of the carried ``fpf_pallas`` prologue / epilogue kernels: the same fused pair-stack blocks for a module family whose
triangle-block Linears carry biases (left/right projection + gate biases and output-projection + gating biases on the triangle
multiplication; gating + output biases on the triangle attention — the biased parameterisation). One FMA per bias inside the same
kernels; every other statement (LayerNorm, bf16 rounding points, MMA dtypes, channel-major plane stores, transposed reads for the
incoming equation / the ending node, tile grid) is the carried kernel's, whose math helpers (``_ln``, ``_dot``) and unchanged stages
(``split_glu_weights``, the contraction, ``attn_prologue``, ``flash_attention_bshd``) are imported, not re-implemented. The bias-free
blocks stay the carried bytes: ``fpf_pallas_serve`` routes here only when the parameter dict carries the bias arrays.

    triangle_multiplication_fused_bias(act, mask, p, *, equation, cfg=None)
        ``p`` = the carried block's dict + ``b_proj[2C]``, ``b_gate[2C]`` (interleaved like ``w_proj`` / ``w_gate``: entry ``2c`` = left channel
        ``c``, ``2c+1`` = right channel ``c``), ``b_out[C]``, ``b_gl[C]``:
        left  = mask * (LN(act) @ Wl + bl) * sigmoid(LN(act) @ Wlg + blg)      (right likewise)
        out   = (LN_c(tri) @ Wout + b_out) * sigmoid(LN(act) @ Wgl + b_gl)
    grid_self_attention_fused_bias(act, pair_mask, kp, *, transpose, ending_bias_transposed=True, cfg=None)
        ``kp`` = the carried block's dict + ``bg[HD]`` (gating bias), ``bo[C]`` (output bias), f32:
        out = ((o * sigmoid(LN(act) @ Wg + bg)) @ Wo) + bo       (q / k / v / pair-bias projections carry no bias, as in both families)

jax and the carried kernels are imported on first use (``_ns()``), never at import.
"""
from types import SimpleNamespace
from typing import Any, Dict, Optional

TRIMUL_BIAS_KEYS = ("b_proj", "b_gate", "b_out", "b_gl")
ATTN_BIAS_KEYS = ("bg", "bo")
_NS: Dict[str, Any] = {}


def _carried():
    from . import fpf_pallas_serve as S  # noqa: WPS433
    return S.kernels()


def _ns() -> SimpleNamespace:
    """Build (once) the +bias kernels over the carried modules' helpers."""
    if "ns" in _NS:
        return _NS["ns"]
    import jax  # noqa: WPS433
    import jax.numpy as jnp  # noqa: WPS433
    from jax.experimental import pallas as pl  # noqa: WPS433
    K, A = _carried()
    F32, BF16 = jnp.float32, jnp.bfloat16
    _CP = K._CP

    # ------------------------------------------------------------------------------------------------ triangle multiplication
    def _k1b_kernel(x_ref, m_ref, s_ref, o_ref, wpa_ref, wga_ref, wpb_ref, wgb_ref, bpa_ref, bga_ref, bpb_ref, bgb_ref, a_ref, b_ref):
        x = x_ref[...].astype(F32)                                   # [T, C]
        xn = K._ln(x, s_ref[...], o_ref[...]).astype(BF16)
        m = m_ref[...].astype(F32)[:, None]                          # [T, 1]
        a = (K._dot(xn, wpa_ref[...]) + bpa_ref[...][None, :]) * jax.nn.sigmoid(K._dot(xn, wga_ref[...]) + bga_ref[...][None, :]) * m
        a_ref[...] = a.T.astype(a_ref.dtype)                         # [C, T] channel-major store
        b = (K._dot(xn, wpb_ref[...]) + bpb_ref[...][None, :]) * jax.nn.sigmoid(K._dot(xn, wgb_ref[...]) + bgb_ref[...][None, :]) * m
        b_ref[...] = b.T.astype(b_ref.dtype)

    def prologue_bias(act, mask, ln_scale, ln_offset, wpa, wga, wpb, wgb, bpa, bga, bpb, bgb, *, transpose_out, t=64, num_warps=4, num_stages=2):
        """As the carried ``prologue`` (planes A, B (C,N,N); transposed stores for the incoming equation) with the four bias vectors [C] f32."""
        N, N2, C = act.shape
        assert N == N2 and N % t == 0, (act.shape, t)
        if not transpose_out:
            x_spec = pl.BlockSpec((None, t, C), lambda i, jb: (i, jb, 0))
            m_spec = pl.BlockSpec((None, t), lambda i, jb: (i, jb))
            o_spec = pl.BlockSpec((C, None, t), lambda i, jb: (0, i, jb))
        else:
            x_spec = pl.BlockSpec((t, None, C), lambda j, kb: (kb, j, 0))
            m_spec = pl.BlockSpec((t, None), lambda j, kb: (kb, j))
            o_spec = pl.BlockSpec((C, None, t), lambda j, kb: (0, j, kb))
        vec = pl.BlockSpec((C,), lambda i, jb: (0,))
        mat = pl.BlockSpec((C, C), lambda i, jb: (0, 0))
        return pl.pallas_call(
            _k1b_kernel, grid=(N, N // t),
            in_specs=[x_spec, m_spec, vec, vec, mat, mat, mat, mat, vec, vec, vec, vec],
            out_specs=[o_spec, o_spec],
            out_shape=[jax.ShapeDtypeStruct((C, N, N), act.dtype), jax.ShapeDtypeStruct((C, N, N), act.dtype)],
            compiler_params=_CP(num_warps=num_warps, num_stages=num_stages),
            name="trimul_prologue_bias",
        )(act, mask, ln_scale, ln_offset, wpa, wga, wpb, wgb, bpa, bga, bpb, bgb)

    def _k2b_kernel(t_ref, x_ref, si_ref, oi_ref, sc_ref, oc_ref, wo_ref, wg_ref, bo_ref, bg_ref, out_ref):
        tt = t_ref[...].astype(F32).T                                # [C, T] -> [T, C]
        y = K._ln(tt, sc_ref[...], oc_ref[...]).astype(BF16)         # centre norm over channels
        z = K._dot(y, wo_ref[...]) + bo_ref[...][None, :]            # output projection + bias
        xn = K._ln(x_ref[...].astype(F32), si_ref[...], oi_ref[...]).astype(BF16)
        g = jax.nn.sigmoid(K._dot(xn, wg_ref[...]) + bg_ref[...][None, :])          # gating linear + bias
        out_ref[...] = (z * g).astype(out_ref.dtype)

    def epilogue_bias(tri, act, ln_in_scale, ln_in_offset, ln_c_scale, ln_c_offset, w_out, w_gate, b_out, b_gate, *, t=64, num_warps=4, num_stages=2):
        """As the carried ``epilogue`` with the output-projection and gating bias vectors [C] f32."""
        N, _, C = act.shape
        assert tri.shape == (C, N, N) and N % t == 0
        vec = pl.BlockSpec((C,), lambda i, jb: (0,))
        mat = pl.BlockSpec((C, C), lambda i, jb: (0, 0))
        return pl.pallas_call(
            _k2b_kernel, grid=(N, N // t),
            in_specs=[pl.BlockSpec((C, None, t), lambda i, jb: (0, i, jb)),
                      pl.BlockSpec((None, t, C), lambda i, jb: (i, jb, 0)),
                      vec, vec, vec, vec, mat, mat, vec, vec],
            out_specs=pl.BlockSpec((None, t, C), lambda i, jb: (i, jb, 0)),
            out_shape=jax.ShapeDtypeStruct((N, N, C), act.dtype),
            compiler_params=_CP(num_warps=num_warps, num_stages=num_stages),
            name="trimul_epilogue_bias",
        )(tri, act, ln_in_scale, ln_in_offset, ln_c_scale, ln_c_offset, w_out, w_gate, b_out, b_gate)

    def triangle_multiplication_fused_bias(act, mask, p, *, equation, cfg=None):
        """The carried block's contract + ``b_proj[2C]``, ``b_gate[2C]`` (interleaved), ``b_out[C]``, ``b_gl[C]``."""
        cfg = {**K.DEFAULT_CFG, **(cfg or {})}
        assert equation in ("ikc,jkc->ijc", "kjc,kic->ijc"), equation
        incoming = equation == "kjc,kic->ijc"
        dt = act.dtype
        wpa, wga, wpb, wgb = [w.astype(dt) for w in K.split_glu_weights(p["w_proj"], p["w_gate"])]
        bpa, bga, bpb, bgb = [b.astype(F32) for b in (p["b_proj"][0::2], p["b_gate"][0::2], p["b_proj"][1::2], p["b_gate"][1::2])]
        A_, B_ = prologue_bias(act, mask.astype(dt), p["ln_in_scale"].astype(F32), p["ln_in_offset"].astype(F32), wpa, wga, wpb, wgb, bpa, bga, bpb, bgb,
                               transpose_out=incoming, t=cfg["t1"], num_warps=cfg["w1"], num_stages=cfg["s1"])
        P, Q = (B_, A_) if incoming else (A_, B_)
        if cfg["ein"] == "pallas":
            tri = K.einsum_pallas(P, Q, tm=cfg["tm"], tn=cfg["tn"], tk=cfg["tk"], num_warps=cfg["we"], num_stages=cfg["se"])
        elif cfg.get("f32_tri"):
            tri = jnp.einsum("cik,cjk->cij", P, Q, preferred_element_type=F32)
        else:
            tri = jnp.einsum("cik,cjk->cij", P, Q)
        return epilogue_bias(tri, act, p["ln_in_scale"].astype(F32), p["ln_in_offset"].astype(F32), p["ln_c_scale"].astype(F32), p["ln_c_offset"].astype(F32),
                             p["w_out"].astype(dt), p["w_gl"].astype(dt), p["b_out"].astype(F32), p["b_gl"].astype(F32),
                             t=cfg["t2"], num_warps=cfg["w2"], num_stages=cfg["s2"])

    # ------------------------------------------------------------------------------------------------ triangle attention
    def _pa2b_kernel(o_ref, x_ref, s_ref, off_ref, wg_ref, wo_ref, bg_ref, bo_ref, out_ref):
        xn = A._ln(x_ref[...].astype(F32), s_ref[...], off_ref[...]).astype(BF16)  # recompute the LayerNorm for the gate
        g = jax.nn.sigmoid(A._dot(xn, wg_ref[...]) + bg_ref[...][None, :])        # [t, HD]
        wa = (o_ref[...].astype(F32) * g).astype(BF16)
        out_ref[...] = (A._dot(wa, wo_ref[...]) + bo_ref[...][None, :]).astype(out_ref.dtype)   # [t, C]

    def attn_epilogue_bias(o, act, ln_scale, ln_offset, wg_t, wo, bg, bo, *, transpose, t=64, num_warps=4, num_stages=2):
        """As the carried ``attn_epilogue`` with the gating bias [HD] and output bias [C] (f32)."""
        N, _, C = act.shape; HD = o.shape[-1]
        o_spec = pl.BlockSpec((None, t, HD), lambda i, jb: (i, jb, 0))
        if not transpose:
            x_spec = pl.BlockSpec((None, t, C), lambda i, jb: (i, jb, 0)); out_spec = pl.BlockSpec((None, t, C), lambda i, jb: (i, jb, 0))
        else:
            x_spec = pl.BlockSpec((t, None, C), lambda i, jb: (jb, i, 0)); out_spec = pl.BlockSpec((t, None, C), lambda i, jb: (jb, i, 0))
        vec = pl.BlockSpec((C,), lambda i, jb: (0,))
        return pl.pallas_call(
            _pa2b_kernel, grid=(N, N // t),
            in_specs=[o_spec, x_spec, vec, vec, pl.BlockSpec((C, HD), lambda i, jb: (0, 0)), pl.BlockSpec((HD, C), lambda i, jb: (0, 0)),
                      pl.BlockSpec((HD,), lambda i, jb: (0,)), vec],
            out_specs=out_spec, out_shape=jax.ShapeDtypeStruct((N, N, C), act.dtype),
            compiler_params=_CP(num_warps=num_warps, num_stages=num_stages), name="triattn_epilogue_bias")(o, act, ln_scale, ln_offset, wg_t, wo, bg, bo)

    def grid_self_attention_fused_bias(act, pair_mask, kp, *, transpose, ending_bias_transposed=True, cfg=None):
        """The carried block's contract + ``kp['bg'] [HD]``, ``kp['bo'] [C]``."""
        N, _, C = act.shape; H, D = kp["H"], kp["D"]
        c = {**A.DEFAULT_ATTN_CFG, **A.ATTN_CFG_BY_N.get(N, {}), **(cfg or {})}
        q, k, v, braw = A.attn_prologue(act, kp["ln_scale"], kp["ln_offset"], kp["wq_t"], kp["wk_t"], kp["wv2"], kp["wb16"], transpose=transpose, t=c["t1"], num_warps=c["w1"])
        bias = jnp.transpose(braw[:, :, :H], (2, 0, 1))
        if transpose and ending_bias_transposed:
            bias = jnp.swapaxes(bias, -1, -2)
        mask2 = jnp.swapaxes(pair_mask, -1, -2) > 0
        o = A.flash_attention_bshd(q.reshape(N, N, H, D), k.reshape(N, N, H, D), v.reshape(N, N, H, D), bias[None], mask2,
                                   bq=c["bq"], bk=c["bk"], num_warps=c["wa"], num_stages=c["sa"], out_dtype=(F32 if c.get("f32_o") else None)).reshape(N, N, H * D)
        return attn_epilogue_bias(o, act, kp["ln_scale"], kp["ln_offset"], kp["wg_t"], kp["wo"], kp["bg"].astype(F32), kp["bo"].astype(F32),
                                  transpose=transpose, t=c["t2"], num_warps=c["w2"])

    ns = SimpleNamespace(prologue_bias=prologue_bias, epilogue_bias=epilogue_bias, triangle_multiplication_fused_bias=triangle_multiplication_fused_bias,
                         attn_epilogue_bias=attn_epilogue_bias, grid_self_attention_fused_bias=grid_self_attention_fused_bias)
    _NS["ns"] = ns
    return ns


def has_bias(params: Dict[str, Any], kind: str) -> Optional[bool]:
    """``True`` when ``params`` carries every bias array of ``kind`` (``"trimul"`` | ``"triattn"``), ``False`` when it carries none, ``None``
    when it carries some but not all (a refusal for the caller to name: ``bias_keys_incomplete``)."""
    keys = TRIMUL_BIAS_KEYS if kind == "trimul" else ATTN_BIAS_KEYS
    present = [k for k in keys if k in params and params[k] is not None]
    if not present:
        return False
    return True if len(present) == len(keys) else None


def triangle_multiplication_fused_bias(act, mask, p, *, equation, cfg=None):
    return _ns().triangle_multiplication_fused_bias(act, mask, p, equation=equation, cfg=cfg)


def grid_self_attention_fused_bias(act, pair_mask, kp, *, transpose, ending_bias_transposed=True, cfg=None):
    return _ns().grid_self_attention_fused_bias(act, pair_mask, kp, transpose=transpose, ending_bias_transposed=ending_bias_transposed, cfg=cfg)
