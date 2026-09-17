"""triattn_pallas.py - Pallas (Triton) flash-attention forward for GridSelfAttention (triangle attention), inference only.
q,k,v in the model's native layout [B, S, H, D] (B = pair rows, S = pair cols, H = 4, D = 32), bias [1, H, S, S] (shared over B), key mask [B, 1, 1, S] bool.
Online softmax in the exp2 domain, f32 accumulation, finite negative for masked keys (fully-masked rows -> uniform average, as the stock softmax; no NaN).
Requires S % bq == 0 and S % bk == 0 (the engine's token buckets are multiples of 256)."""
import functools, math
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
_CP = getattr(plgpu, "CompilerParams", None) or getattr(plgpu, "TritonCompilerParams")
F32 = jnp.float32
LOG2E = math.log2(math.e)
NEG = -1e30


def _dot(a, b):
    return jnp.dot(a, b, preferred_element_type=F32)


def _fa_kernel(q_ref, k_ref, v_ref, b_ref, m_ref, o_ref, *, bk, nk, qk_scale):
    bq, D = q_ref.shape
    q = q_ref[...]
    m_i = jnp.full((bq,), NEG, F32); l_i = jnp.zeros((bq,), F32); acc = jnp.zeros((bq, D), F32)

    def body(j, carry):
        acc, m_i, l_i = carry
        ks = pl.ds(j * bk, bk)
        k = k_ref[ks, :]; v = v_ref[ks, :]
        s = _dot(q, k.T) * qk_scale + b_ref[:, ks].astype(F32) * LOG2E        # [bq, bk], log2 domain
        km = m_ref[ks]
        s = jnp.where(km[None, :], s, NEG)
        m_new = jnp.maximum(m_i, jnp.max(s, axis=1))
        alpha = jnp.exp2(m_i - m_new)
        p = jnp.exp2(s - m_new[:, None])
        l_new = alpha * l_i + p.sum(axis=1)
        acc = acc * alpha[:, None] + _dot(p.astype(v.dtype), v)
        return acc, m_new, l_new

    acc, m_i, l_i = jax.lax.fori_loop(0, nk, body, (acc, m_i, l_i))
    o_ref[...] = (acc / l_i[:, None]).astype(o_ref.dtype)


def flash_attention_bshd(q, k, v, bias, mask, *, bq=128, bk=64, num_warps=4, num_stages=2, out_dtype=None):
    """q,k,v [B,S,H,D]; bias [1,H,S,S] or [H,S,S]; mask bool broadcastable from [B,1,1,S] (key mask per batch row) -> o [B,S,H,D]."""
    B, S, H, D = q.shape
    assert k.shape == (B, S, H, D) and v.shape == (B, S, H, D), (q.shape, k.shape, v.shape)
    assert S % bq == 0 and S % bk == 0, (S, bq, bk)
    if bias.ndim == 4:
        assert bias.shape[0] == 1, bias.shape
        bias = bias[0]
    assert bias.shape == (H, S, S), bias.shape
    mask = jnp.broadcast_to(jnp.reshape(mask, (mask.shape[0], -1))[:, -S:], (B, S)) if mask.ndim > 2 else jnp.broadcast_to(mask, (B, S))
    qk_scale = LOG2E / math.sqrt(D)
    grid = (S // bq, H, B)      # batch innermost: the bias tile [h, q-block, :] stays hot in L2 across the B sweep
    return pl.pallas_call(
        functools.partial(_fa_kernel, bk=bk, nk=S // bk, qk_scale=qk_scale), grid=grid,
        in_specs=[pl.BlockSpec((None, bq, None, D), lambda i, h, b: (b, i, h, 0)),
                  pl.BlockSpec((None, S, None, D), lambda i, h, b: (b, 0, h, 0)),
                  pl.BlockSpec((None, S, None, D), lambda i, h, b: (b, 0, h, 0)),
                  pl.BlockSpec((None, bq, S), lambda i, h, b: (h, i, 0)),
                  pl.BlockSpec((None, S), lambda i, h, b: (b, 0))],
        out_specs=pl.BlockSpec((None, bq, None, D), lambda i, h, b: (b, i, h, 0)),
        out_shape=jax.ShapeDtypeStruct((B, S, H, D), out_dtype or q.dtype),
        compiler_params=_CP(num_warps=num_warps, num_stages=num_stages),
        name="triattn_fwd",
    )(q, k, v, bias, mask.astype(jnp.bool_))


def reference_attention_bshd(q, k, v, bias, mask, precision=None, chunk=64):
    """f32 reference, chunked over B to bound memory: softmax((q k^T)/sqrt(D) + bias, masked keys -> -1e9-ish) v."""
    B, S, H, D = q.shape
    bias = bias.reshape((-1,) + bias.shape[-3:])[0].astype(F32)
    mask2 = jnp.reshape(mask, (mask.shape[0], -1))[:, -S:] if mask.ndim > 2 else mask
    mask2 = jnp.broadcast_to(mask2, (B, S))
    outs = []
    for b0 in range(0, B, chunk):
        qq = q[b0:b0 + chunk].astype(F32); kk = k[b0:b0 + chunk].astype(F32); vv = v[b0:b0 + chunk].astype(F32)
        logits = jnp.einsum("bqhd,bkhd->bhqk", qq / math.sqrt(D), kk, precision=precision) + bias[None]
        logits = jnp.where(mask2[b0:b0 + chunk][:, None, None, :], logits, -1e9)
        w = jax.nn.softmax(logits, axis=-1)
        outs.append(jnp.einsum("bhqk,bkhd->bqhd", w, vv, precision=precision))
    return jnp.concatenate(outs, 0)


# =========================================================================================== fused GridSelfAttention (prologue / epilogue)
BF16 = jnp.bfloat16
LN_EPS = 1e-5


def _ln(x, scale, offset):
    mean = jnp.mean(x, axis=1, keepdims=True)
    var = jnp.mean(x * x, axis=1, keepdims=True) - mean * mean
    return (x - mean) * (jax.lax.rsqrt(var + LN_EPS) * scale[None, :]) + offset[None, :]


def _pa1_kernel(x_ref, s_ref, o_ref, wq_ref, wk_ref, wv_ref, wb_ref, q_ref, k_ref, v_ref, b_ref):
    xn = _ln(x_ref[...].astype(F32), s_ref[...], o_ref[...]).astype(BF16)      # act_norm output is bf16 in the stock module
    q_ref[...] = _dot(xn, wq_ref[...]).astype(q_ref.dtype)
    k_ref[...] = _dot(xn, wk_ref[...]).astype(k_ref.dtype)
    v_ref[...] = _dot(xn, wv_ref[...]).astype(v_ref.dtype)
    b_ref[...] = _dot(xn, wb_ref[...]).astype(b_ref.dtype)                     # pair-bias logits, H padded to 16 columns


def attn_prologue(act, ln_scale, ln_offset, wq_t, wk_t, wv2, wb16, *, transpose, t=64, num_warps=4, num_stages=2):
    """act (N,N,C) -> q,k,v (N,N,HD) in [batch,seq,(h d)] layout of the attention (batch = row of act, or column if transpose), bias_raw (N,N,16) indexed by
    UN-transposed pixel (a,b). transpose=True reads act transposed through the BlockSpec (no materialised swapaxes)."""
    N, _, C = act.shape; HD = wq_t.shape[1]; NB = wb16.shape[1]
    assert N % t == 0
    if not transpose:      # program (i, jb): pixels (i, j-seg) -> q[b=i, s=j-seg]
        x_spec = pl.BlockSpec((None, t, C), lambda i, jb: (i, jb, 0)); b_spec = pl.BlockSpec((None, t, NB), lambda i, jb: (i, jb, 0))
    else:                  # program (i, jb): pixels (j-seg, i) [strided read] -> q[b=i, s=j-seg] ; bias_raw[a=j-seg, b=i]
        x_spec = pl.BlockSpec((t, None, C), lambda i, jb: (jb, i, 0)); b_spec = pl.BlockSpec((t, None, NB), lambda i, jb: (jb, i, 0))
    qkv_spec = pl.BlockSpec((None, t, HD), lambda i, jb: (i, jb, 0))
    vec = pl.BlockSpec((C,), lambda i, jb: (0,)); mat = pl.BlockSpec((C, HD), lambda i, jb: (0, 0)); matb = pl.BlockSpec((C, NB), lambda i, jb: (0, 0))
    sd = jax.ShapeDtypeStruct
    return pl.pallas_call(
        _pa1_kernel, grid=(N, N // t), in_specs=[x_spec, vec, vec, mat, mat, mat, matb], out_specs=[qkv_spec, qkv_spec, qkv_spec, b_spec],
        out_shape=[sd((N, N, HD), act.dtype), sd((N, N, HD), act.dtype), sd((N, N, HD), act.dtype), sd((N, N, NB), act.dtype)],
        compiler_params=_CP(num_warps=num_warps, num_stages=num_stages), name="triattn_prologue")(act, ln_scale, ln_offset, wq_t, wk_t, wv2, wb16)


def _pa2_kernel(o_ref, x_ref, s_ref, off_ref, wg_ref, wo_ref, out_ref):
    xn = _ln(x_ref[...].astype(F32), s_ref[...], off_ref[...]).astype(BF16)    # recompute act_norm for the gate instead of storing it
    g = jax.nn.sigmoid(_dot(xn, wg_ref[...]))                                 # [t, HD]
    wa = (o_ref[...].astype(F32) * g).astype(BF16)
    out_ref[...] = _dot(wa, wo_ref[...]).astype(out_ref.dtype)                # [t, C]


def attn_epilogue(o, act, ln_scale, ln_offset, wg_t, wo, *, transpose, t=64, num_warps=4, num_stages=2):
    """o (N,N,HD) attention output [b,s,(h d)] -> module output (N,N,C) in the ORIGINAL (un-transposed) frame."""
    N, _, C = act.shape; HD = o.shape[-1]
    o_spec = pl.BlockSpec((None, t, HD), lambda i, jb: (i, jb, 0))
    if not transpose:
        x_spec = pl.BlockSpec((None, t, C), lambda i, jb: (i, jb, 0)); out_spec = pl.BlockSpec((None, t, C), lambda i, jb: (i, jb, 0))
    else:                  # o[b=i, s=j] -> out pixel (j, i); gate from act[j, i]
        x_spec = pl.BlockSpec((t, None, C), lambda i, jb: (jb, i, 0)); out_spec = pl.BlockSpec((t, None, C), lambda i, jb: (jb, i, 0))
    vec = pl.BlockSpec((C,), lambda i, jb: (0,))
    return pl.pallas_call(
        _pa2_kernel, grid=(N, N // t), in_specs=[o_spec, x_spec, vec, vec, pl.BlockSpec((C, HD), lambda i, jb: (0, 0)), pl.BlockSpec((HD, C), lambda i, jb: (0, 0))],
        out_specs=out_spec, out_shape=jax.ShapeDtypeStruct((N, N, C), act.dtype),
        compiler_params=_CP(num_warps=num_warps, num_stages=num_stages), name="triattn_epilogue")(o, act, ln_scale, ln_offset, wg_t, wo)


DEFAULT_ATTN_CFG = dict(t1=64, w1=4, t2=64, w2=4, bq=64, bk=64, wa=4, sa=3)
ATTN_CFG_BY_N = {1024: dict(bq=128, bk=32, wa=4, sa=3)}


def attn_params_from_haiku(hp):
    """hp: {'act_norm': {scale, offset}, 'pair_bias_projection': {weights (C,H)}, 'q_projection': {weights (H,D,C)}, 'k_projection', 'v_projection': {weights (C,H,D)},
    'gating_query': {weights (HD, C)}, 'output_projection': {weights (HD, C)}} -> kernel-side arrays (weights in bf16 = the model's bfloat16 getter)."""
    wq = hp["q_projection"]["weights"]; H, D, C = wq.shape; HD = H * D
    wb = hp["pair_bias_projection"]["weights"]                      # (C, H)
    wb16 = jnp.zeros((C, 16), wb.dtype).at[:, :H].set(wb)
    return dict(ln_scale=hp["act_norm"]["scale"].astype(F32), ln_offset=hp["act_norm"]["offset"].astype(F32),
                wq_t=wq.reshape(HD, C).T.astype(BF16), wk_t=hp["k_projection"]["weights"].reshape(HD, C).T.astype(BF16),
                wv2=hp["v_projection"]["weights"].reshape(C, HD).astype(BF16), wb16=wb16.astype(BF16),
                wg_t=hp["gating_query"]["weights"].T.astype(BF16), wo=hp["output_projection"]["weights"].astype(BF16), H=H, D=D)


def grid_self_attention_fused(act, pair_mask, kp, *, transpose, ending_bias_transposed=True, cfg=None):
    """Drop-in for modules.GridSelfAttention.__call__ (inference): act (N,N,C) bf16, pair_mask (N,N) -> (N,N,C)."""
    N, _, C = act.shape; H, D = kp["H"], kp["D"]
    c = {**DEFAULT_ATTN_CFG, **ATTN_CFG_BY_N.get(N, {}), **(cfg or {})}
    q, k, v, braw = attn_prologue(act, kp["ln_scale"], kp["ln_offset"], kp["wq_t"], kp["wk_t"], kp["wv2"], kp["wb16"], transpose=transpose, t=c["t1"], num_warps=c["w1"])
    bias = jnp.transpose(braw[:, :, :H], (2, 0, 1))                       # (H, N, N) from the un-transposed normalised act, as stock
    if transpose and ending_bias_transposed:
        bias = jnp.swapaxes(bias, -1, -2)
    mask2 = jnp.swapaxes(pair_mask, -1, -2) > 0                            # stock: pair_mask = swapaxes(pair_mask); mask = pair_mask[:, None, None, :]
    o = flash_attention_bshd(q.reshape(N, N, H, D), k.reshape(N, N, H, D), v.reshape(N, N, H, D), bias[None], mask2,
                             bq=c["bq"], bk=c["bk"], num_warps=c["wa"], num_stages=c["sa"], out_dtype=(F32 if c.get("f32_o") else None)).reshape(N, N, H * D)
    return attn_epilogue(o, act, kp["ln_scale"], kp["ln_offset"], kp["wg_t"], kp["wo"], transpose=transpose, t=c["t2"], num_warps=c["w2"])


# =========================================================================================== flash attention v2: pre-scaled q / bias, additive key mask
def _fa2_kernel(q_ref, k_ref, v_ref, b_ref, mb_ref, o_ref, *, bk, nk):
    bq, D = q_ref.shape
    q = q_ref[...]                                     # already scaled by log2(e)/sqrt(D) (folded into Wq)
    m_i = jnp.full((bq,), NEG, F32); l_i = jnp.zeros((bq,), F32); acc = jnp.zeros((bq, D), F32)

    def body(j, carry):
        acc, m_i, l_i = carry
        ks = pl.ds(j * bk, bk)
        k = k_ref[ks, :]; v = v_ref[ks, :]
        s = _dot(q, k.T) + b_ref[:, ks].astype(F32) + mb_ref[ks][None, :]      # bias pre-scaled by log2(e); mb = 0 (valid) / NEG (masked key)
        m_new = jnp.maximum(m_i, jnp.max(s, axis=1))
        alpha = jnp.exp2(m_i - m_new)
        p = jnp.exp2(s - m_new[:, None])
        l_new = alpha * l_i + p.sum(axis=1)
        acc = acc * alpha[:, None] + _dot(p.astype(v.dtype), v)
        return acc, m_new, l_new

    acc, m_i, l_i = jax.lax.fori_loop(0, nk, body, (acc, m_i, l_i))
    o_ref[...] = (acc / l_i[:, None]).astype(o_ref.dtype)


def flash_attention_bshd_v2(q, k, v, bias_l2, mask_bias, *, bq=128, bk=64, num_warps=4, num_stages=2):
    """q pre-scaled [B,S,H,D]; bias_l2 [H,S,S] pre-scaled by log2(e); mask_bias [B,S] f32 (0 / NEG) -> o [B,S,H,D]."""
    B, S, H, D = q.shape
    assert S % bq == 0 and S % bk == 0, (S, bq, bk)
    return pl.pallas_call(
        functools.partial(_fa2_kernel, bk=bk, nk=S // bk), grid=(S // bq, H, B),
        in_specs=[pl.BlockSpec((None, bq, None, D), lambda i, h, b: (b, i, h, 0)),
                  pl.BlockSpec((None, S, None, D), lambda i, h, b: (b, 0, h, 0)),
                  pl.BlockSpec((None, S, None, D), lambda i, h, b: (b, 0, h, 0)),
                  pl.BlockSpec((None, bq, S), lambda i, h, b: (h, i, 0)),
                  pl.BlockSpec((None, S), lambda i, h, b: (b, 0))],
        out_specs=pl.BlockSpec((None, bq, None, D), lambda i, h, b: (b, i, h, 0)),
        out_shape=jax.ShapeDtypeStruct((B, S, H, D), q.dtype),
        compiler_params=_CP(num_warps=num_warps, num_stages=num_stages), name="triattn_fwd_v2",
    )(q, k, v, bias_l2, mask_bias)


def grid_self_attention_fused_v2(act, pair_mask, kp, *, transpose, ending_bias_transposed=True, cfg=None):
    """As grid_self_attention_fused, but the softmax scale*log2(e) is folded into Wq and log2(e) into the pair-bias weights (free, in the prologue),
    and the key mask is additive. kp from attn_params_from_haiku (unscaled); scaling applied here (tiny weight ops, fused by XLA)."""
    N, _, C = act.shape; H, D = kp["H"], kp["D"]
    c = {**DEFAULT_ATTN_CFG, **ATTN_CFG_BY_N.get(N, {}), **(cfg or {})}
    wq_s = (kp["wq_t"].astype(F32) * (LOG2E / math.sqrt(D))).astype(BF16); wb_s = (kp["wb16"].astype(F32) * LOG2E).astype(BF16)
    q, k, v, braw = attn_prologue(act, kp["ln_scale"], kp["ln_offset"], wq_s, kp["wk_t"], kp["wv2"], wb_s, transpose=transpose, t=c["t1"], num_warps=c["w1"])
    bias = jnp.transpose(braw[:, :, :H], (2, 0, 1))
    if transpose and ending_bias_transposed:
        bias = jnp.swapaxes(bias, -1, -2)
    mask2 = jnp.swapaxes(pair_mask, -1, -2) > 0
    mb = jnp.where(mask2, 0.0, NEG).astype(F32)
    o = flash_attention_bshd_v2(q.reshape(N, N, H, D), k.reshape(N, N, H, D), v.reshape(N, N, H, D), bias, mb,
                                bq=c["bq"], bk=c["bk"], num_warps=c["wa"], num_stages=c["sa"]).reshape(N, N, H * D)
    return attn_epilogue(o, act, kp["ln_scale"], kp["ln_offset"], kp["wg_t"], kp["wo"], transpose=transpose, t=c["t2"], num_warps=c["w2"])
