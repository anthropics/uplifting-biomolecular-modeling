"""af3_jax_atom_attn_pallas — fused windowed attention for AlphaFold 3's atom cross-attention (Pallas, Triton backend; fwd only).
Self-contained: imports jax / Pallas only; inprocess/atom_attn.py loads it by file path.

The stock core it replaces (alphafold3/model/network/diffusion_transformer.py ``cross_attention``, between the projections and the gate):
    bias    = 1e9 * (mask_q - 1)[..., None, :, None] * (mask_k - 1)[..., None, None, :]      # + only where query AND key are padding
    logits  = einsum('...qhc,...khc->...hqk', f32(q) * key_dim**-0.5, f32(k)) + bias + pair_logits     # [S, H, 32, 128] f32
    weights = softmax(logits).astype(x_q.dtype)
    out     = einsum('...hqk,...khc->...qhc', weights, v).reshape(..., H*Dv)
One program per (query subset [, vmapped sample]) computes all H heads: per head Q.K^T over the subset's 32 queries x 128 keys (f32 logits,
the operand precision a word: 'f32' = the dot's default precision on f32 operands, XLA's own class; 'f32hi' = HIGHEST; 'bf16' = bf16
operands), + the stock mask term + the block's pair logits, a plain softmax over the 128 keys with f32 statistics (no online re-association:
the whole key window is one tile), the weights cast to the activation dtype as stock does, P.V with f32 accumulation, the head's output
written into its Dv columns. ``skip``: a query subset whose 32 query atoms are ALL padding writes zeros and exits — its rows are
unobservable downstream (the encoder/decoder multiply the transformer output by queries_mask; key windows gather real atoms only, so no
padded query row is ever a key; transitions / layer norms are row-wise), stock computes them in full. A leading sample axis arrives through
vmap: the pallas_call batching rule prepends a grid axis and indexes the unbatched masks / pair logits without copying them.
Numerics: NOT bitwise with stock (summation order inside the two products, exp implementation; the padded-subset rows differ by design) —
a tolerance-class lever; max |delta| vs an f32 HIGHEST reference is of the order of stock's own (both round the weights and the output to bf16).
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

F32, BF16 = jnp.float32, jnp.bfloat16
_CP = getattr(plgpu, "CompilerParams", None) or getattr(plgpu, "TritonCompilerParams")
DEFAULT_CFG = dict(layout="hgrid", qk_prec="f32", skip=True, num_warps=2, num_stages=2)   # hgrid: one program per (query subset, head), 2 warps; sloop (atom_attn_pallas)
# loops all heads in one program and is used with skip=True
SUPPORTED = dict(n_queries=(16, 32, 64), n_keys=(32, 64, 128, 256), head_dim=(16, 32, 64, 128))   # power-of-two tiles the Triton dots accept


def eligible(q_shape, k_shape, v_shape, pair_logits_shape) -> str:
    """'' when the kernel serves these shapes (q [S,Q,H,D], k [S,K,H,D], v [S,K,H,Dv], pair_logits [S,H,Q,K]), else the reason word."""
    if len(q_shape) != 4 or len(k_shape) != 4 or len(v_shape) != 4:
        return f"rank(q{len(q_shape)},k{len(k_shape)},v{len(v_shape)})!=4"
    S, Q, H, D = q_shape; _, K, Hk, Dk = k_shape; _, Kv, Hv, Dv = v_shape
    if (Hk, Dk) != (H, D) or (Kv, Hv) != (K, H) or k_shape[0] != S or v_shape[0] != S:
        return "head/shape_mismatch"
    if Q not in SUPPORTED["n_queries"] or K not in SUPPORTED["n_keys"] or D not in SUPPORTED["head_dim"] or Dv not in SUPPORTED["head_dim"]:
        return f"tile(q{Q},k{K},d{D},dv{Dv})"
    if pair_logits_shape is None or tuple(pair_logits_shape) != (S, H, Q, K):
        return f"pair_logits{tuple(pair_logits_shape) if pair_logits_shape is not None else None}"
    return ""


def _atom_kernel(mq_ref, mk_ref, q_ref, k_ref, v_ref, b_ref, o_ref, *, n_head, scale, qk_prec, skip):
    mq = mq_ref[...]                                                     # [Q] f32 query mask of this subset

    def compute():
        mk = mk_ref[...]                                                 # [K] f32 key mask
        mb = 1e9 * (mq - 1.0)[:, None] * (mk - 1.0)[None, :]             # stock's mask term: + only where query AND key are padding
        for h in range(n_head):
            qh = q_ref[:, h, :]                                          # [Q, D]
            kh = k_ref[:, h, :]                                          # [K, D]
            vh = v_ref[:, h, :]                                          # [K, Dv]
            if qk_prec == "bf16":
                s = jnp.dot((qh.astype(F32) * scale).astype(BF16), kh.astype(BF16).T, preferred_element_type=F32)
            elif qk_prec == "f32hi":
                s = jnp.dot(qh.astype(F32) * scale, kh.astype(F32).T, preferred_element_type=F32, precision=jax.lax.Precision.HIGHEST)
            else:                                                        # "f32": the default precision of an f32 dot (XLA's stock einsum is an f32 GEMM at default precision)
                s = jnp.dot(qh.astype(F32) * scale, kh.astype(F32).T, preferred_element_type=F32)
            s = s + mb + b_ref[h].astype(F32)                            # [Q, K] f32 logits
            m = jnp.max(s, axis=1)
            p = jnp.exp(s - m[:, None])
            l = jnp.sum(p, axis=1)
            w = (p / l[:, None]).astype(vh.dtype)                        # stock: softmax in f32, weights cast to the activation dtype
            oh = jnp.dot(w, vh, preferred_element_type=F32)              # [Q, Dv], f32 accumulation
            o_ref[:, h, :] = oh.astype(o_ref.dtype)

    if skip:
        live = jnp.max(mq) > 0.0
        pl.when(live)(compute)

        @pl.when(jnp.logical_not(live))
        def _():
            o_ref[...] = jnp.zeros(o_ref.shape, o_ref.dtype)
    else:
        compute()


def atom_attn(q, k, v, mask_q, mask_k, pair_logits, *, scale=None, layout=None, **kw):
    """The lever's entry: DEFAULT_CFG['layout'] picks the program decomposition (hgrid: one program per (subset, head); sloop: per subset, heads looped)."""
    lay = layout or DEFAULT_CFG["layout"]
    return (atom_attn_pallas_hgrid if lay == "hgrid" else atom_attn_pallas)(q, k, v, mask_q, mask_k, pair_logits, scale=scale, **kw)


def atom_attn_pallas(q, k, v, mask_q, mask_k, pair_logits, *, scale=None, qk_prec=None, skip=None, num_warps=None, num_stages=None):
    """q [S,Q,H,D], k [S,K,H,D], v [S,K,H,Dv] (activation dtype), mask_q [S,Q], mask_k [S,K], pair_logits [S,H,Q,K] -> [S,Q,H*Dv] in q's dtype."""
    cfg = dict(DEFAULT_CFG)
    cfg.update({k_: v_ for k_, v_ in dict(qk_prec=qk_prec, skip=skip, num_warps=num_warps, num_stages=num_stages).items() if v_ is not None})
    S, nq, nh, d = q.shape; nk = k.shape[-3]; dv = v.shape[-1]
    scale = float(d) ** -0.5 if scale is None else float(scale)
    kern = functools.partial(_atom_kernel, n_head=nh, scale=scale, qk_prec=cfg["qk_prec"], skip=bool(cfg["skip"]))
    out = pl.pallas_call(
        kern, out_shape=jax.ShapeDtypeStruct((S, nq, nh, dv), q.dtype), grid=(S,),
        in_specs=[pl.BlockSpec((None, nq), lambda s: (s, 0)), pl.BlockSpec((None, nk), lambda s: (s, 0)),
                  pl.BlockSpec((None, nq, nh, d), lambda s: (s, 0, 0, 0)), pl.BlockSpec((None, nk, nh, d), lambda s: (s, 0, 0, 0)),
                  pl.BlockSpec((None, nk, nh, dv), lambda s: (s, 0, 0, 0)), pl.BlockSpec((None, nh, nq, nk), lambda s: (s, 0, 0, 0))],
        out_specs=pl.BlockSpec((None, nq, nh, dv), lambda s: (s, 0, 0, 0)),
        compiler_params=_CP(num_warps=int(cfg["num_warps"]), num_stages=int(cfg["num_stages"])),
        name=f"af3_jax_atom_attn_q{nq}k{nk}h{nh}d{d}",
    )(mask_q.astype(F32), mask_k.astype(F32), q, k, v.astype(q.dtype), pair_logits)
    return jnp.reshape(out, (S, nq, nh * dv))


def _atom_kernel_h(mq_ref, mk_ref, q_ref, k_ref, v_ref, b_ref, o_ref, *, scale, qk_prec, skip):
    """One program per (subset, head [, sample]): q [Q,D], k [K,D], v [K,Dv], bias [Q,K]."""
    mq = mq_ref[...]

    def compute():
        mk = mk_ref[...]
        mb = 1e9 * (mq - 1.0)[:, None] * (mk - 1.0)[None, :]
        qh = q_ref[...]; kh = k_ref[...]; vh = v_ref[...]
        if qk_prec == "bf16":
            sc = jnp.dot((qh.astype(F32) * scale).astype(BF16), kh.astype(BF16).T, preferred_element_type=F32)
        elif qk_prec == "f32hi":
            sc = jnp.dot(qh.astype(F32) * scale, kh.astype(F32).T, preferred_element_type=F32, precision=jax.lax.Precision.HIGHEST)
        else:
            sc = jnp.dot(qh.astype(F32) * scale, kh.astype(F32).T, preferred_element_type=F32)
        sc = sc + mb + b_ref[...].astype(F32)
        m = jnp.max(sc, axis=1)
        p = jnp.exp(sc - m[:, None])
        l = jnp.sum(p, axis=1)
        w = (p / l[:, None]).astype(vh.dtype)
        o_ref[...] = jnp.dot(w, vh, preferred_element_type=F32).astype(o_ref.dtype)

    if skip:
        live = jnp.max(mq) > 0.0
        pl.when(live)(compute)

        @pl.when(jnp.logical_not(live))
        def _():
            o_ref[...] = jnp.zeros(o_ref.shape, o_ref.dtype)
    else:
        compute()


def atom_attn_pallas_hgrid(q, k, v, mask_q, mask_k, pair_logits, *, scale=None, qk_prec=None, skip=None, num_warps=None, num_stages=None):
    """Same contract as atom_attn_pallas; the heads are a grid axis (one program per (subset, head)) instead of an in-program loop."""
    cfg = dict(DEFAULT_CFG)
    cfg.update({k_: v_ for k_, v_ in dict(qk_prec=qk_prec, skip=skip, num_warps=num_warps, num_stages=num_stages).items() if v_ is not None})
    S, nq, nh, d = q.shape; nk = k.shape[-3]; dv = v.shape[-1]
    scale = float(d) ** -0.5 if scale is None else float(scale)
    kern = functools.partial(_atom_kernel_h, scale=scale, qk_prec=cfg["qk_prec"], skip=bool(cfg["skip"]))
    out = pl.pallas_call(
        kern, out_shape=jax.ShapeDtypeStruct((S, nq, nh, dv), q.dtype), grid=(S, nh),
        in_specs=[pl.BlockSpec((None, nq), lambda s, h: (s, 0)), pl.BlockSpec((None, nk), lambda s, h: (s, 0)),
                  pl.BlockSpec((None, nq, None, d), lambda s, h: (s, 0, h, 0)), pl.BlockSpec((None, nk, None, d), lambda s, h: (s, 0, h, 0)),
                  pl.BlockSpec((None, nk, None, dv), lambda s, h: (s, 0, h, 0)), pl.BlockSpec((None, None, nq, nk), lambda s, h: (s, h, 0, 0))],
        out_specs=pl.BlockSpec((None, nq, None, dv), lambda s, h: (s, 0, h, 0)),
        compiler_params=_CP(num_warps=int(cfg["num_warps"]), num_stages=int(cfg["num_stages"])),
        name=f"af3_jax_atom_attn_h_q{nq}k{nk}h{nh}d{d}",
    )(mask_q.astype(F32), mask_k.astype(F32), q, k, v.astype(q.dtype), pair_logits)
    return jnp.reshape(out, (S, nq, nh * dv))


def atom_attn_reference(q, k, v, mask_q, mask_k, pair_logits, *, scale=None, precision=None, weights_dtype=None):
    """Op-by-op replica of the stock core (weights_dtype=None: the activation dtype, as stock; precision=HIGHEST + f32 inputs = the reference)."""
    scale = float(q.shape[-1]) ** -0.5 if scale is None else float(scale)
    bias = 1e9 * (mask_q - 1.0)[..., None, :, None] * (mask_k - 1.0)[..., None, None, :]
    logits = jnp.einsum('...qhc,...khc->...hqk', q.astype(F32) * scale, k.astype(F32), precision=precision) + bias.astype(F32) + pair_logits
    weights = jax.nn.softmax(logits, axis=-1).astype(weights_dtype or q.dtype)
    wa = jnp.einsum('...hqk,...khc->...qhc', weights, v.astype(weights.dtype), precision=precision)
    return jnp.reshape(wa, wa.shape[:-2] + (-1,))
