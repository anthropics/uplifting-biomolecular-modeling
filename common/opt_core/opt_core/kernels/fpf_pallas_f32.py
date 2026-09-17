"""float32 variants of the carried ``fpf_pallas`` pair-stack blocks — the triangle attention and the triangle multiplication — for a pair
stack that runs in float32 (a float32 monomer model's Evoformer, say).

TRIANGLE ATTENTION: the same three launches as the carried block (ONE prologue kernel LayerNorm +
q|k|v|pair-bias projections with transposed BlockSpec reads for the ending node, the flash core with online softmax in the exp2 domain,
ONE epilogue kernel LayerNorm-recompute + sigmoid gate + output projection with transposed writes for the ending node). TRIANGLE
MULTIPLICATION: the carried block's two launches around the cubic contraction (ONE prologue kernel input LayerNorm + left|right projections
(+bias) * mask * sigmoid(left|right gates (+bias)) written channel-major ``[C,N,N]`` — transposed stores for the incoming equation; the
contraction ``tri[c,i,j] = sum_k P[c,i,k] Q[c,j,k]`` as ONE XLA batched GEMM over channels; ONE epilogue kernel centre LayerNorm + output
projection (+bias) * sigmoid(gating linear (+bias) of the recomputed input LayerNorm)). Nothing is rounded to bf16 anywhere: activations, LayerNorm output,
q/k/v, the pair bias, the softmax probabilities handed to the PV product, the gated output and every kernel output stay f32; every MMA is
``f32 x f32 -> f32`` at a NAMED operand precision (``precision=``):

    The word is REQUIRED on every call (``precision=None`` / ``"std"`` raise ``precision_required``): each is a deliberate speed / accuracy class,
    none is a default —
    ``"tf32"``    Triton's default f32 dot: operands truncated to tf32 (10-bit mantissa) on the tensor cores, f32 accumulate — the class of an XLA
                f32 GEMM at default precision on sm_80+ parts (what a stock f32 model runs there), NOT f32-exact; the fast word a kit names and
                prints as ``mma=tf32`` (both blocks);
    ``"tf32x3"``  3-pass tf32 (the ``TF32_TF32_F32_X3`` dot-algorithm preset on the dots; a jax without the preset raises — ``Precision.HIGH``
                is never substituted, it lowers to plain tf32 here): f32-class accuracy; faster than the stock-like XLA body for the attention,
                SLOWER than it for the triangle multiplication (a kit wanting f32-class multiplication routes that block to stock by name);
    ``"ieee"``    f32 FMA dots (``jax.lax.Precision.HIGHEST``): f32-exact products on the CUDA cores — in Pallas-Triton a scalar FMA loop, 45-60x
                the tf32 time: validation only, never a serving word.
    The class a word lands in is per jax release line — ``fpf_pallas_serve.F32_LINES`` / ``f32_class(precision)``: on jax >= 0.10 Pallas-Triton
    rounds f32 operands to nearest tf32 (per-dot rel-rms 2.9e-4, == XLA's DEFAULT f32 GEMM: 'tf32' = the body's class); on jax 0.5 it TRUNCATES
    them (per-dot 7.7e-4; over a block the bf16-operand class: 'tf32' -> f32_class=bf16op) and the attention prologue at 'ieee' (HIGHEST)
    cannot launch (98304 B of shared memory per program; a bare 128^3 HIGHEST dot asks 131072 B) — the serve layer refuses that word for the
    attention there by name; the multiplication's 'ieee' kernels launch.  These are PER-CELL classes: end to end they
    do not propagate monotonically (a float32 model with the attention at 'tf32x3' ran +27-37% slower AND further from its stock
    run-to-run band than at 'tf32') — which word a model ships is the kit's end-to-end evidence, never read off this table.
    Timing (H100, C=128, ms per call; tf32x3 | tf32 | XLA-default f32 body | XLA HIGHEST): attention N=448 2.20 | 1.18 | 4.36 | 5.36,
    N=1024 16.05 | 8.51 | 45.8 | 54.4 (tf32x3 err 4-5e-7; tf32 2.3e-4, equal to XLA-default's); multiplication N=448 2.24 | 0.81 | 1.74 | 2.81,
    N=1024 12.67 | 3.29 | 8.22 | 15.6.
    (The triangle multiplication's cubic contraction is an XLA GEMM: ``"tf32"`` runs it at XLA's default precision, ``"tf32x3"`` and ``"ieee"``
    at ``Precision.HIGHEST`` — XLA names no 3-pass tf32 GEMM, so both words are f32-exact THERE and differ only in the Pallas kernels.)

The math helpers (``_ln``: LayerNorm with f32 statistics, eps 1e-5; ``LOG2E``, ``NEG``, the compiler-params class) are the carried
module's, imported, not re-implemented; the pallas_call wrappers are re-stated with f32 out-shapes and the bias FMAs. The gating bias
``bg [HD]`` and output bias ``bo [C]`` are kernel arguments always (a bias-free parameterisation passes zeros: one kernel body serves both
the biased layout, whose gate and output Linears carry biases, and the bias-free layout, whose Linears carry none; q/k/v and the
pair-bias projection carry no bias in either).

    grid_self_attention_fused_f32(act, pair_mask, kp, *, transpose, ending_bias_transposed=True, cfg=None, precision="tf32")
        ``act [N,N,C]`` f32, ``pair_mask [N,N]`` (key ``k`` of batch line ``b`` valid iff ``pair_mask[k, b] > 0`` — the carried rule; the
        serve layer maps the caller's convention), ``kp`` = the ``fpf_pallas_serve.attn_params(..., weights_dtype=float32)`` dict
        (``ln_scale/ln_offset [C]``, ``wq_t/wk_t/wv2/wg_t [C,HD]``, ``wb16 [C,16]``, ``wo [HD,C]``, ``H``, ``D``; optional ``bg [HD]``,
        ``bo [C]``) -> ``[N,N,C]`` f32. ``cfg``: ``t1,w1,hc1`` (prologue tile rows / warps / head-channel chunk: the q|k|v projections run
        ``HD/hc1`` dots of ``hc1`` output columns each, bounding the f32 weight tile a program holds), ``bq,bk,wa,sa`` (core), ``t2,w2,hc2``
        (epilogue; the gate and the output projection's contraction run in ``hc2``-column chunks, partial products summed in f32), ``vt``
        (1 = the prologue writes ``v`` transposed ``[B,HD,S]`` so the PV product reads a K-major operand; 0 = ``[B,S,HD]``).
    attn_prologue_f32 / flash_attention_bshd_f32 / attn_epilogue_f32: the three kernels alone (signatures below).
    triangle_multiplication_fused_f32(act, mask, p, *, equation, cfg=None, precision="tf32")
        ``act [N,N,C]`` f32, ``mask [N,N]``, ``p`` = the ``fpf_pallas_serve.trimul_params`` dict (``ln_in_scale/ln_in_offset [C]``, interleaved
        ``w_proj/w_gate [C,2C]``, ``ln_c_scale/ln_c_offset [C]``, ``w_out/w_gl [C,C]``; optional ``b_proj/b_gate [2C]``, ``b_out/b_gl [C]`` —
        zeros when absent, one kernel body), ``equation`` in {``'ikc,jkc->ijc'`` outgoing, ``'kjc,kic->ijc'`` incoming} -> ``[N,N,C]`` f32.
        ``cfg``: ``t1,w1,hc1`` (prologue tile pixels / warps / output-channel chunk: each chunk is 4 dots of ``hc1`` output channels),
        ``t2,w2,hc2`` (epilogue), ``ein`` = ``"xla"`` (the contraction).
    trimul_prologue_f32 / trimul_epilogue_f32: the two kernels alone.

Weights are handed to the kernels K-major (``[HD,C]`` for the projections, ``[C,HD]`` for the output projection: tiny transposes of the
parameter arrays at trace time) so every dot's second operand is contiguous along the contraction — the layout tf32 tensor-core MMAs read
without a shared-memory transpose. ``S % bq == 0``, ``S % bk == 0``, ``N % t == 0`` as the carried kernels. Run-to-run deterministic (no
atomics). jax and the carried kernels are imported on first use (``_ns()``), never at import.
"""
from types import SimpleNamespace
from typing import Any, Dict

PRECISIONS = ("tf32", "tf32x3", "ieee")               # operand precision of every MMA (module doc); the SERVED-line word is mma=<name>; REQUIRED on every call
PRECISION_WORDS_HELP = ("'tf32' = the stock model's class and the fast word (both blocks); 'tf32x3' = f32-class PER CELL, still faster than the stock-like XLA body for the ATTENTION but SLOWER than it for the TRIANGLE MULTIPLICATION (route that block to stock by name instead); per-cell classes do not propagate monotonically end to end (kit evidence decides the shipped word); 'ieee' = CUDA-core FMA, validation only. JAXF32 measured timing (0.5.5 bytes, H100, C=128, ms per call, tf32x3 | tf32 | XLA-default f32 body | XLA HIGHEST): triangle attention N=448 2.20 | 1.18 | 4.36 | 5.36, N=1024 16.05 | 8.51 | 45.8 | 54.4 (tf32x3 err 4-5e-7; tf32 2.3e-4 == XLA-default); triangle multiplication N=448 2.24 | 0.81 | 1.74 | 2.81, N=1024 12.67 | 3.29 | 8.22 | 15.6; ieee = 45-60x tf32 (validation only)")
DEFAULT_PRECISION = None                               # there is NO default word: a call that names none raises precision_required (module doc: each word is a speed/accuracy class)
DEFAULT_CFG = dict(t1=64, w1=4, hc1=128, t2=64, w2=4, hc2=64, bq=64, bk=64, wa=4, sa=2, vt=1)   # = the H100 row; the serve layer's tile table overrides this per GPU / N
TRIMUL_DEFAULT_CFG = dict(t1=64, w1=4, hc1=32, t2=64, w2=4, hc2=32, ein="xla")                     # idem for the triangle multiplication
TRIMUL_EQUATIONS = ("ikc,jkc->ijc", "kjc,kic->ijc")
_NS: Dict[str, Any] = {}


def _carried():
    from . import fpf_pallas_serve as S  # noqa: WPS433
    return S.kernels()


def _ns() -> SimpleNamespace:
    """Build (once) the f32 kernels over the carried module's helpers."""
    if "ns" in _NS:
        return _NS["ns"]
    import functools  # noqa: WPS433
    import math  # noqa: WPS433
    import jax  # noqa: WPS433
    import jax.numpy as jnp  # noqa: WPS433
    from jax.experimental import pallas as pl  # noqa: WPS433
    K, A = _carried()                                   # K: the carried trimul module, A: the carried attention module
    F32 = jnp.float32
    _CP, LOG2E, NEG, _ln = A._CP, A.LOG2E, A.NEG, A._ln
    presets = getattr(jax.lax, "DotAlgorithmPreset", None)
    PREC = {"tf32": None,                                                                   # Triton's default f32 dot: tf32 operands (round-to-nearest), f32 accumulate
            "tf32x3": getattr(presets, "TF32_TF32_F32_X3", None) if presets is not None else None,   # 3-pass tf32 = the dot-algorithm PRESET only
            "ieee": jax.lax.Precision.HIGHEST}                                              # f32 FMA dot

    def _prec(name):
        if name is None or name == "std":                                                   # no default word: every class is a deliberate speed/accuracy choice
            raise ValueError("fpf_pallas_f32: precision_required — name one of " + PRECISION_WORDS_HELP)
        if name not in PREC:
            raise ValueError(f"fpf_pallas_f32: precision {name!r} not in {PRECISIONS} — " + PRECISION_WORDS_HELP)
        if name == "tf32x3" and PREC[name] is None:                                         # never Precision.HIGH in its place: that lowers to plain tf32 here
            raise ValueError("fpf_pallas_f32: precision 'tf32x3' needs jax.lax.DotAlgorithmPreset.TF32_TF32_F32_X3, absent in this jax "
                             "(jax.lax.Precision.HIGH would lower to plain tf32 in Pallas-Triton, so it is not substituted) — use 'ieee' or 'tf32'")
        return PREC[name]

    def _dot_t(a, bt, prec):
        """``a [M,K] @ bt[N,K]^T -> [M,N]`` f32: the second operand is handed K-major (contiguous along the contraction)."""
        return jax.lax.dot_general(a, bt, (((1,), (1,)), ((), ())), precision=prec, preferred_element_type=F32)

    def _dot(a, b, prec):
        """``a [M,K] @ b [K,N] -> [M,N]`` f32."""
        return jnp.dot(a, b, precision=prec, preferred_element_type=F32)

    # ------------------------------------------------------------------------------------------------ prologue
    def _pa1f_kernel(x_ref, s_ref, o_ref, wq_ref, wk_ref, wv_ref, wb_ref, q_ref, k_ref, v_ref, b_ref, *, prec, vt, hc):
        xn = _ln(x_ref[...].astype(F32), s_ref[...], o_ref[...])                 # [t, C] f32 — NOT rounded
        HD = k_ref.shape[-1]
        for c0 in range(0, HD, hc):                                               # hc output columns per dot: the [hc, C] f32 weight slab a program holds at once
            q_ref[:, c0:c0 + hc] = _dot_t(xn, wq_ref[c0:c0 + hc, :], prec).astype(q_ref.dtype)
            k_ref[:, c0:c0 + hc] = _dot_t(xn, wk_ref[c0:c0 + hc, :], prec).astype(k_ref.dtype)
            v = _dot_t(xn, wv_ref[c0:c0 + hc, :], prec)                            # [t, hc]
            if vt:
                v_ref[c0:c0 + hc, :] = v.T.astype(v_ref.dtype)                     # [hc, t]: v stored transposed (K-major for the PV product)
            else:
                v_ref[:, c0:c0 + hc] = v.astype(v_ref.dtype)
        b_ref[...] = _dot_t(xn, wb_ref[...], prec).astype(b_ref.dtype)           # pair-bias logits, H padded to 16 columns

    def attn_prologue_f32(act, ln_scale, ln_offset, wq_kt, wk_kt, wv_kt, wb_kt, *, transpose, t=32, num_warps=4, num_stages=2, precision=None, vt=0, hc=32):
        """act (N,N,C) f32; weights K-major: wq_kt/wk_kt/wv_kt (HD,C), wb_kt (16,C) -> q,k (N,N,HD) [batch, seq, (h d)] (batch = row of act, or
        column if transpose), v (N,N,HD) or (N,HD,N) if vt, bias_raw (N,N,16) indexed by the UN-transposed pixel. transpose=True reads act
        transposed through the BlockSpec."""
        N, _, C = act.shape; HD = wq_kt.shape[0]; NB = wb_kt.shape[0]
        hc = min(int(hc), HD)
        assert N % t == 0 and HD % hc == 0, (N, t, HD, hc)
        if not transpose:      # program (i, jb): pixels (i, j-seg) -> q[b=i, s=j-seg]
            x_spec = pl.BlockSpec((None, t, C), lambda i, jb: (i, jb, 0)); b_spec = pl.BlockSpec((None, t, NB), lambda i, jb: (i, jb, 0))
        else:                  # program (i, jb): pixels (j-seg, i) [strided read] -> q[b=i, s=j-seg] ; bias_raw[a=j-seg, b=i]
            x_spec = pl.BlockSpec((t, None, C), lambda i, jb: (jb, i, 0)); b_spec = pl.BlockSpec((t, None, NB), lambda i, jb: (jb, i, 0))
        qk_spec = pl.BlockSpec((None, t, HD), lambda i, jb: (i, jb, 0))
        v_spec = pl.BlockSpec((None, HD, t), lambda i, jb: (i, 0, jb)) if vt else qk_spec
        vec = pl.BlockSpec((C,), lambda i, jb: (0,)); mat = pl.BlockSpec((HD, C), lambda i, jb: (0, 0)); matb = pl.BlockSpec((NB, C), lambda i, jb: (0, 0))
        sd = jax.ShapeDtypeStruct
        return pl.pallas_call(
            functools.partial(_pa1f_kernel, prec=precision, vt=vt, hc=hc), grid=(N, N // t), in_specs=[x_spec, vec, vec, mat, mat, mat, matb],
            out_specs=[qk_spec, qk_spec, v_spec, b_spec],
            out_shape=[sd((N, N, HD), F32), sd((N, N, HD), F32), sd((N, HD, N) if vt else (N, N, HD), F32), sd((N, N, NB), F32)],
            compiler_params=_CP(num_warps=num_warps, num_stages=num_stages), name="triattn_prologue_f32")(act, ln_scale, ln_offset, wq_kt, wk_kt, wv_kt, wb_kt)

    # ------------------------------------------------------------------------------------------------ flash core
    def _faf_kernel(q_ref, k_ref, v_ref, b_ref, m_ref, o_ref, *, bk, nk, qk_scale, prec, vt):
        bq, D = q_ref.shape
        q = q_ref[...].astype(F32)
        m_i = jnp.full((bq,), NEG, F32); l_i = jnp.zeros((bq,), F32); acc = jnp.zeros((bq, D), F32)

        def body(j, carry):
            acc, m_i, l_i = carry
            ks = pl.ds(j * bk, bk)
            k = k_ref[ks, :].astype(F32)                                            # [bk, D]
            s = _dot_t(q, k, prec) * qk_scale + b_ref[:, ks].astype(F32) * LOG2E   # [bq, bk], log2 domain
            s = jnp.where(m_ref[ks][None, :], s, NEG)
            m_new = jnp.maximum(m_i, jnp.max(s, axis=1))
            alpha = jnp.exp2(m_i - m_new)
            p = jnp.exp2(s - m_new[:, None])                                        # f32, handed to the PV product un-rounded
            l_new = alpha * l_i + p.sum(axis=1)
            if vt:
                pv = _dot_t(p, v_ref[:, ks].astype(F32), prec)                      # v tile [D, bk] (K-major)
            else:
                pv = _dot(p, v_ref[ks, :].astype(F32), prec)                        # v tile [bk, D]
            return acc * alpha[:, None] + pv, m_new, l_new

        acc, m_i, l_i = jax.lax.fori_loop(0, nk, body, (acc, m_i, l_i))
        o_ref[...] = (acc / l_i[:, None]).astype(o_ref.dtype)

    def flash_attention_bshd_f32(q, k, v, bias, mask, *, bq=64, bk=32, num_warps=4, num_stages=2, precision=None, vt=0):
        """q,k [B,S,H,D] f32; v [B,S,H,D] (or [B,H,D,S] if vt); bias [1,H,S,S] or [H,S,S]; mask bool [B,S] (key mask per batch row) or
        broadcastable from [B,1,1,S] -> o [B,S,H,D] f32. Fully-masked rows give the uniform average (finite negative, never NaN)."""
        B, S, H, D = q.shape
        assert k.shape == (B, S, H, D) and v.shape == ((B, H, D, S) if vt else (B, S, H, D)), (q.shape, k.shape, v.shape, vt)
        assert S % bq == 0 and S % bk == 0, (S, bq, bk)
        if bias.ndim == 4:
            assert bias.shape[0] == 1, bias.shape
            bias = bias[0]
        assert bias.shape == (H, S, S), bias.shape
        mask = jnp.broadcast_to(jnp.reshape(mask, (mask.shape[0], -1))[:, -S:], (B, S)) if mask.ndim > 2 else jnp.broadcast_to(mask, (B, S))
        qk_scale = LOG2E / math.sqrt(D)
        v_spec = pl.BlockSpec((None, None, D, S), lambda i, h, b: (b, h, 0, 0)) if vt else pl.BlockSpec((None, S, None, D), lambda i, h, b: (b, 0, h, 0))
        return pl.pallas_call(
            functools.partial(_faf_kernel, bk=bk, nk=S // bk, qk_scale=qk_scale, prec=precision, vt=vt), grid=(S // bq, H, B),
            in_specs=[pl.BlockSpec((None, bq, None, D), lambda i, h, b: (b, i, h, 0)),
                      pl.BlockSpec((None, S, None, D), lambda i, h, b: (b, 0, h, 0)),
                      v_spec,
                      pl.BlockSpec((None, bq, S), lambda i, h, b: (h, i, 0)),
                      pl.BlockSpec((None, S), lambda i, h, b: (b, 0))],
            out_specs=pl.BlockSpec((None, bq, None, D), lambda i, h, b: (b, i, h, 0)),
            out_shape=jax.ShapeDtypeStruct((B, S, H, D), F32),
            compiler_params=_CP(num_warps=num_warps, num_stages=num_stages), name="triattn_fwd_f32",
        )(q, k, v, bias, mask.astype(jnp.bool_))

    # ------------------------------------------------------------------------------------------------ epilogue
    def _pa2f_kernel(o_ref, x_ref, s_ref, off_ref, wg_ref, wo_ref, bg_ref, bo_ref, out_ref, *, prec, hc):
        xn = _ln(x_ref[...].astype(F32), s_ref[...], off_ref[...])                # recompute the LayerNorm for the gate (f32, not rounded)
        HD = o_ref.shape[-1]
        acc = jnp.zeros(out_ref.shape, F32) + bo_ref[...][None, :]                # [t, C]
        for c0 in range(0, HD, hc):                                               # gate + output projection in hc-column chunks of HD
            g = jax.nn.sigmoid(_dot_t(xn, wg_ref[c0:c0 + hc, :], prec) + bg_ref[c0:c0 + hc][None, :])   # [t, hc]
            wa = o_ref[:, c0:c0 + hc].astype(F32) * g
            acc = acc + _dot_t(wa, wo_ref[:, c0:c0 + hc], prec)                    # partial contraction over these hc head channels
        out_ref[...] = acc.astype(out_ref.dtype)

    def attn_epilogue_f32(o, act, ln_scale, ln_offset, wg_kt, wo_kt, bg, bo, *, transpose, t=32, num_warps=4, num_stages=2, precision=None, hc=32):
        """o (N,N,HD) attention output [b,s,(h d)] f32; weights K-major wg_kt (HD,C), wo_kt (C,HD); bg [HD], bo [C] f32 -> module output
        (N,N,C) f32 in the ORIGINAL (un-transposed) frame."""
        N, _, C = act.shape; HD = o.shape[-1]
        hc = min(int(hc), HD)
        assert N % t == 0 and HD % hc == 0, (N, t, HD, hc)
        o_spec = pl.BlockSpec((None, t, HD), lambda i, jb: (i, jb, 0))
        if not transpose:
            x_spec = pl.BlockSpec((None, t, C), lambda i, jb: (i, jb, 0)); out_spec = pl.BlockSpec((None, t, C), lambda i, jb: (i, jb, 0))
        else:                  # o[b=i, s=j] -> out pixel (j, i); gate from act[j, i]
            x_spec = pl.BlockSpec((t, None, C), lambda i, jb: (jb, i, 0)); out_spec = pl.BlockSpec((t, None, C), lambda i, jb: (jb, i, 0))
        vec = pl.BlockSpec((C,), lambda i, jb: (0,))
        return pl.pallas_call(
            functools.partial(_pa2f_kernel, prec=precision, hc=hc), grid=(N, N // t),
            in_specs=[o_spec, x_spec, vec, vec, pl.BlockSpec((HD, C), lambda i, jb: (0, 0)), pl.BlockSpec((C, HD), lambda i, jb: (0, 0)),
                      pl.BlockSpec((HD,), lambda i, jb: (0,)), vec],
            out_specs=out_spec, out_shape=jax.ShapeDtypeStruct((N, N, C), F32),
            compiler_params=_CP(num_warps=num_warps, num_stages=num_stages), name="triattn_epilogue_f32")(o, act, ln_scale, ln_offset, wg_kt, wo_kt, bg, bo)

    # ------------------------------------------------------------------------------------------------ block
    def grid_self_attention_fused_f32(act, pair_mask, kp, *, transpose, ending_bias_transposed=True, cfg=None, precision=None):
        """Module doc. ``kp`` weights of any float dtype are read as f32 (exact for bf16/f16/f32 parameters)."""
        N, _, C = act.shape; H, D = int(kp["H"]), int(kp["D"]); HD = H * D
        c = {**DEFAULT_CFG, **(cfg or {})}
        prec = _prec(precision); vt = int(bool(c.get("vt", 0)))
        f = lambda a: jnp.asarray(a).astype(F32)  # noqa: E731
        wq_kt, wk_kt, wv_kt, wb_kt, wg_kt = (f(kp[n]).T for n in ("wq_t", "wk_t", "wv2", "wb16", "wg_t"))    # (HD|16, C): K-major
        wo_kt = f(kp["wo"]).T                                                                             # (C, HD): K-major
        bg = f(kp["bg"]).reshape(HD) if kp.get("bg") is not None else jnp.zeros((HD,), F32)
        bo = f(kp["bo"]).reshape(C) if kp.get("bo") is not None else jnp.zeros((C,), F32)
        x = f(act)
        q, k, v, braw = attn_prologue_f32(x, f(kp["ln_scale"]), f(kp["ln_offset"]), wq_kt, wk_kt, wv_kt, wb_kt, transpose=transpose,
                                          t=c["t1"], num_warps=c["w1"], precision=prec, vt=vt, hc=c.get("hc1", HD))
        bias = jnp.transpose(braw[:, :, :H], (2, 0, 1))                       # (H, N, N) from the un-transposed normalised act, as stock
        if transpose and ending_bias_transposed:
            bias = jnp.swapaxes(bias, -1, -2)
        mask2 = jnp.swapaxes(pair_mask, -1, -2) > 0                            # key k of batch line b <- pair_mask[k, b]
        vv = v.reshape(N, H, D, N) if vt else v.reshape(N, N, H, D)
        o = flash_attention_bshd_f32(q.reshape(N, N, H, D), k.reshape(N, N, H, D), vv, bias[None], mask2,
                                     bq=c["bq"], bk=c["bk"], num_warps=c["wa"], num_stages=c["sa"], precision=prec, vt=vt).reshape(N, N, HD)
        return attn_epilogue_f32(o, x, f(kp["ln_scale"]), f(kp["ln_offset"]), wg_kt, wo_kt, bg, bo, transpose=transpose,
                                 t=c["t2"], num_warps=c["w2"], precision=prec, hc=c.get("hc2", HD))


    # ================================================================================================ triangle multiplication
    def _k1f_kernel(x_ref, m_ref, s_ref, o_ref, wpa_ref, wga_ref, wpb_ref, wgb_ref, bpa_ref, bga_ref, bpb_ref, bgb_ref, a_ref, b_ref, *, prec, hc):
        xn = K._ln(x_ref[...].astype(F32), s_ref[...], o_ref[...])                # [t, C] input LayerNorm, f32 (not rounded)
        m = m_ref[...].astype(F32)[:, None]                                        # [t, 1]
        C = a_ref.shape[0]
        for c0 in range(0, C, hc):                                                # hc output channels per chunk: 4 dots of [t,C] x [C,hc]
            a = (_dot_t(xn, wpa_ref[c0:c0 + hc, :], prec) + bpa_ref[c0:c0 + hc][None, :]) \
                * jax.nn.sigmoid(_dot_t(xn, wga_ref[c0:c0 + hc, :], prec) + bga_ref[c0:c0 + hc][None, :]) * m
            a_ref[c0:c0 + hc, :] = a.T.astype(a_ref.dtype)                        # [hc, t] channel-major store
            b = (_dot_t(xn, wpb_ref[c0:c0 + hc, :], prec) + bpb_ref[c0:c0 + hc][None, :]) \
                * jax.nn.sigmoid(_dot_t(xn, wgb_ref[c0:c0 + hc, :], prec) + bgb_ref[c0:c0 + hc][None, :]) * m
            b_ref[c0:c0 + hc, :] = b.T.astype(b_ref.dtype)

    def trimul_prologue_f32(act, mask, ln_scale, ln_offset, wpa_kt, wga_kt, wpb_kt, wgb_kt, bpa, bga, bpb, bgb, *, transpose_out, t=64, num_warps=4,
                            num_stages=2, precision=None, hc=32):
        """act (N,N,C) f32, mask (N,N); weights K-major ``[C_out, C_in]``; biases [C] f32 -> planes A, B (C,N,N) f32: ``A[c,i,j] = a(pixel i,j)``
        (``transpose_out``: ``A[c,j,i]``, for the incoming equation)."""
        N, N2, C = act.shape
        hc = min(int(hc), C)
        assert N == N2 and N % t == 0 and C % hc == 0, (act.shape, t, hc)
        if not transpose_out:   # program (i, jb): pixels (i, jb*t : jb*t+t) -> A[:, i, jb*t:...]
            x_spec = pl.BlockSpec((None, t, C), lambda i, jb: (i, jb, 0)); m_spec = pl.BlockSpec((None, t), lambda i, jb: (i, jb))
            o_spec = pl.BlockSpec((C, None, t), lambda i, jb: (0, i, jb))
        else:                   # program (j, kb): pixels (kb*t : kb*t+t, j) -> A[:, j, kb*t:...]
            x_spec = pl.BlockSpec((t, None, C), lambda j, kb: (kb, j, 0)); m_spec = pl.BlockSpec((t, None), lambda j, kb: (kb, j))
            o_spec = pl.BlockSpec((C, None, t), lambda j, kb: (0, j, kb))
        vec = pl.BlockSpec((C,), lambda i, jb: (0,)); mat = pl.BlockSpec((C, C), lambda i, jb: (0, 0))
        return pl.pallas_call(
            functools.partial(_k1f_kernel, prec=precision, hc=hc), grid=(N, N // t),
            in_specs=[x_spec, m_spec, vec, vec, mat, mat, mat, mat, vec, vec, vec, vec], out_specs=[o_spec, o_spec],
            out_shape=[jax.ShapeDtypeStruct((C, N, N), F32), jax.ShapeDtypeStruct((C, N, N), F32)],
            compiler_params=_CP(num_warps=num_warps, num_stages=num_stages), name="trimul_prologue_f32",
        )(act, mask, ln_scale, ln_offset, wpa_kt, wga_kt, wpb_kt, wgb_kt, bpa, bga, bpb, bgb)

    def _k2f_kernel(t_ref, x_ref, si_ref, oi_ref, sc_ref, oc_ref, wo_ref, wg_ref, bo_ref, bg_ref, out_ref, *, prec, hc):
        y = K._ln(t_ref[...].astype(F32).T, sc_ref[...], oc_ref[...])           # [C, t] -> [t, C]; centre LayerNorm over channels, f32
        xn = K._ln(x_ref[...].astype(F32), si_ref[...], oi_ref[...])             # recompute the input LayerNorm for the gate
        C = out_ref.shape[-1]
        for c0 in range(0, C, hc):                                                # hc output channels per chunk: output projection + gating linear
            z = _dot_t(y, wo_ref[c0:c0 + hc, :], prec) + bo_ref[c0:c0 + hc][None, :]
            g = jax.nn.sigmoid(_dot_t(xn, wg_ref[c0:c0 + hc, :], prec) + bg_ref[c0:c0 + hc][None, :])
            out_ref[:, c0:c0 + hc] = (z * g).astype(out_ref.dtype)

    def trimul_epilogue_f32(tri, act, ln_in_scale, ln_in_offset, ln_c_scale, ln_c_offset, wo_kt, wg_kt, b_out, b_gl, *, t=64, num_warps=4, num_stages=2,
                            precision=None, hc=32):
        """tri (C,N,N) f32 contraction result ``[c,i,j]``; act (N,N,C) module input; weights K-major ``[C_out, C_in]`` -> module output (N,N,C) f32."""
        N, _, C = act.shape
        hc = min(int(hc), C)
        assert tri.shape == (C, N, N) and N % t == 0 and C % hc == 0, (tri.shape, act.shape, t, hc)
        vec = pl.BlockSpec((C,), lambda i, jb: (0,)); mat = pl.BlockSpec((C, C), lambda i, jb: (0, 0))
        return pl.pallas_call(
            functools.partial(_k2f_kernel, prec=precision, hc=hc), grid=(N, N // t),
            in_specs=[pl.BlockSpec((C, None, t), lambda i, jb: (0, i, jb)), pl.BlockSpec((None, t, C), lambda i, jb: (i, jb, 0)),
                      vec, vec, vec, vec, mat, mat, vec, vec],
            out_specs=pl.BlockSpec((None, t, C), lambda i, jb: (i, jb, 0)), out_shape=jax.ShapeDtypeStruct((N, N, C), F32),
            compiler_params=_CP(num_warps=num_warps, num_stages=num_stages), name="trimul_epilogue_f32",
        )(tri, act, ln_in_scale, ln_in_offset, ln_c_scale, ln_c_offset, wo_kt, wg_kt, b_out, b_gl)

    EIN_PREC = {"tf32": None, "tf32x3": jax.lax.Precision.HIGHEST, "ieee": jax.lax.Precision.HIGHEST}   # the XLA contraction (module doc)

    def triangle_multiplication_fused_f32(act, mask, p, *, equation, cfg=None, precision=None):
        """Module doc. Weights of any float dtype are read as f32."""
        assert equation in TRIMUL_EQUATIONS, equation
        c = {**TRIMUL_DEFAULT_CFG, **(cfg or {})}
        if c.get("ein", "xla") != "xla":
            raise ValueError(f"fpf_pallas_f32: ein={c.get('ein')!r}; the pinned f32 contraction is 'xla'")
        prec = _prec(precision)
        incoming = equation == "kjc,kic->ijc"
        N, _, C = act.shape
        f = lambda a: jnp.asarray(a).astype(F32)  # noqa: E731
        wpa, wga, wpb, wgb = (f(w).T for w in K.split_glu_weights(p["w_proj"], p["w_gate"]))       # [C_out, C_in]: K-major
        if p.get("b_proj") is not None:
            bpa, bga, bpb, bgb = (f(b) for b in (p["b_proj"][0::2], p["b_gate"][0::2], p["b_proj"][1::2], p["b_gate"][1::2]))
            b_out, b_gl = f(p["b_out"]), f(p["b_gl"])
        else:
            bpa = bga = bpb = bgb = b_out = b_gl = jnp.zeros((C,), F32)
        x = f(act)
        lns, lno = f(p["ln_in_scale"]), f(p["ln_in_offset"])
        A_, B_ = trimul_prologue_f32(x, f(mask), lns, lno, wpa, wga, wpb, wgb, bpa, bga, bpb, bgb, transpose_out=incoming,
                                     t=c["t1"], num_warps=c["w1"], precision=prec, hc=c.get("hc1", C))
        P, Q = (B_, A_) if incoming else (A_, B_)          # incoming: out[c,i,j] = sum_k b(k,i) a(k,j) = sum_k B'[c,i,k] A'[c,j,k]
        tri = jnp.einsum("cik,cjk->cij", P, Q, precision=EIN_PREC[precision], preferred_element_type=F32)
        return trimul_epilogue_f32(tri, x, lns, lno, f(p["ln_c_scale"]), f(p["ln_c_offset"]), f(p["w_out"]).T, f(p["w_gl"]).T, b_out, b_gl,
                                   t=c["t2"], num_warps=c["w2"], precision=prec, hc=c.get("hc2", C))

    ns = SimpleNamespace(attn_prologue_f32=attn_prologue_f32, flash_attention_bshd_f32=flash_attention_bshd_f32, attn_epilogue_f32=attn_epilogue_f32,
                         grid_self_attention_fused_f32=grid_self_attention_fused_f32, trimul_prologue_f32=trimul_prologue_f32,
                         trimul_epilogue_f32=trimul_epilogue_f32, triangle_multiplication_fused_f32=triangle_multiplication_fused_f32, precision_arg=_prec)
    _NS["ns"] = ns
    return ns


def grid_self_attention_fused_f32(act, pair_mask, kp, *, transpose, ending_bias_transposed=True, cfg=None, precision=None):
    return _ns().grid_self_attention_fused_f32(act, pair_mask, kp, transpose=transpose, ending_bias_transposed=ending_bias_transposed, cfg=cfg, precision=precision)


def flash_attention_bshd_f32(q, k, v, bias, mask, *, bq=64, bk=32, num_warps=4, num_stages=2, precision=None, vt=0):
    n = _ns()
    return n.flash_attention_bshd_f32(q, k, v, bias, mask, bq=bq, bk=bk, num_warps=num_warps, num_stages=num_stages, precision=n.precision_arg(precision), vt=vt)


def triangle_multiplication_fused_f32(act, mask, p, *, equation, cfg=None, precision=None):
    return _ns().triangle_multiplication_fused_f32(act, mask, p, equation=equation, cfg=cfg, precision=precision)
