"""ef2_pairbias_attn — the structure module's attention with pair bias, made cheaper inside diffusion sampling.

Where it runs.  ESMFold2-Experimental(-Fast) has NO attention in its folding trunk (pair-only blocks: triangle multiplication +
transition).  `AttentionPairBias` (modeling_esmfold2_common) lives in the structure module's token transformer
(`structure_head.diffusion_module.token_transformer.attn_blocks`: 12 blocks, 16 heads, head dim 48, pair dim 256) and runs under
`torch.no_grad()` once per diffusion sampling step: 1 step per design step in the annealing phases, `num_sampling_steps=50` (34 steps
after the sigma cap) at the confidence phase, 200 in every hero-critic fold — in fp32 (sampling runs outside the trunk's autocast).
Stock computes, in EVERY block at EVERY sampling step, the pair bias `pair_bias_proj(pair_norm(z))` (LayerNorm + Linear over the
L x L x 256 fp32 conditioned pair tensor; backend 'fused': the fork's `fused_pair_bias` Triton kernel over a bf16 copy of it) although
`z` is one fixed tensor for the whole `structure_head.sample()` call (the fork itself fills `inference_cache["z"]` once and reuses it,
modeling_esmfold2_common.py DiffusionConditioning.forward l.1385-1395 at the pin; the block's dispatch and eager arithmetic are
AttentionPairBias.forward l.1107-1200, the fused route's kernel call l.1141-1163), then the
attention core as  einsum -> *scale -> +bias -> +mask -> softmax(dim=-2) -> .to -> einsum -> gate*  (backend 'fused':
`F.scaled_dot_product_attention` with the bias as a dense additive mask).

  variant="hoist"  numerics: EXACT (bitwise).  The per-block pair bias is computed once per block per `sample()` call by the very
                   modules / kernel stock calls, on the same tensor in the same context, and reused for the remaining sampling steps
                   (12 x (steps-1) LayerNorm+Linear passes over L²x256 fp32 removed per fold).  Everything else is stock's code verbatim.
  variant="fused"  numerics: reordered accumulation (fast class).  hoist + ONE Triton kernel for the attention core per block call:
                   flash-style (online softmax over key tiles, fp32 statistics and accumulation, split-mantissa 3xTF32 tensor-core dot
                   products for fp32 operands — fp32-accurate, no bf16/tf32 rounding of the contraction), pair bias + key mask + sigmoid
                   gate fused, no L x L logits / probabilities written to memory, 64-bit addressing.  Error vs an fp64 reference is at
                   or below stock's own fp32 arithmetic (unit test, kit standard <= 1.25x).

Scope and fallbacks (named, counted in `stats`, never silent): the bias cache is live only inside a `structure_head.sample()` call and
is dropped when it returns (no pair-sized tensor outlives the fold); a block call under grad, with a 3-D (scalar) z, with the
num_diffusion_samples>1 z expansion, or one the fork dispatches to the cuEquivariance pair-bias op (backend 'cuequivariance' above 750
queries) runs the stock code ('fallback_grad' / 'fallback_shape' / 'stock_cueq').  Instance-level, reversible: enable(model, variant) /
disable(model); nothing global is modified.

Usage:
    import ef2_pairbias_attn as pba
    pba.enable(model, variant="hoist")      # exact
    pba.enable(model, variant="fused")      # fast class
    pba.describe(model) -> "pairbias_attn variant=fused blocks=12 served=... computed=... fused_calls=... fallback=..."
    pba.disable(model)
"""
from __future__ import annotations

import contextlib
import functools
import types

import torch
import torch.nn.functional as F
from transformers.models.esmfold2 import modeling_esmfold2_common as C

try:
    import triton
    import triton.language as tl
    TRITON_OK, _TRITON_ERR = True, None
except Exception as e:  # noqa: BLE001
    TRITON_OK, _TRITON_ERR = False, e

VARIANTS = ("hoist", "fused")
# fp32 operands: split-mantissa 3xTF32 tensor-core dot products ('tf32x3': fp32-accurate — error vs an fp64 gold at or below stock's
# fp32 einsum at 200/450/700 tokens — and faster than SDPA here; plain 'tf32' is 1e-3-relative and 'ieee' FMA dots are 5-20x slower:
# tools/ef2inv_bench/pairbias/microbench_attn.py).  32-query x 64-key tiles: the core is latency/occupancy-bound at these sizes.
DOT_PRECISION = "tf32x3"
BLOCK_M, BLOCK_N, NUM_WARPS, NUM_STAGES = 32, 64, 4, 2


# ----------------------------------------------------------------------------------------------------------------------------
# Triton kernel: O[b,i,h,:] = sigmoid(G[b,i,h,:]) * sum_j softmax_j( (Q[b,i,h]·K[b,j,h]) * scale + BIAS[b,h,i,j] + maskmin(j) ) V[b,j,h,:]
# ----------------------------------------------------------------------------------------------------------------------------
if TRITON_OK:
    @triton.jit
    def _pba_fwd_kernel(Q, K, V, BIAS, G, MASK, O,
                        sqb, sqn, sqh, sqd, skb, skn, skh, skd, svb, svn, svh, svd,
                        sbb, sbh, sbq, sbk, sgb, sgn, sgh, sgd, smb, smk, sob, son, soh, sod,
                        NQ, NK, scale, mask_min,
                        H: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                        HAS_MASK: tl.constexpr, HAS_GATE: tl.constexpr, IP: tl.constexpr, LOWP: tl.constexpr):
        LOG2E: tl.constexpr = 1.4426950408889634
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = (pid_bh // H).to(tl.int64)
        h = (pid_bh % H).to(tl.int64)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        m_valid = offs_m < NQ
        d_valid = offs_d < D
        offs_m64 = offs_m.to(tl.int64)
        offs_d64 = offs_d.to(tl.int64)
        q = tl.load(Q + b * sqb + h * sqh + offs_m64[:, None] * sqn + offs_d64[None, :] * sqd,
                    mask=m_valid[:, None] & d_valid[None, :], other=0.0)
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
        for start_n in range(0, NK, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_valid = offs_n < NK
            offs_n64 = offs_n.to(tl.int64)
            kt = tl.load(K + b * skb + h * skh + offs_n64[None, :] * skn + offs_d64[:, None] * skd,
                         mask=n_valid[None, :] & d_valid[:, None], other=0.0)
            if LOWP:
                s = tl.dot(q, kt)
            else:
                s = tl.dot(q, kt, input_precision=IP)
            bias = tl.load(BIAS + b * sbb + h * sbh + offs_m64[:, None] * sbq + offs_n64[None, :] * sbk,
                           mask=m_valid[:, None] & n_valid[None, :], other=0.0)
            s = s * scale + bias.to(tl.float32)
            if HAS_MASK:
                km = tl.load(MASK + b * smb + offs_n64 * smk, mask=n_valid, other=1)
                s = tl.where(km[None, :] != 0, s, s + mask_min)
            # log2 domain with a finite floor: a masked key (stock adds finfo.min; the fused-backend bias folds its mask in) stays
            # finite, so a fully-masked row gives stock's uniform average instead of NaN; keys beyond NK are excluded (-inf)
            s = tl.where(n_valid[None, :], tl.maximum(s * LOG2E, -3.0e38), float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            v = tl.load(V + b * svb + h * svh + offs_n64[:, None] * svn + offs_d64[None, :] * svd,
                        mask=n_valid[:, None] & d_valid[None, :], other=0.0)
            if LOWP:
                acc = tl.dot(p.to(v.dtype), v, acc)
            else:
                acc = tl.dot(p, v, acc, input_precision=IP)
            m_i = m_new
        acc = acc / l_i[:, None]
        if HAS_GATE:
            g = tl.load(G + b * sgb + h * sgh + offs_m64[:, None] * sgn + offs_d64[None, :] * sgd,
                        mask=m_valid[:, None] & d_valid[None, :], other=0.0).to(tl.float32)
            acc = acc * (1.0 / (1.0 + tl.exp2(-g * LOG2E)))
        tl.store(O + b * sob + h * soh + offs_m64[:, None] * son + offs_d64[None, :] * sod,
                 acc.to(O.dtype.element_ty), mask=m_valid[:, None] & d_valid[None, :])


def fused_attention_core(q, k, v, bias_bhqk, gate_raw=None, key_mask=None, scale=None, mask_min=None):
    """q, k, v, gate_raw: (B, NQ|NK, H, D) strided views (any strides; fp32 or bf16/fp16, one dtype); bias_bhqk: (B, H, NQ, NK)
    (fp32 or bf16; key mask may already be folded in); key_mask: (B, NK) bool/int or None (added as `mask_min` where 0, stock's
    finfo.min convention); returns ctx (B, NQ, H, D) contiguous in q.dtype = sigmoid(gate_raw) * softmax_j(q·kᵀ·scale + bias + mask)·v."""
    if not TRITON_OK:
        raise RuntimeError(f"triton unavailable: {_TRITON_ERR!r}")
    B, NQ, H, D = q.shape
    NK = k.shape[1]
    assert k.shape == (B, NK, H, D) and v.shape == (B, NK, H, D), (q.shape, k.shape, v.shape)
    assert bias_bhqk.shape == (B, H, NQ, NK), (bias_bhqk.shape, (B, H, NQ, NK))
    assert q.dtype == k.dtype == v.dtype and q.dtype in (torch.float32, torch.bfloat16, torch.float16)
    lowp = q.dtype != torch.float32
    scale = float(D ** -0.5) if scale is None else float(scale)
    mask_min = float(torch.finfo(torch.float32 if not lowp else q.dtype).min) if mask_min is None else float(mask_min)
    out = torch.empty((B, NQ, H, D), device=q.device, dtype=q.dtype)
    BLOCK_D = max(16, triton.next_power_of_2(D))
    has_mask = key_mask is not None
    has_gate = gate_raw is not None
    km = key_mask if has_mask else q
    if has_mask and km.dtype == torch.bool:
        km = km.to(torch.uint8)
    g = gate_raw if has_gate else q
    grid = (triton.cdiv(NQ, BLOCK_M), B * H)
    _pba_fwd_kernel[grid](
        q, k, v, bias_bhqk, g, km, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        bias_bhqk.stride(0), bias_bhqk.stride(1), bias_bhqk.stride(2), bias_bhqk.stride(3),
        g.stride(0), g.stride(1), g.stride(2), g.stride(3),
        km.stride(0) if has_mask else 0, km.stride(1) if has_mask else 0,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        NQ, NK, scale, mask_min,
        H=H, D=D, BLOCK_D=BLOCK_D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        HAS_MASK=has_mask, HAS_GATE=has_gate, IP=DOT_PRECISION, LOWP=lowp,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )
    return out


# ----------------------------------------------------------------------------------------------------------------------------
# The patched AttentionPairBias.forward (stock control flow; the two lever points marked LEVER)
# ----------------------------------------------------------------------------------------------------------------------------
class _State:
    def __init__(self, variant: str):
        self.variant = variant
        self.blocks: list = []                 # every patched AttentionPairBias
        self.in_sample = 0                     # depth of open bias scopes (structure_head.sample() opens one; the cache is live only inside)
        self.stats = dict(served=0, computed=0, unscoped=0, fused_calls=0, fallback_grad=0, fallback_shape=0, stock_cueq=0)


@contextlib.contextmanager
def bias_scope(module_or_state):
    """The pair-bias cache is live inside this scope and dropped when the outermost scope exits.  `structure_head.sample()` runs
    inside one once enabled; open it explicitly only around a token-transformer loop driven by hand (tests, benches)."""
    st: _State = module_or_state if isinstance(module_or_state, _State) else module_or_state._pba_state
    st.in_sample += 1
    try:
        yield st
    finally:
        st.in_sample -= 1
        if st.in_sample <= 0:
            for blk in st.blocks:
                blk._pba_cache = None


def _version(z) -> int:
    """The tensor's in-place version counter, -1 for inference tensors (sample() runs under torch.inference_mode(), whose tensors do
    not track versions; the fork never updates z in place)."""
    return -1 if z.is_inference() else z._version


def _cached_bias(self, st: _State, z, key: str, make):
    """The block's pair bias for tensor `z` under cache slot `key`: computed by `make()` on a miss; cached (with a reference to z, so
    its identity cannot be recycled) only inside a sample() scope; a hit requires the same tensor object at the same version."""
    if st.in_sample <= 0:
        st.stats["unscoped"] += 1
        return make()
    ent = getattr(self, "_pba_cache", None)
    if ent is not None and ent[0] == key and ent[1] is z and ent[2] == _version(z):
        st.stats["served"] += 1
        return ent[3]
    bias = make()
    self._pba_cache = (key, z, _version(z), bias)
    st.stats["computed"] += 1
    return bias


def _apb_forward(self, a, s, z, beta=0.0, attention_mask=None, num_diffusion_samples: int = 1):
    st: _State = self._pba_state
    bsz, n_queries, d_model = a.shape
    if torch.is_grad_enabled():
        st.stats["fallback_grad"] += 1
        return self._pba_orig_forward(a, s, z, beta, attention_mask, num_diffusion_samples)
    if z.dim() != 4 or (z.shape[0] != bsz and num_diffusion_samples > 1) or not hasattr(self, "pair_bias_proj"):
        st.stats["fallback_shape"] += 1
        return self._pba_orig_forward(a, s, z, beta, attention_mask, num_diffusion_samples)
    use_fused_bias = self._can_use_fused_pair_bias(z, n_queries, beta)
    if not use_fused_bias and self._can_use_cueq_pair_bias(z, n_queries, beta):
        st.stats["stock_cueq"] += 1                                   # the cuEquivariance op (> 750 queries): stock, as dispatched
        return self._pba_orig_forward(a, s, z, beta, attention_mask, num_diffusion_samples)

    # ---- stock prologue (verbatim) ----
    if s is not None:
        x = self.adaln(a, s)
    else:
        x = self.pre_norm(a)
    n_keys = x.shape[1]
    q = self.q_proj(x).view(bsz, n_queries, self.num_heads, self.head_dim)
    kv = self.kv_proj(x)
    k, v = kv.chunk(2, dim=-1)
    k = k.view(bsz, n_keys, self.num_heads, self.head_dim)
    v = v.view(bsz, n_keys, self.num_heads, self.head_dim)
    if attention_mask is not None and attention_mask.shape[0] != bsz and num_diffusion_samples > 1:
        attention_mask = attention_mask.repeat_interleave(num_diffusion_samples, dim=0)

    if use_fused_bias:
        # ---- backend 'fused' (no grad): stock = fused_pair_bias kernel per call + SDPA with the bias as additive mask ----
        kernel_mask = attention_mask if attention_mask is not None else torch.ones(bsz, n_queries, device=a.device, dtype=torch.bool)

        def make_fused():
            pair_norm_w = self.pair_norm.weight
            pair_norm_b = self.pair_norm.bias if self.pair_norm.bias is not None else torch.zeros_like(pair_norm_w)
            z_bf = z if z.dtype == torch.bfloat16 else z.to(torch.bfloat16)
            return C._fused_pair_bias(z_bf, kernel_mask, self.pair_bias_proj.weight, num_heads=self.num_heads,
                                      pair_norm_w=pair_norm_w, pair_norm_b=pair_norm_b)          # (B, H, Q, K), key mask folded in
        bias = _cached_bias(self, st, z, "fused", make_fused)                                     # LEVER (hoist)
        if st.variant == "fused":
            st.stats["fused_calls"] += 1
            ctx = fused_attention_core(q, k, v, bias, gate_raw=self.g_proj(x).view(bsz, n_queries, self.num_heads, self.head_dim),
                                       key_mask=None, scale=self.scale)                           # LEVER (fused core; mask is in the bias)
            out = self.out_proj(ctx.view(bsz, n_queries, d_model))
        else:
            attn_out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=bias.to(q.dtype))
            g = torch.sigmoid(self.g_proj(x)).view(bsz, n_queries, self.num_heads, self.head_dim)
            ctx = g * attn_out.transpose(1, 2)
            out = self.out_proj(ctx.reshape(bsz, n_queries, d_model))
        if s is not None:
            out = torch.sigmoid(self.out_gate(s)) * out
        return out

    # ---- reference path (backend None / 'cuequivariance' <= 750 queries): stock = eager einsum attention ----
    if st.variant == "fused":
        def make_bhqk():
            return self.pair_bias_proj(self.pair_norm(z)).permute(0, 3, 1, 2).contiguous()       # (B, H, Q, K) fp32, once per sample()
        bias = _cached_bias(self, st, z, "bhqk", make_bhqk)                                       # LEVER (hoist)
        st.stats["fused_calls"] += 1
        mask_min = float(torch.finfo(q.dtype).min)
        ctx = fused_attention_core(q, k, v, bias, gate_raw=self.g_proj(x).view(bsz, n_queries, self.num_heads, self.head_dim),
                                   key_mask=(attention_mask.bool() if attention_mask is not None else None),
                                   scale=self.scale, mask_min=mask_min)                            # LEVER (fused core)
        out = self.out_proj(ctx.view(bsz, n_queries, d_model))
    else:
        g = torch.sigmoid(self.g_proj(x)).view(bsz, n_queries, self.num_heads, self.head_dim)
        logits = torch.einsum("... i h d, ... j h d -> ... i j h", q, k) * self.scale
        pair_bias = _cached_bias(self, st, z, "eager", lambda: self.pair_bias_proj(self.pair_norm(z)))   # LEVER (hoist): stock's expression
        logits = logits + pair_bias.to(dtype=logits.dtype)
        if attention_mask is not None:
            min_val = torch.finfo(logits.dtype).min
            mask_bias = torch.where(attention_mask.bool()[:, None, :, None], 0.0, min_val)
            logits = logits + mask_bias.to(dtype=logits.dtype)
        attn = torch.softmax(logits, dim=-2).to(dtype=v.dtype)
        ctx = torch.einsum("... i j h, ... j h d -> ... i h d", attn, v)
        ctx = g * ctx
        out = self.out_proj(ctx.reshape(bsz, n_queries, d_model))
    if s is not None:
        out = torch.sigmoid(self.out_gate(s)) * out
    return out


def _scoped_sample_for(st, w, *args, **kwargs):
    """structure_head.sample with the pair-bias cache live for its duration (cleared on exit: no pair-sized tensor outlives the fold)."""
    with bias_scope(st):
        return w._ef2_prev(*args, **kwargs)


# ---- chain-aware instance-method wrapping (shared convention with ef2_sampler_graph: several levers may wrap sample()) ----
TAG = "ef2_pairbias_attn"


class _Wrapped:
    """`obj.name` replaced by this callable: calls fn(self, *a, **kw); fn delegates to self._ef2_prev (the callable it wrapped —
    the bound method or another lever's wrapper) and reads the module as self.obj."""

    def __init__(self, fn, prev, tag, obj):
        self.fn, self._ef2_prev, self._ef2_tag, self.obj = fn, prev, tag, obj

    def __call__(self, *a, **kw):
        return self.fn(self, *a, **kw)


def _wrap_method(obj, name: str, fn, tag: str) -> None:
    setattr(obj, name, _Wrapped(fn, getattr(obj, name), tag, obj))


def _unwrap_method(obj, name: str, tag: str) -> bool:
    cur, parent = getattr(obj, name, None), None
    while cur is not None and getattr(cur, "_ef2_tag", None) != tag:
        parent, cur = cur, getattr(cur, "_ef2_prev", None)
    if cur is None:
        return False
    if parent is not None:
        parent._ef2_prev = cur._ef2_prev
    elif getattr(cur._ef2_prev, "_ef2_tag", None) is None and name in vars(obj):
        delattr(obj, name)
    else:
        setattr(obj, name, cur._ef2_prev)
    return True


# ----------------------------------------------------------------------------------------------------------------------------
# enable / disable / describe
# ----------------------------------------------------------------------------------------------------------------------------
def _structure_heads(model):
    heads = [m for m in model.modules() if isinstance(m, C.DiffusionStructureHead)]
    if isinstance(model, C.DiffusionStructureHead) and model not in heads:
        heads.insert(0, model)
    return heads


def enable(model: torch.nn.Module, variant: str = "hoist") -> _State:
    """Patch every AttentionPairBias in `model` and open the bias scope around every DiffusionStructureHead.sample() in it
    (instance-level, reversible via disable()).  variant: 'hoist' (exact) | 'fused' (fast class: Triton attention core).
    A module without a structure head (a bare DiffusionTransformer) is patched too; its caller opens `bias_scope(module)`."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if variant == "fused" and not TRITON_OK:
        raise RuntimeError(f"variant 'fused' needs triton: {_TRITON_ERR!r}")
    if hasattr(model, "_pba_state"):
        if model._pba_state.variant != variant:
            raise RuntimeError(f"ef2_pairbias_attn already enabled with variant {model._pba_state.variant!r}; disable(model) first")
        return model._pba_state
    st = _State(variant)
    st.blocks = [m for m in model.modules() if isinstance(m, C.AttentionPairBias)]
    if not st.blocks:
        raise RuntimeError("no AttentionPairBias module in this model: nothing to patch")
    for blk in st.blocks:
        blk._pba_state, blk._pba_cache = st, None
        blk._pba_orig_forward = blk.forward
        blk.forward = types.MethodType(_apb_forward, blk)
    for sh in _structure_heads(model):
        scoped = functools.partial(_scoped_sample_for, st)
        _wrap_method(sh, "sample", scoped, TAG)
    model._pba_state = st
    return st


def disable(model: torch.nn.Module) -> None:
    for sh in _structure_heads(model):
        _unwrap_method(sh, "sample", TAG)
    for blk in [m for m in model.modules() if isinstance(m, C.AttentionPairBias)]:
        if hasattr(blk, "_pba_orig_forward"):
            blk.forward = blk._pba_orig_forward
            del blk._pba_orig_forward
        for n in ("_pba_state", "_pba_cache"):
            if hasattr(blk, n):
                delattr(blk, n)
    if hasattr(model, "_pba_state"):
        del model._pba_state


def describe(model) -> str:
    st = getattr(model, "_pba_state", None)
    if st is None:
        return "pairbias_attn=off"
    n = len(st.blocks)
    s = st.stats
    fb = s["fallback_grad"] + s["fallback_shape"]
    return (f"pairbias_attn variant={st.variant} blocks={n} served={s['served']} computed={s['computed']} unscoped={s['unscoped']} "
            f"fused_calls={s['fused_calls']} stock_cueq={s['stock_cueq']} fallback={fb}")
