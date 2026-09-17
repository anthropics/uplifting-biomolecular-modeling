"""ef2_esmc_rope — exact fused rotary embedding for the ESMC trunk: one Triton kernel per tensor and direction in place of
the fork's differentiable torch branch, with that branch's arithmetic.

`RotaryEmbedding.forward(q, k)` (transformers.models.esmc.modeling_esmc) rotates q and k through `_apply_rotary_emb_torch`
whenever the flash-attn Triton rotary is not bound (`_flash_attn_rotary_available` False — the state in which the q.k path
carries a gradient; the kit's opt-in upstream RoPE fix pins it). That branch costs 16 small kernels per layer forward (repeat x2, neg,
cat, mul x2, add, cat — for q and for k) and about 20 backward: ~1300 graph nodes per 80-layer LM pass, ~2900 per
pseudo-perplexity forward+backward, a quarter of the trunk's kernel time at design sizes and the same share of every
cudaGraphLaunch. This module rebinds `forward` on each `RotaryEmbedding` INSTANCE of one loaded trunk to a
`torch.autograd.Function` whose forward and backward are one Triton kernel per tensor, computing exactly what the torch
branch computes, rounding point for rounding point (h = rotary_dim / 2, r = cat(-x[h:2h], x[:h]), rnd = round to x.dtype):
    forward   out[j <  h] = rnd( rnd(x[j]*cos[j])   - rnd(x[j+h]*sin[j])   )      out[j >= 2h] = x[j]
              out[j >= h] = rnd( rnd(x[j]*cos[j-h]) + rnd(x[j-h]*sin[j-h]) )
    backward  dx[j <  h]  = rnd( rnd(g[j]*cos[j])   + rnd(g[j+h]*sin[j])   )      dx[j >= 2h]  = g[j]
              dx[j >= h]  = rnd( rnd(g[j]*cos[j-h]) - rnd(g[j-h]*sin[j-h]) )
The product of two bf16 (or fp16) values is exact in fp32, so each inner rnd(.) is the correctly rounded result torch's
elementwise kernel produces, negation commutes with rounding, and a two-term sum is order-free — which is why the torch
branch's autograd (MulBackward, NegBackward, Cat/SliceBackward, gradient accumulation over the two uses of x plus an
all-zero third) lands on the same bits. fp32 inputs are served with floating-point contraction disabled so mul and add stay
two roundings. Outputs AND input-gradients are tensor-equal to the torch branch at the design loop's shapes

(k/test_ef2_esmc_rope.py).

Not claimed, not touched: a module whose call would take the flash-attn Triton branch (stock as shipped — a different
rounding and no q.k gradient), an interleaved (GPT-J) layout, xPos scaling, non-CUDA or float64 tensors — those calls go to
the module's own forward and are counted under `fallback` by reason, never silently.

Usage:
    import ef2_esmc_rope as eer
    h = eer.enable(model)      # model: an ESMFold2 model (._esmc), an ESMCForMaskedLM (.esmc) or an ESMCModel; idempotent
    h.stats                    # {'served': n_calls, 'fallback': {reason: n}}
    eer.disable(model)
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl

_ATTR = "_ef2_esmc_rope"
_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


@triton.jit
def _rope_kernel(X, COS, SIN, OUT, S, H,
                 sxb, sxs, sxh, sob, sos, soh, scs,
                 HALF: tl.constexpr, D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_HALF: tl.constexpr,
                 BLOCK_TAIL: tl.constexpr, BACKWARD: tl.constexpr):
    pid = tl.program_id(0)                    # one program per (batch, position): all heads, whole head dim
    b = pid // S
    s = pid % S
    hh = tl.arange(0, BLOCK_H)[:, None]
    jj = tl.arange(0, BLOCK_HALF)[None, :]
    m = (hh < H) & (jj < HALF)
    xrow = X + b.to(tl.int64) * sxb + s.to(tl.int64) * sxs + hh * sxh
    orow = OUT + b.to(tl.int64) * sob + s.to(tl.int64) * sos + hh * soh
    x1 = tl.load(xrow + jj, mask=m, other=0.0)
    x2 = tl.load(xrow + HALF + jj, mask=m, other=0.0)
    c = tl.load(COS + s * scs + jj, mask=jj < HALF, other=0.0).to(tl.float32)
    sn = tl.load(SIN + s * scs + jj, mask=jj < HALF, other=0.0).to(tl.float32)
    DT = x1.dtype
    x1f = x1.to(tl.float32)
    x2f = x2.to(tl.float32)
    p1c = (x1f * c).to(DT).to(tl.float32)     # rnd(x[:h] * cos)
    p2s = (x2f * sn).to(DT).to(tl.float32)    # rnd(x[h:] * sin)
    p2c = (x2f * c).to(DT).to(tl.float32)     # rnd(x[h:] * cos)
    p1s = (x1f * sn).to(DT).to(tl.float32)    # rnd(x[:h] * sin)
    if BACKWARD:
        o1 = (p1c + p2s).to(DT)
        o2 = (p2c - p1s).to(DT)
    else:
        o1 = (p1c - p2s).to(DT)
        o2 = (p2c + p1s).to(DT)
    tl.store(orow + jj, o1, mask=m)
    tl.store(orow + HALF + jj, o2, mask=m)
    if BLOCK_TAIL > 0:                        # rotary_dim < head_dim: the tail passes through (forward and backward)
        tt = tl.arange(0, BLOCK_TAIL)[None, :]
        mt = (hh < H) & (tt < D - 2 * HALF)
        xt = tl.load(xrow + 2 * HALF + tt, mask=mt, other=0.0)
        tl.store(orow + 2 * HALF + tt, xt, mask=mt)


def _launch(x, cos, sin, backward):
    if x.stride(-1) != 1:
        x = x.contiguous()
    B, S, H, D = x.shape
    half = cos.shape[-1]
    out = torch.empty((B, S, H, D), dtype=x.dtype, device=x.device)
    if out.numel() == 0:
        return out
    tail = D - 2 * half
    _rope_kernel[(B * S,)](x, cos, sin, out, S, H,
                           x.stride(0), x.stride(1), x.stride(2), out.stride(0), out.stride(1), out.stride(2), cos.stride(0),
                           HALF=half, D=D, BLOCK_H=triton.next_power_of_2(H), BLOCK_HALF=triton.next_power_of_2(half),
                           BLOCK_TAIL=triton.next_power_of_2(tail) if tail > 0 else 0, BACKWARD=backward,
                           num_warps=4, enable_fp_fusion=False)
    return out


class _ExactRope(torch.autograd.Function):
    """out = the torch branch's _apply_rotary_emb_torch(x, cos, sin, interleaved=False); cos/sin need no gradient (buffers).

    cos/sin are views of the module's cache, which the fold's feature pass builds under torch.inference_mode(); the grad-enabled
    pseudo-perplexity pass then reads the same cache (the shared trunk). An inference tensor cannot go through
    save_for_backward — the torch branch only gets away with it because its `.repeat()` makes a fresh tensor — so the views
    are held as plain context attributes (read by the backward kernel outside autograd; the cache is only ever re-assigned,
    never modified in place)."""

    @staticmethod
    def forward(ctx, x, cos, sin):
        ctx.cos_sin = (cos, sin)
        return _launch(x, cos, sin, backward=False)

    @staticmethod
    def backward(ctx, g):
        cos, sin = ctx.cos_sin
        ctx.cos_sin = None
        return _launch(g, cos, sin, backward=True), None, None


def _fallback_reason(mod, ME, q, k):
    if ME._flash_attn_rotary_available and q.device.type == "cuda":
        return "flash_triton_branch"                         # stock as shipped: a different rounding, no q.k gradient — not this lever's to change
    if mod.interleaved:
        return "interleaved"
    if mod.scale is not None:
        return "xpos_scale"
    for t in (q, k):
        if t.device.type != "cuda" or t.dtype not in _DTYPES or t.dim() != 4:
            return "tensor_form"
    if q.shape[1] != k.shape[1] or q.dtype != k.dtype:
        return "tensor_form"
    return None


class _Handle:
    def __init__(self, esmc):
        self.esmc = esmc
        self.modules = []
        self.stats = {"served": 0, "fallback": {}}
        self.seqlen_offset = 0                      # extra position offset for every call (a caller running one chain alone at its original positions sets it)

    def _make_forward(self, mod, ME):
        stats = self.stats
        stock_forward = type(mod).forward

        def forward(q, k, seqlen_offset: int = 0):
            seqlen_offset = seqlen_offset + self.seqlen_offset
            why = _fallback_reason(mod, ME, q, k)
            if why is not None:
                stats["fallback"][why] = stats["fallback"].get(why, 0) + 1
                return stock_forward(mod, q, k, seqlen_offset)
            mod._update_cos_sin_cache(q.shape[1] + seqlen_offset, device=q.device, dtype=q.dtype)   # as the stock forward does
            cos = mod._cos_cached[seqlen_offset:]
            sin = mod._sin_cached[seqlen_offset:]
            if cos.shape[0] < q.shape[1] or cos.stride(-1) != 1 or 2 * cos.shape[-1] > q.shape[-1]:
                stats["fallback"]["cache_form"] = stats["fallback"].get("cache_form", 0) + 1
                return stock_forward(mod, q, k, seqlen_offset)
            stats["served"] += 1
            return _ExactRope.apply(q, cos, sin), _ExactRope.apply(k, cos, sin)
        return forward

    def install(self):
        from transformers.models.esmc import modeling_esmc as ME
        for m in self.esmc.modules():
            if type(m) is ME.RotaryEmbedding and "forward" not in vars(m):     # the plain class only (not the varlen _TritonRotaryEmbedding)
                m.forward = self._make_forward(m, ME)
                self.modules.append(m)
        return self

    def remove(self):
        for m in self.modules:
            vars(m).pop("forward", None)
        self.modules = []


def _esmc_of(model):
    for name in ("_esmc", "esmc"):
        sub = getattr(model, name, None)
        if isinstance(sub, torch.nn.Module):
            return sub
    if isinstance(model, torch.nn.Module):
        return model
    raise TypeError("enable(model): expected an ESMFold2 model (._esmc), an ESMCForMaskedLM (.esmc) or an ESMC module")


def enable(model):
    esmc = _esmc_of(model)
    h = getattr(esmc, _ATTR, None)
    if h is None:
        h = _Handle(esmc).install()
        setattr(esmc, _ATTR, h)
    return h


def disable(model):
    esmc = _esmc_of(model)
    h = getattr(esmc, _ATTR, None)
    if h is not None:
        h.remove()
        delattr(esmc, _ATTR)


def handle(model):
    return getattr(_esmc_of(model), _ATTR, None)
