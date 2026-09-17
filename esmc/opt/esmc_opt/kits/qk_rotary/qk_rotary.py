"""The seam entry: `qk_rotary.ops(model) -> {'qk_ln_rotary': fn}`
for esmc_opt.kits.fused._patch._OPS. fn(rot, ln_q, ln_k, qkv_packed, cu_seqlens, max_seqlen) -> None: LN(q), LN(k)
(aten NT-128 order) RNE-bf16 into slots 0/1 of qkv.view(T, 3, H, Dh) AND the rotary applied in place — ONE Triton launch (the JIT form,
qk_ln_rope.qk_ln_rope_packed; Triton compiles each class once per machine into its own cache — the fused kit warms the classes at
apply). No knobs. The gamma pair is cached ONCE per (q_ln, k_ln) id as a [2, N] copy."""
import torch

from . import qk_ln_rope as K


def _gamma2(ln_q, ln_k):
    return torch.stack([ln_q.weight.detach(), ln_k.weight.detach()], 0).contiguous()


def ops(model):
    cache = {}

    def qk_ln_rotary(rot, ln_q, ln_k, qkv_packed, cu_seqlens, max_seqlen):
        key = (id(ln_q), id(ln_k))
        g = cache.get(key)
        if g is None:
            g = cache[key] = _gamma2(ln_q, ln_k)
        rot._update_cos_sin_cache(max_seqlen, device=qkv_packed.device, dtype=qkv_packed.dtype)
        K.qk_ln_rope_packed(qkv_packed, g, rot._cos_cached, rot._sin_cached, cu_seqlens, eps=float(ln_q.eps))
        return None

    return {"qk_ln_rotary": qk_ln_rotary}
