"""Lever lm_sdpa — AtlasLM MultiHeadAttention.forward (atlaslm/layers/attention.py L75-113). Stock, when AtlasFold asks for the attention
LOGITS of every layer (model.py L460: return_attn_logits=True): q *= Dh**-0.5; a = q @ k^T; a.masked_fill_(~mask, -inf); softmax(a); out =
p @ v; returns (out, a). The logits ``a`` are consumed downstream (36 -> 128 projection into the pair track), so they are kept EXACTLY as stock
forms them; only softmax(a) @ v is replaced by F.scaled_dot_product_attention(q, k, v, attn_mask=mask) on the same scaled q (scale=1.0), which
never materializes the fp32 probabilities. Class: fast (softmax re-association). Calls without logits requested take the stock path, counted."""
from . import Installed, size_gated, rebind

TARGET = "atlaslm.layers.attention"


MIN_TOKENS = int(__import__('os').environ.get('AFO_LM_SDPA_MIN_TOKENS', '1000'))


def install(mode: str, tag: str, ctx: dict) -> Installed:
    import importlib
    import torch
    import torch.nn.functional as Fn
    from opt_core.counters import Ledger
    try:
        am = importlib.import_module(TARGET)
    except Exception as e:  # noqa: BLE001
        return Installed("lm_sdpa", False, reason=f"import:{TARGET}:{type(e).__name__}")
    cls = am.MultiHeadAttention
    stock = cls.forward
    ledger = Ledger("LOCAL.atlasfold.lm_sdpa", impl=f"torch.sdpa@{torch.__version__.split('+')[0]}", origin="kit", min_tokens=MIN_TOKENS,
                    expected=("no_logits_requested", "below_min_tokens"))

    def forward(self, x, seq_id=None, pos_id=None, return_attn: bool = False, return_attn_logits: bool = False):
        if return_attn or not return_attn_logits:
            ledger.fallback("no_logits_requested")
            return stock(self, x, seq_id, pos_id, return_attn, return_attn_logits)
        B, L, D = x.shape
        if L < MIN_TOKENS:                                          # below ~1,000 tokens the extra GEMM launch costs more than the [H,L,L] softmax copy it saves
            ledger.fallback("below_min_tokens")
            return stock(self, x, seq_id, pos_id, return_attn, return_attn_logits)
        H, Dh = self.n_heads, self.d_head
        q, k, v = self.layernorm_qkv(x).chunk(3, dim=-1)
        q, k = self.q_ln(q).to(q.dtype), self.k_ln(k).to(k.dtype)
        q, k, v = map(lambda t: t.view(B, L, H, Dh), (q, k, v))
        q, k = self.rotary(q, k, pos_id)
        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))
        mask = (seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)).unsqueeze(1) if seq_id is not None else None
        q = q * Dh**-0.5
        a = torch.matmul(q, k.transpose(-2, -1))                  # the exported logits: same operands, same GEMM as stock
        if mask is not None:
            a.masked_fill_(~mask, float("-inf"))
        out = Fn.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)   # softmax(a) @ v without the [B,H,L,L] probabilities
        ledger.serve(f"L{L}xH{H}xD{Dh}")
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out), a
    forward.__qualname__ = "MultiHeadAttention.forward[atlasfold_opt:lm_sdpa]"
    rebind(cls, "forward", forward, stock)
    return Installed("lm_sdpa", True, lines=[lambda: ledger.line(tag)], gates=[size_gated(ledger)], facts={"impl": getattr(ledger, "impl", None)})
