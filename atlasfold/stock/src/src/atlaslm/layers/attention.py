import torch
import torch.nn.functional as F
from torch import nn

from .rotary import RotaryEmbedding


class MultiHeadAttention(nn.Module):
    """A multi-head attention module with rotary positional embeddings."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = False,
        qk_layernorm: bool = True,
    ):
        super().__init__()

        self.d_model: int = d_model
        self.n_heads: int = n_heads
        self.d_head: int = self.d_model // self.n_heads

        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 3, bias=bias),
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        if qk_layernorm:
            self.q_ln = nn.LayerNorm(d_model, bias=bias)
            self.k_ln = nn.LayerNorm(d_model, bias=bias)
        else:
            self.q_ln = nn.Identity()
            self.k_ln = nn.Identity()

        self.rotary = RotaryEmbedding(d_model // n_heads, max_seqlen=20000)

    def forward(
        self,
        x: torch.Tensor,
        seq_id: torch.Tensor | None = None,
        pos_id: torch.Tensor | None = None,
        return_attn: bool = False,
        return_attn_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass for Multi-Head Attention.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch, seq_len, d_model).
        seq_id: torch.Tensor | None
            Optional tensor of shape (batch, seq_len) identifying sequence boundaries
            for masking within a packed batch.
        pos_id: torch.Tensor | None
            Optional tensor of shape (batch, seq_len) providing specific position
            indices for rotary embeddings.
        return_attn: bool
            If True, manually computes attention and returns weights.
            If False, uses efficient `scaled_dot_product_attention`.
        return_attn_logits: bool
            If True, returns raw attention logits instead of probabilities.


        Return
        ------
        out: torch.Tensor
            Output tensor of shape (batch, seq_len, d_model).
        attn_weights:
            Attention weights or logits of shape (batch, n_heads, seq_len, seq_len)
            if return_attn is True, otherwise None.
        """
        B, L, D = x.shape
        H, Dh = self.n_heads, self.d_head

        # [B, L, D] -> 3 * [B, L, D]
        q, k, v = self.layernorm_qkv(x).chunk(3, dim=-1)
        q, k = self.q_ln(q).to(q.dtype), self.k_ln(k).to(k.dtype)

        # [B, L, D] -> [B, L, H, Dh]
        q, k, v = map(lambda t: t.view(B, L, H, Dh), (q, k, v))
        q, k = self.rotary(q, k, pos_id)

        # [B, L, H, Dh] -> [B, H, L, Dh]
        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))

        if seq_id is not None:
            # NOTE: here we use == instead of & to avoid zero-division
            # during softmax for padding tokens.
            mask = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
            mask = mask.unsqueeze(1)  # [B, 1, L, L]
        else:
            mask = None

        if return_attn or return_attn_logits:
            q *= Dh**-0.5
            a = torch.matmul(q, k.transpose(-2, -1))  # [B, H, L, L]
            if mask is not None:
                a.masked_fill_(~mask, float("-inf"))
            attn_weights = F.softmax(a, dim=-1)
            out = torch.matmul(attn_weights, v)  # [B, H, L, Dh]
            if return_attn_logits:
                attn_weights = a
        else:
            out = F.scaled_dot_product_attention(q, k, v, mask)
            attn_weights = None

        # [B, H, L, Dh] -> [B, L, H, Dh] -> [B, L, D]
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)
        return out, attn_weights
