"""Attention Pair Bias layer.
See Section 3.7 Algorithm 24 of the AlphaFold3 paper.
"""

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from atlasfold.model.network.primitives import AdaLN, LayerNorm, Linear, LinearNoBias


class Attention(nn.Module):
    def __init__(
        self,
        channel: int,
        num_heads: int,
        use_high_precision: bool = False,
        inf: float = 1e9,
    ) -> None:
        """Initialize the attention pair bias layer.

        Parameters
        ----------
        channel : int
            The atom/token dimension.
        num_heads : int
            The number of heads.
        """
        super().__init__()
        self.inf: float = inf
        assert channel % num_heads == 0
        self.channel: int = channel
        self.num_heads: int = num_heads
        self.head_dim: int = channel // num_heads
        self.use_high_precision: bool = use_high_precision

        self.linear_q = Linear(channel, channel)
        self.linear_k = LinearNoBias(channel, channel)
        self.linear_v = LinearNoBias(channel, channel)

    def forward(
        self,
        a_q: torch.Tensor,
        a_k: torch.Tensor,
        mask: torch.Tensor,
        pair_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        a_q : torch.Tensor
            The query input tensor (*, Lq, c_a).
        a_k : torch.Tensor
            The key/value input tensor (*, Lk, c_a).
        mask : torch.Tensor
            The attention mask tensor (*, 1, Lk) or (*, Lq, Lk)
        pair_bias : torch.Tensor | None
            The attention bias tensor (*, H, Lq, Lk)

        Returns
        -------
        out : torch.Tensor
            The output tensor (*, Lq, c_a)
        """
        # Compute the query, key, and value tensors from the input tensors.
        q = self.linear_q(a_q)  # [*, Lq, c]
        k, v = self.linear_k(a_k), self.linear_v(a_k)  # [*, Lk, c]

        # Reshape the query, key, and value tensors for multi-head attention.
        q, k, v = map(
            lambda t: einops.rearrange(t, "... l (h d) -> ... h l d", h=self.num_heads),
            (q, k, v),
        )  # [*, H, Lq/k, c_h]

        # Compute attention weights and output
        if self.use_high_precision:
            # Compute attention weights and output in high precision (float32)
            with torch.autocast(device_type=q.device.type, enabled=False):
                q, k = q.float(), k.float()
                q *= self.head_dim**-0.5
                # Compute attention weights
                attn = q @ k.mT  # [*, H, Lq, Lk]
                attn += ((~mask).float() * -self.inf).unsqueeze(-3)
                if pair_bias is not None:
                    attn += pair_bias
                # Compute attention output
                attn = attn.softmax(dim=-1).to(v.dtype)
            out = attn @ v  # [*, H, Lq, c_h]
        else:
            # Use Scaled Dot-Product Attention (SDPA)
            # Compute bias
            attn_bias = (~mask).to(q.dtype) * (-self.inf)
            attn_bias = attn_bias.unsqueeze(-3)  # [*, 1, Lq, Lk]
            if pair_bias is not None:
                attn_bias = attn_bias + pair_bias.to(q.dtype)  # [*, H, Lq, Lk]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        out = einops.rearrange(out, "... h lq d -> ... lq (h d)")  # [*, Lq, c]
        return out


class SelfAttention(nn.Module):
    def __init__(
        self,
        channel_a: int,
        num_heads: int,
        channel_cond: int | None,
        *,
        use_pair_bias: bool = True,
        use_high_precision: bool = False,
    ) -> None:
        """Initialize the attention pair bias layer.

        Parameters
        ----------
        channel_a : int
            The atom/token dimension.
        num_heads : int
            The number of heads.
        channel_cond : int | None
            The single conditioning dimension.
        use_pair_bias : bool
            Whether to use attention pair bias.
        """
        super().__init__()
        self.attn = Attention(channel_a, num_heads, use_high_precision)
        self.linear_g = LinearNoBias(channel_a, channel_a, init="gating")

        self.use_pair_bias: bool = use_pair_bias
        self.use_conditioning: bool = channel_cond is not None
        if self.use_conditioning:
            assert channel_cond is not None
            self.linear_o = LinearNoBias(channel_a, channel_a)
            self.adaln_a = AdaLN(channel_a, channel_cond)
            self.linear_ada_out = Linear(channel_cond, channel_a, init="gating_closed")
        else:
            self.linear_o = LinearNoBias(channel_a, channel_a, init="zero")
            self.layernorm_a = LayerNorm(channel_a, create_offset=True)

    def forward(
        self,
        a: torch.Tensor,
        mask: torch.Tensor,
        pair_bias: torch.Tensor | None = None,
        single_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        a : torch.Tensor
            The input tensor (*, L, c_a).
        mask : torch.Tensor
            The attention mask tensor (*, L) or (*, L, L)
        pair_bias : torch.Tensor | None
            The optional attention bias tensor (*, H, L, L)
        single_cond : torch.Tensor | None
            The optional single conditioning tensor (*, L, c_s).

        Returns
        -------
        a : torch.Tensor
            The output tensor (*, L, c_a)
        """
        if self.use_pair_bias:
            assert pair_bias is not None
        else:
            assert pair_bias is None

        if self.use_conditioning:
            assert single_cond is not None
            a = self.adaln_a(a, single_cond)
        else:
            assert single_cond is None
            a = self.layernorm_a(a)

        if mask.ndim == a.ndim - 1:
            mask = mask.unsqueeze(-2)  # [*, L] -> [*, 1, L]

        out = self.attn(a, a, mask, pair_bias)  # [*, L, c]

        # Gate output
        g = torch.sigmoid(self.linear_g(a))  # [*, L, c]
        a = self.linear_o(g * out)  # [*, L, c]

        if self.use_conditioning:
            assert single_cond is not None
            a = torch.sigmoid(self.linear_ada_out(single_cond)) * a
        return a


class CrossAttention(nn.Module):
    def __init__(
        self,
        channel_a: int,
        num_heads: int,
        channel_cond: int | None,
        *,
        use_pair_bias: bool = True,
        use_high_precision: bool = False,
    ) -> None:
        """Initialize the attention pair bias layer.

        Parameters
        ----------
        channel_a : int
            The atom/token dimension.
        num_heads : int
            The number of heads.
        channel_cond : int | None
            The single conditioning dimension.
        use_pair_bias : bool
            Whether to use attention pair bias.
        """
        super().__init__()
        self.attn = Attention(channel_a, num_heads, use_high_precision)
        self.linear_g = LinearNoBias(channel_a, channel_a, init="gating")

        self.use_pair_bias: bool = use_pair_bias
        self.use_conditioning: bool = channel_cond is not None

        if self.use_conditioning:
            assert channel_cond is not None
            self.linear_o = LinearNoBias(channel_a, channel_a)
            self.adaln_a_q = AdaLN(channel_a, channel_cond)
            self.adaln_a_k = AdaLN(channel_a, channel_cond)
            self.linear_ada_out = Linear(channel_cond, channel_a, init="gating_closed")
        else:
            self.linear_o = LinearNoBias(channel_a, channel_a, init="zero")
            self.layernorm_a_q = LayerNorm(channel_a, create_offset=True)
            self.layernorm_a_k = LayerNorm(channel_a, create_offset=True)

    def forward(
        self,
        a_q: torch.Tensor,
        a_k: torch.Tensor,
        mask: torch.Tensor,
        pair_bias: torch.Tensor | None = None,
        single_cond_q: torch.Tensor | None = None,
        single_cond_k: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        a_q : torch.Tensor
            The query input tensor (*, Lq, c_a).
        a_k : torch.Tensor
            The key/value input tensor (*, Lk, c_a).
        mask : torch.Tensor
            The attention mask tensor (*, Lk) or (*, Lq, Lk)
        pair_bias : torch.Tensor | None
            The attention bias tensor (*, H, Lq, Lk)
        single_cond_q : torch.Tensor | None
            The query single conditioning tensor (*, Lq, c_cond).
        single_cond_k : torch.Tensor | None
            The key/value single conditioning tensor (*, Lk, c_cond).


        Returns
        -------
        a_q : torch.Tensor
            The output query tensor. (*, W, Lq, c_a)
        """
        if self.use_conditioning:
            assert single_cond_q is not None and single_cond_k is not None
            a_q = self.adaln_a_q(a_q, single_cond_q)
            a_k = self.adaln_a_k(a_k, single_cond_k)
        else:
            assert single_cond_q is None and single_cond_k is None
            a_q = self.layernorm_a_q(a_q)
            a_k = self.layernorm_a_k(a_k)

        if self.use_pair_bias:
            assert pair_bias is not None
        else:
            assert pair_bias is None

        if mask.ndim == a_q.ndim - 1:
            mask = mask.unsqueeze(-2)  # [*, Lk] -> [*, 1, Lk]

        # Prepare attention pair bias input
        # [*, W, Lq/k, c] -> [*, W, H, Lq/k, c_h]
        out = self.attn(a_q, a_k, mask, pair_bias)  # [*, Lq, c]

        # Gate output
        g = torch.sigmoid(self.linear_g(a_q))  # [*, Lq, c]
        a_q = self.linear_o(g * out)  # [*, Lq, c]

        if self.use_conditioning:
            assert single_cond_q is not None
            a_q = torch.sigmoid(self.linear_ada_out(single_cond_q)) * a_q
        return a_q
