import math

import einops
import torch
import torch.nn as nn

try:
    from cuequivariance_torch.primitives.triangle import (
        triangle_attention as _cueq_triangle_attention,
    )
    from cuequivariance_torch.primitives.triangle import (
        triangle_multiplicative_update as _cueq_triangle_multiplicative_update,
    )
except ImportError:
    _cueq_triangle_attention = None
    _cueq_triangle_multiplicative_update = None

from .linear import LinearNoBias
from .normalization import LayerNorm


@torch.compiler.disable
def cueq_tri_mul_outgoing(
    z: torch.Tensor,
    mask: torch.Tensor,
    norm_in_weight: torch.Tensor,
    norm_in_bias: torch.Tensor,
    p_in_weight: torch.Tensor,
    g_in_weight: torch.Tensor,
    norm_out_weight: torch.Tensor,
    norm_out_bias: torch.Tensor,
    p_out_weight: torch.Tensor,
    g_out_weight: torch.Tensor,
    eps: float,
):
    if _cueq_triangle_multiplicative_update is None:
        raise ImportError(
            "cuequivariance_torch is not installed. "
            "Please install cuequivariance_torch to use the kernel implementation."
        )
    original_shape = z.shape
    return _cueq_triangle_multiplicative_update(
        z.view(-1, *original_shape[-3:]),
        direction="outgoing",
        mask=mask.view(-1, *mask.shape[-2:]),
        norm_in_weight=norm_in_weight,
        norm_in_bias=norm_in_bias,
        p_in_weight=p_in_weight,
        g_in_weight=g_in_weight,
        norm_out_weight=norm_out_weight,
        norm_out_bias=norm_out_bias,
        p_out_weight=p_out_weight,
        g_out_weight=g_out_weight,
        eps=eps,
    ).view(*original_shape)


@torch.compiler.disable
def cueq_tri_mul_incoming(
    z: torch.Tensor,
    mask: torch.Tensor,
    norm_in_weight: torch.Tensor,
    norm_in_bias: torch.Tensor,
    p_in_weight: torch.Tensor,
    g_in_weight: torch.Tensor,
    norm_out_weight: torch.Tensor,
    norm_out_bias: torch.Tensor,
    p_out_weight: torch.Tensor,
    g_out_weight: torch.Tensor,
    eps: float,
):
    if _cueq_triangle_multiplicative_update is None:
        raise ImportError(
            "cuequivariance_torch is not installed. "
            "Please install cuequivariance_torch to use the kernel implementation."
        )
    original_shape = z.shape
    return _cueq_triangle_multiplicative_update(
        z.view(-1, *original_shape[-3:]),
        direction="incoming",
        mask=mask.view(-1, *mask.shape[-2:]),
        norm_in_weight=norm_in_weight,
        norm_in_bias=norm_in_bias,
        p_in_weight=p_in_weight,
        g_in_weight=g_in_weight,
        norm_out_weight=norm_out_weight,
        norm_out_bias=norm_out_bias,
        p_out_weight=p_out_weight,
        g_out_weight=g_out_weight,
        eps=eps,
    ).view(*original_shape)


@torch.compiler.disable
def cueq_tri_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if _cueq_triangle_attention is None:
        raise ImportError(
            "cuequivariance_torch is not installed. "
            "Please install cuequivariance_torch to use the kernel implementation."
        )
    original_shape = q.shape
    return _cueq_triangle_attention(  # type: ignore
        q.view(-1, *q.shape[-4:]),
        k.view(-1, *k.shape[-4:]),
        v.view(-1, *v.shape[-4:]),
        bias.view(-1, *bias.shape[-4:]).float(),
        mask=mask.view(-1, *mask.shape[-4:]),
        scale=scale,
    ).view(*original_shape)


class TriangleMultiplicationOutgoing(nn.Module):
    """TriangleMultiplication.
    See Section 3.4 Algorithm 12 of AlphaFold2 paper.
    """

    def __init__(self, channel: int) -> None:
        """Initialize the TriangularUpdate module.

        Parameters
        ----------
        channel: int
            The input channel dimension.
        """
        super().__init__()
        self.layernorm_in = LayerNorm(channel)
        self.linear_in = LinearNoBias(channel, 2 * channel, init="default")
        self.linear_g_in = LinearNoBias(channel, 2 * channel, init="gating")

        self.layernorm_out = LayerNorm(channel)
        self.linear_out = LinearNoBias(channel, channel, init="final")
        self.linear_g_out = LinearNoBias(channel, channel, init="gating")

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        kernel_backend: str = "torch",
    ) -> torch.Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        z: torch.Tensor
            The input data of shape (*, L, L, C)
        mask: torch.Tensor
            The input mask of shape (*, L, L)
        kernel_backend: str
            Triangle operation backend: "torch" or "cuequiv".
            Defaults to "torch".

        Returns
        -------
        x: torch.Tensor
            The output data of shape (*, L, L, C)

        """
        if (
            kernel_backend == "cuequiv"
            and _cueq_triangle_multiplicative_update is not None
        ):
            return cueq_tri_mul_outgoing(
                z,
                mask=mask,
                norm_in_weight=self.layernorm_in.weight,
                norm_in_bias=self.layernorm_in.bias,
                p_in_weight=self.linear_in.weight,
                g_in_weight=self.linear_g_in.weight,
                norm_out_weight=self.layernorm_out.weight,
                norm_out_bias=self.layernorm_out.bias,
                p_out_weight=self.linear_out.weight,
                g_out_weight=self.linear_g_out.weight,
                eps=1e-5,
            )

        # Line 1
        z = self.layernorm_in(z)
        z = z * mask.unsqueeze(-1)

        # Line 2
        a, b = torch.chunk(self.linear_g_in(z).sigmoid() * self.linear_in(z), 2, dim=-1)

        # Line 3
        g = self.linear_g_out(z).sigmoid()

        # Line 4
        # a, b shape: (*, L, L, C) -> move to (*, C, L, L)
        a, b = a.movedim(-1, -3), b.movedim(-1, -3)
        z = a @ b.mT
        # move back to (*, L, L, C)
        z = z.movedim(-3, -1)
        z = g * self.linear_out(self.layernorm_out(z))

        return z


class TriangleMultiplicationIncoming(nn.Module):
    """TriangleMultiplication.
    See Section 3.4 Algorithm 13 of AlphaFold2 paper.
    """

    def __init__(self, channel: int) -> None:
        """Initialize the TriangularUpdate module.

        Parameters
        ----------
        channel: int
            The input channel dimension.

        """
        super().__init__()

        self.layernorm_in = LayerNorm(channel)
        self.linear_in = LinearNoBias(channel, 2 * channel, init="default")
        self.linear_g_in = LinearNoBias(channel, 2 * channel, init="gating")

        self.layernorm_out = LayerNorm(channel)
        self.linear_out = LinearNoBias(channel, channel, init="final")
        self.linear_g_out = LinearNoBias(channel, channel, init="gating")

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        kernel_backend: str = "torch",
    ) -> torch.Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        z: torch.Tensor
            The input data of shape (*, L, L, C)
        mask: torch.Tensor
            The input mask of shape (*, L, L)
        kernel_backend: str
            Triangle operation backend: "torch" or "cuequiv".
            Defaults to "torch".

        Returns
        -------
        x: torch.Tensor
            The output data of shape (*, L, L, C)

        """
        if (
            kernel_backend == "cuequiv"
            and _cueq_triangle_multiplicative_update is not None
        ):
            return cueq_tri_mul_incoming(
                z,
                mask=mask,
                norm_in_weight=self.layernorm_in.weight,
                norm_in_bias=self.layernorm_in.bias,
                p_in_weight=self.linear_in.weight,
                g_in_weight=self.linear_g_in.weight,
                norm_out_weight=self.layernorm_out.weight,
                norm_out_bias=self.layernorm_out.bias,
                p_out_weight=self.linear_out.weight,
                g_out_weight=self.linear_g_out.weight,
                eps=1e-5,
            )

        # Line 1
        z = self.layernorm_in(z)
        z = z * mask.unsqueeze(-1)

        # Line 2
        a, b = torch.chunk(self.linear_g_in(z).sigmoid() * self.linear_in(z), 2, dim=-1)

        # Line 3
        g = self.linear_g_out(z).sigmoid()

        # Line 4
        # a, b shape: (*, L, L, C) -> move to (*, C, L, L)
        a, b = a.movedim(-1, -3), b.movedim(-1, -3)
        z = a.mT @ b
        # move back to (*, L, L, C)
        z = z.movedim(-3, -1)
        z = g * self.linear_out(self.layernorm_out(z))

        return z


class TriangleAttentionStartingNode(nn.Module):
    """TriangleAttention with starting=True.
    See Section 3.4 Algorithm 13 in the AlphaFold3 paper.
    """

    def __init__(self, channel: int, num_heads: int, inf: float = 1e9) -> None:
        super().__init__()
        assert channel % num_heads == 0, (
            f"channel ({channel}) must be divisible by num_heads ({num_heads})"
        )

        self.channel: int = channel
        self.num_heads: int = num_heads
        self.channel_hidden: int = channel // num_heads
        self.inf: float = inf
        self.layernorm = LayerNorm(self.channel)
        self.linear_bias = LinearNoBias(self.channel, self.num_heads)

        self.linear_qkv = LinearNoBias(self.channel, self.channel * 3, init="default")
        self.linear_out = LinearNoBias(self.channel, self.channel, init="final")
        self.linear_g = LinearNoBias(self.channel, self.channel, init="gating")
        self.scale = 1.0 / math.sqrt(self.channel_hidden)

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        kernel_backend: str = "torch",
    ) -> torch.Tensor:
        """Compute triangle attention.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor of shape (*, L, L, C)
        mask : torch.Tensor
            Attention mask of shape (*, L, L)
        kernel_backend : str
            Triangle operation backend: "torch" or "cuequiv".
            Defaults to "torch".

        Returns
        -------
        torch.Tensor
            Output tensor of shape (*, L, L, C)

        """
        # Line 1: Initial layer norm
        z = self.layernorm(z)

        # Line 2: Prepare q, k, v
        # (*, L, H, L, Ch)
        q, k, v = self.linear_qkv(z).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: einops.rearrange(t, "... l (h c) -> ... h l c", h=self.num_heads),
            (q, k, v),
        )

        # Line 3: Prepare bias and mask
        # (*, 1, H, L, L)
        bias = self.linear_bias(z)
        bias = einops.rearrange(bias, "... i j h -> ... 1 h i j")
        # (*, L, 1, 1, L)
        mask = einops.rearrange(mask, "... i j -> ... i 1 1 j")

        # Line 4: Prepare gating
        # (*, L, L, H, C_h)
        g = torch.sigmoid(self.linear_g(z))
        g = einops.rearrange(g, "... (h c) -> ... h c", h=self.num_heads)

        # Summary:
        # q,k,v (*, L, H, L, Ch)
        # bias  (*, 1, H, L, L)
        # mask  (*, L, 1, 1, L)
        # g     (*, L, L, C)

        # Line 5-6: Attention
        # (*, L, H, L, L)
        if kernel_backend == "cuequiv" and _cueq_triangle_attention is not None:
            out = cueq_tri_attn(
                q,
                k,
                v,
                bias=bias,
                mask=mask,
                scale=self.scale,
            )
        else:
            q *= self.scale
            k = k.transpose(-1, -2)
            a = torch.matmul(q, k)
            # Apply mask and bias
            a += (-self.inf) * (~mask.bool()).to(q.dtype)
            a += bias
            # Return attention output
            a = a.softmax(dim=-1)
            out = torch.matmul(a, v)

        # Re-arrange output
        # (*, L, H, L, C_h) -> (*, L, L, H, C_h)
        out = einops.rearrange(out, "... h l c -> ... l h c")

        # Gating
        out = g * out

        # Line 7: Output projection
        out = self.linear_out(out.flatten(-2))

        return out


class TriangleAttentionEndingNode(nn.Module):
    """TriangleAttention with starting=False.
    See Section 3.4 Algorithm 14 in the AlphaFold3 paper.
    """

    def __init__(self, channel: int, num_heads: int, inf: float = 1e9) -> None:
        super().__init__()
        assert channel % num_heads == 0, (
            f"channel ({channel}) must be divisible by num_heads ({num_heads})"
        )

        self.channel: int = channel
        self.num_heads: int = num_heads
        self.channel_hidden: int = channel // num_heads
        self.inf: float = inf
        self.layernorm = LayerNorm(self.channel)
        self.linear_bias = LinearNoBias(self.channel, self.num_heads)

        self.linear_qkv = LinearNoBias(self.channel, self.channel * 3, init="default")
        self.linear_out = LinearNoBias(self.channel, self.channel, init="final")
        self.linear_g = LinearNoBias(self.channel, self.channel, init="gating")
        self.scale = 1.0 / math.sqrt(self.channel_hidden)

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        kernel_backend: str = "torch",
    ) -> torch.Tensor:
        """Compute triangle attention.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor of shape (*, L, L, C)
        mask : torch.Tensor
            Attention mask of shape (*, L, L)
        kernel_backend : str
            Triangle operation backend: "torch" or "cuequiv".
            Defaults to "torch".

        Returns
        -------
        torch.Tensor
            Output tensor of shape (*, L, L, C)

        """
        z = z.transpose(-2, -3)
        mask = mask.transpose(-1, -2)

        # Line 1: Initial layer norm
        z = self.layernorm(z)

        # Line 2: Prepare q, k, v
        # (*, L, H, L, Ch)
        q, k, v = self.linear_qkv(z).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: einops.rearrange(t, "... l (h c) -> ... h l c", h=self.num_heads),
            (q, k, v),
        )

        # Line 3: Prepare bias and mask
        # (*, 1, H, L, L)
        bias = self.linear_bias(z)
        bias = einops.rearrange(bias, "... i j h -> ... 1 h i j")
        # (*, L, 1, 1, L)
        mask = einops.rearrange(mask, "... i j -> ... i 1 1 j")

        # Line 4: Prepare gating
        # (*, L, L, H, C_h)
        g = torch.sigmoid(self.linear_g(z))
        g = einops.rearrange(g, "... (h c) -> ... h c", h=self.num_heads)

        # Summary:
        # q,k,v (*, L, H, L, Ch)
        # bias  (*, 1, H, L, L)
        # mask  (*, L, 1, 1, L)
        # g     (*, L, L, C)

        # Line 5-6: Attention
        # (*, L, H, L, L)
        if kernel_backend == "cuequiv" and _cueq_triangle_attention is not None:
            out = cueq_tri_attn(
                q,
                k,
                v,
                bias=bias,
                mask=mask,
                scale=self.scale,
            )
        else:
            q *= self.scale
            k = k.transpose(-1, -2)
            a = torch.matmul(q, k)
            # Apply mask and bias
            a += (-self.inf) * (~mask.bool()).to(q.dtype)
            a += bias
            # Return attention output
            a = a.softmax(dim=-1)
            out = torch.matmul(a, v)

        # Re-arrange output
        # (*, L, H, L, C_h) -> (*, L, L, H, C_h)
        out = einops.rearrange(out, "... h l c -> ... l h c")

        # Gating
        out = g * out

        # Line 7: Output projection
        out = self.linear_out(out.flatten(-2))

        out = out.transpose(-2, -3)

        return out
