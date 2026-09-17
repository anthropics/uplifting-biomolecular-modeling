import math

import torch

from .attention import MultiHeadAttention


class SwiGLU(torch.nn.Module):
    """SwiGLU activation function as an nn.Module"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(x1) * x2


class TransformerBlock(torch.nn.Module):
    """A transformer block

    Parameters
    ----------
    d_model : int
        The dimensionality of the input and output features of the transformer block.
    n_heads : int
        The number of attention heads in the multi-head attention mechanism.
    n_layers : int
        The number of layers in the transformer block.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = False,
        expansion_ratio: float = 4.0,
        residue_scaling_factor: float = 1.0,
        qk_layernorm: bool = True,
    ):
        super().__init__()
        hidden_dim = int(((expansion_ratio * d_model) + 255) // 256 * 256)
        self.attn = MultiHeadAttention(d_model, n_heads, bias, qk_layernorm)
        self.ffn = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, hidden_dim * 2, bias=bias),
            SwiGLU(),
            torch.nn.Linear(hidden_dim, d_model, bias=bias),
        )
        self.scaling_factor: float = 1 / residue_scaling_factor

    def forward(
        self,
        x: torch.Tensor,
        seq_id: torch.Tensor | None = None,
        pos_id: torch.Tensor | None = None,
        return_attn: bool = False,
        return_attn_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        (B, L, D) -> (B, L, D)
        """
        r1, attn = self.attn(x, seq_id, pos_id, return_attn, return_attn_logits)
        x = torch.add(x, r1, alpha=self.scaling_factor)
        r2 = self.ffn(x)
        x = torch.add(x, r2, alpha=self.scaling_factor)
        return x, attn


class TransformerStack(torch.nn.Module):
    """
    A stack of transformer blocks used in ESM-3.

    Reference: "Simulating 500 million years of evolution with a language model"

    Parameters
    ----------
    d_model: int
        The dimensionality of the input and output feature vectors.
    n_heads: int
        The number of attention heads.
    n_layers: int
        The number of transformer blocks in the stack.
    scale_residue: bool
        Whether to scale the residue connections in each transformer block.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        scale_residue: bool = True,
        bias: bool = False,
        qk_layernorm: bool = True,
        expansion_ratio: float = 8 / 3,
    ):
        super().__init__()
        self.d_model: int = d_model
        self.n_heads: int = n_heads
        self.n_layers: int = n_layers

        self.blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    residue_scaling_factor=(
                        math.sqrt(n_layers / 36) if scale_residue else 1.0
                    ),
                    expansion_ratio=expansion_ratio,
                    bias=bias,
                    qk_layernorm=qk_layernorm,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = torch.nn.LayerNorm(d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        seq_id: torch.Tensor | None = None,
        pos_id: torch.Tensor | None = None,
        return_attn: bool = False,
        return_attn_logits: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """
        Forward pass of the TransformerStack.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor of shape (batch_size, seq_len, d_model).

        Returns
        -------
        out: torch.Tensor
            The output tensor of shape (batch_size, seq_len, d_model)
        hiddens: list[torch.Tensor]
            A list of length n_layers containing the hidden states from
            each transformer block.
        attns: list[torch.Tensor]
            A list of length n_layers containing the attention maps from
            each transformer block (if return_attn is True), otherwise None.
        """
        hiddens: list[torch.Tensor] = []
        attns: list[torch.Tensor] = []

        assert x.ndim == 3, "Input tensor must be of shape (batch_size, seq_len, d_model)"
        for block in self.blocks:
            x, attn = block(x, seq_id, pos_id, return_attn, return_attn_logits)
            hiddens.append(x)
            attns.append(attn)

        out = self.norm(x)
        return out, hiddens, attns
