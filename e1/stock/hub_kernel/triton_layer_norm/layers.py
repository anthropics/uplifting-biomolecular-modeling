import torch
from torch import nn

from .layer_norm import rms_norm_fn


class LlamaRMSNorm(nn.Module):
    """
    RMS Layer Norm for Llama models.

    Triton-optimized RMS layer norm. The interface is compatible with `LLamaRMSNorm` in
    `transformers`.

    Attributes:
        weight (`torch.Tensor`): The learnable scaling parameter.
        variance_epsilon (`float`): The epsilon value for numerical stability.
    """
    weight: torch.Tensor
    variance_epsilon: float

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization to the input hidden states.

        Args:
            hidden_states (`torch.Tensor`):
                Input tensor of shape `(batch_size, sequence_length, hidden_size)` or any shape
                where the last dimension is the feature dimension to be normalized.

        Returns:
            `torch.Tensor`:
                The normalized tensor with the same shape as the input `hidden_states`.
        """
        return rms_norm_fn(
            hidden_states,
            self.weight,
            bias=None,
            residual=None,
            eps=self.variance_epsilon,
            dropout_p=0.0,
            prenorm=False,
            residual_in_fp32=False,
        )


__all__ = ["LlamaRMSNorm"]
