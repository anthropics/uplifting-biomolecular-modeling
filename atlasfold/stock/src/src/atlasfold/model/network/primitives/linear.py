import torch
import torch.nn as nn
import torch.nn.functional as F

from . import initialize


class Linear(nn.Module):
    """A linear layer with various initialization methods."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: str = "default",
        precision: str | int | torch.dtype | None = None,
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters(init)

        if isinstance(precision, str):
            assert precision in {"bfloat16", "float32"}, (
                f"Unsupported precision string: {precision}. "
                "Supported values are 'bfloat16' and 'float32'."
            )
            precision = getattr(torch, precision)
        elif isinstance(precision, int):
            if precision == 32:
                precision = torch.float32
            else:
                raise ValueError(
                    f"Unsupported precision integer: {precision}. Supported values is 32."
                )

        self.precision: torch.dtype | None = precision

    def reset_parameters(self, init: str) -> None:
        # Before initialization, set bias to zero if it exists
        if self.bias is not None:
            initialize.bias_zero_init_(self.bias)

        if init == "default":
            initialize.lecun_normal_init_(self.weight)
        elif init == "relu":
            initialize.he_normal_init_(self.weight)
        elif init == "gating":
            # weight: zero
            initialize.zero_init_(self.weight)
        elif init == "gating_closed":
            # weight: zero, bias: -2 (so that sigmoid(bias) ~= 0.12)
            assert self.bias is not None, (
                "Bias must be True for gating_closed initialization."
            )
            initialize.zero_init_(self.weight)
            initialize.bias_init_(self.bias, value=-2.0)
        elif init == "gating_opened":
            # weight: zero, bias: +2 (so that sigmoid(bias) ~= 0.88)
            assert self.bias is not None, (
                "Bias must be True for gating_opened initialization."
            )
            initialize.zero_init_(self.weight)
            initialize.bias_init_(self.bias, value=+2.0)
        elif init == "final":
            # weight: zero
            initialize.zero_init_(self.weight)
        elif init == "zero":
            # weight: zero
            initialize.zero_init_(self.weight)
        else:
            raise ValueError(f"Unknown initialization method: {init}")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.precision is not None:
            d = self.precision
            out_d = input.dtype
            weight = self.weight.to(d)
            bias = self.bias.to(d) if self.bias is not None else None
            with torch.autocast(input.device.type, enabled=False):
                out = F.linear(input.to(d), weight, bias)

            return out.to(out_d)

        return F.linear(input, self.weight, self.bias)


class LinearNoBias(Linear):
    """A linear layer without bias term."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init: str = "default",
        precision: str | int | torch.dtype | None = None,
    ):
        super().__init__(
            in_features,
            out_features,
            bias=False,
            init=init,
            precision=precision,
        )
