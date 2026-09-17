from collections.abc import Sequence

import torch


class Dropout(torch.nn.Module):
    def __init__(self, p: float, dim: int | Sequence[int] | None) -> None:
        super().__init__()
        self.p: float = p
        assert 0.0 <= self.p < 1.0, (
            f"Dropout probability must be in the range [0.0, 1.0), got {self.p}"
        )
        if dim is None:
            self.dim = []
        elif isinstance(dim, int):
            self.dim = [dim]
        else:
            self.dim = list(dim)

    def forward(self, x: torch.Tensor, training: bool | None = None) -> torch.Tensor:
        training = training if training is not None else self.training
        if not training or self.p == 0.0:
            return x

        shape = list(x.shape)
        for d in self.dim:
            shape[d] = 1

        mask = x.new_ones(shape)
        mask = torch.nn.functional.dropout(mask, p=self.p, training=True)
        return x * mask


class DropoutRowwise(Dropout):
    """Row-wise dropout layer."""

    def __init__(self, p: float) -> None:
        super().__init__(p, dim=-3)


class DropoutColumnwise(Dropout):
    """Column-wise dropout layer."""

    def __init__(self, p: float) -> None:
        super().__init__(p, dim=-2)
