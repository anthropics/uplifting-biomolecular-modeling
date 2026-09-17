import torch


class RegressionHead(torch.nn.Sequential):
    def __init__(
        self,
        d_model: int,
        output_dim: int,
        hidden_dim: int | None = None,
    ):
        hidden_dim = hidden_dim if hidden_dim is not None else d_model
        super().__init__(
            torch.nn.Linear(d_model, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, output_dim),
        )
