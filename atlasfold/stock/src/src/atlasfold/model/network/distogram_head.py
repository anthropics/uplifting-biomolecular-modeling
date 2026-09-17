import einops
import torch

from atlasfold.model.network.primitives import LinearNoBias


class DistogramHead(torch.nn.Module):
    def __init__(
        self,
        channel_z: int,
        num_bins: int,
        min_dist: float,
        max_dist: float,
    ) -> None:
        super().__init__()
        self.num_bins: int = num_bins
        self.min_dist: float = min_dist
        self.max_dist: float = max_dist

        min_d, max_d = self.min_dist, self.max_dist
        bin_size: float = (max_d - min_d) / self.num_bins  # =0.3125
        self.first_bin: float = min_d + bin_size  # =2.3125
        self.last_bin: float = max_d - bin_size  # =21.6875
        self.contact_bin = int((8.0 - self.first_bin) / bin_size)  # = 18

        self.linear = LinearNoBias(channel_z, num_bins, init="final")

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass of distogram head module.

        Parameters
        ----------
        z : torch.Tensor
            Tensor of shape (*, L, L, c_z) containing pair feature

        Returns
        -------
        logits: torch.Tensor
            Tensor of shape (*, L, L, num_bins) containing distogram logits.
        boundaries: torch.Tensor
            Tensor of shape (num_bins - 1,) containing the boundaries of the bins.
        """
        logits = self.linear(z)
        # symmetrize logits
        logits = logits + einops.rearrange(logits, "... i j c -> ... j i c")

        boundaries = torch.linspace(
            self.first_bin, self.last_bin, self.num_bins - 1, device=z.device
        )
        return {"logits": logits, "boundaries": boundaries}
