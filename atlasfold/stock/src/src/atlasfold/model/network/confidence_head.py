import torch
import torch.nn as nn
import torch.nn.functional as F

from atlasfold.model.network.block import PairStack
from atlasfold.model.network.primitives import LayerNorm, LinearNoBias
from atlasfold.utils.torch_utils import get_context_dtype, index_select_dim


def get_bin_centers(
    min_value: float,
    max_value: float,
    num_bins: int,
    device: torch.device,
) -> torch.Tensor:
    """Get the centers of the bins for a given range and number of bins."""
    d = (max_value - min_value) / num_bins
    return torch.linspace(min_value + d / 2, max_value - d / 2, num_bins, device=device)


@torch.no_grad()
def get_distogram(
    x: torch.Tensor,
    cbeta_idx: torch.Tensor,
    boundaries: torch.Tensor,
) -> torch.Tensor:
    """Compute the distogram from the predicted coordinates."""
    with torch.autocast(x.device.type, enabled=False):
        x_repr = index_select_dim(
            x, dim=-2, index=cbeta_idx[..., None, None]
        )  # [*, L, 3]
        d = (x_repr[..., :, None, :] - x_repr[..., None, :, :]).norm(dim=-1)  # [*, L, L]
        num_bins = boundaries.shape[0] + 1
        return F.one_hot(
            (d[..., None] > boundaries).sum(dim=-1), num_bins
        )  # [*, L, L, num_bins]


class ExperimentallyResolvedHead(nn.Module):
    """Predicts the probability of each atom being experimentally resolved"""

    def __init__(self, channel_s: int = 384) -> None:
        super().__init__()
        self.head = nn.Sequential(
            LayerNorm(channel_s),
            LinearNoBias(channel_s, 37, init="final"),
        )

    def forward(self, s: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass of experimentally resolved head module.

        Parameters
        ----------
        s : torch.Tensor
            The single representation, shape [B, L, c_s].

        Returns
        -------
        logits: torch.Tensor
            The predicted resolved atom logits, shape [B, L, 37].
        """
        return self.head(s)  # [B, L, 37]


class PredictedLDDTHead(nn.Module):
    """Predicts the predicted lDDT-Calpha score for each residue."""

    def __init__(
        self,
        channel_s: int = 384,
        num_bins: int = 50,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            LayerNorm(channel_s),
            LinearNoBias(channel_s, num_bins, init="final"),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Forward pass of predicted lDDT head module.

        Parameters
        ----------
        s : torch.Tensor
            The single representation, shape [B, L, c_s].
        z : torch.Tensor
            The pair representation, shape [B, L, L, c_z].

        Returns
        -------
        logits: torch.Tensor
            The predicted lDDT-Calpha logits, shape [B, L, num_bins].
        """
        return self.head(s)  # [B, L, num_bins]


class PredictedAlignedErrorHead(nn.Module):
    """Predicts the predicted aligned error (PAE) for each residue pair."""

    def __init__(
        self,
        channel_z: int = 128,
        num_bins: int = 64,
    ):
        super().__init__()
        self.num_bins: int = num_bins
        self.head = nn.Sequential(
            LayerNorm(channel_z),
            LinearNoBias(channel_z, num_bins, init="final"),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass of predicted aligned error head module.

        Parameters
        ----------
        z : torch.Tensor
            The pair representation, shape [B, L, L, c_z].
        mask : torch.Tensor
            The mask for valid residues, shape [B, L].

        Returns
        -------
        logits: torch.Tensor
            The predicted PAE logits, shape [B, L, L, num_bins].
        """
        return self.head(z.float())  # [B, L, L, num_bins]


class PredictedDistanceErrorHead(nn.Module):
    """Predicts the distance error for each residue pair."""

    def __init__(
        self,
        channel_z: int = 128,
        num_bins: int = 64,
    ):
        super().__init__()
        self.num_bins: int = num_bins
        self.head = nn.Sequential(
            LayerNorm(channel_z),
            LinearNoBias(channel_z, num_bins, init="final"),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass of predicted distance error head module.

        Parameters
        ----------
        z : torch.Tensor
            The pair representation, shape [B, L, L, c_z].

        Returns
        -------
        logits: torch.Tensor
            The predicted PDE logits, shape [B, L, L, num_bins].
        """
        z = z + z.transpose(1, 2)
        return self.head(z.float())  # [B, L, L, num_bins]


class ConfidenceHead_Monomer(nn.Module):
    def __init__(
        self,
        channel_s: int = 384,
        channel_z: int = 128,
        num_heads: int = 16,
        num_tri_heads: int = 4,
        num_blocks: int = 2,
        dropout_z: float = 0.25,
        # distogram bins
        num_bins: int = 39,
        min_dist: float = 3.25,
        max_dist: float = 50.75,
        # head dimensions
        num_plddt_bins: int = 50,
        num_pae_bins: int = 64,
        max_pae_error: float = 32.0,
        # for train
        blocks_per_ckpt: int | None = None,
    ) -> None:
        super().__init__()
        self.embed_aa = LinearNoBias(21, channel_z)

        # Prepare pair representation with distogram features
        self.num_bins: int = num_bins
        self.min_dist: float = min_dist
        self.max_dist: float = max_dist
        self.linear_distogram = LinearNoBias(num_bins, channel_z, init="default")

        # Attention stack for confidence prediction
        self.single_stack: PairStack = PairStack(
            channel_s=channel_s,
            channel_z=channel_z,
            num_heads_attn=num_heads,
            num_heads_tri_attn=num_tri_heads,
            dropout_z=dropout_z,
            num_blocks=num_blocks,
            single_to_pair=False,
            pair_to_pair=True,
            pair_to_single=True,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.num_plddt_bins: int = num_plddt_bins
        self.plddt_head = PredictedLDDTHead(channel_s, num_plddt_bins)
        self.experimentally_resolved_head = ExperimentallyResolvedHead(channel_s)

        self.pair_stack: PairStack = PairStack(
            channel_s=channel_s,
            channel_z=channel_z,
            num_heads_attn=num_heads,
            num_heads_tri_attn=num_tri_heads,
            dropout_z=dropout_z,
            num_blocks=num_blocks,
            single_to_pair=False,
            pair_to_pair=True,
            pair_to_single=False,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.num_pae_bins: int = num_pae_bins
        self.max_pae_error: float = max_pae_error
        self.pae_head = PredictedAlignedErrorHead(channel_z, num_pae_bins)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        s: torch.Tensor,
        z: torch.Tensor,
        x_pred: torch.Tensor,
        kernel_backend: str = "torch",
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Forward pass of confidence head module.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            The input batch.
        s : torch.Tensor
            The single representation, shape [B, L, c_s].
        z : torch.Tensor
            The pair representation, shape [B, L, L, c_z].
        x_pred: torch.Tensor
            The predicted coordinates, shape (B, N, L, 14, 3).
        kernel_backend : str
            Triangle operation backend: "torch" or "cuequiv".
            Defaults to "torch".

        Returns
        -------
        plddt_logits: torch.Tensor
            The predicted local distance difference test (lDDT-Calpha) logits,
            shape [B, N, L, num_plddt_bins].
        experimentally_resolved_logits: torch.Tensor
            The predicted resolved atom logits, shape [B, N, L, 37].
        pae_logits: torch.Tensor
            The predicted aligned error (PAE) logits, shape [B, N, L, L, num_pae_bins].
        """
        # Detach the inputs to prevent gradients from flowing into the trunk
        s, z, x_pred = map(lambda x: x.detach(), (s, z, x_pred))

        # Cast the inputs to the appropriate dtype
        device = s.device
        dtype = get_context_dtype(device.type)
        s, z = s.to(dtype), z.to(dtype)

        aa_emb = self.embed_aa(batch["aatype"])  # [B, L, c_z]
        z = z + aa_emb[:, :, None, :] + aa_emb[:, None, :, :]

        # Prepare the mask
        mask = batch["seq_mask"]  # [B, L]

        # Distogram boundaries
        distogram_boundaries = torch.linspace(
            self.min_dist, self.max_dist, self.num_bins - 1, device=device
        )

        def compute_confidences_single(
            s: torch.Tensor,
            z: torch.Tensor,
            mask: torch.Tensor,
            coords: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            # Clone the representations to avoid in-place modifications
            s = s.clone()

            # Prepare pair representation with distogram features
            # Since z is updated with distogram features, we don't need to clone it
            cbeta_idx = batch["pseudo_beta"]  # [B, L]
            distogram = get_distogram(coords, cbeta_idx, distogram_boundaries)
            z = z + self.linear_distogram(distogram.to(z.dtype))  # [B, L, L, c_z]

            # Run the stack
            _s, _ = self.single_stack(s, z.clone(), mask, kernel_backend)
            _, _z = self.pair_stack(None, z.clone(), mask, kernel_backend)
            s, z = _s.float(), _z.float()
            del _s, _z

            # Compute the confidence logits
            logits = {}
            with torch.autocast(device.type, enabled=False):
                logits["plddt"] = self.plddt_head(s)
                logits["experimentally_resolved"] = self.experimentally_resolved_head(s)
                logits["pae"] = self.pae_head(z)
            return logits

        B, N, L, _, _ = x_pred.shape
        if N == 1:
            logits = compute_confidences_single(s, z, mask, x_pred[:, 0])
            plddt_logits = logits["plddt"].unsqueeze(1)
            exp_resolved_logits = logits["experimentally_resolved"].unsqueeze(1)
            pae_logits = logits["pae"].unsqueeze(1)
        else:
            plddt_logits = torch.zeros(B, N, L, self.num_plddt_bins, device=device)
            exp_resolved_logits = torch.zeros(B, N, L, 37, device=device)
            pae_logits = torch.zeros(B, N, L, L, self.num_pae_bins, device=device)
            for i in range(N):
                logits = compute_confidences_single(s, z, mask, x_pred[:, i])
                plddt_logits[:, i] = logits["plddt"]
                exp_resolved_logits[:, i] = logits["experimentally_resolved"]
                pae_logits[:, i] = logits["pae"]

        out: dict[str, dict[str, torch.Tensor]] = {}
        out["experimentally_resolved"] = {"logits": exp_resolved_logits}
        plddt_bin_centers = get_bin_centers(0.0, 1.0, self.num_plddt_bins, device=device)
        out["plddt"] = {"logits": plddt_logits, "bin_centers": plddt_bin_centers}
        pae_bin_centers = get_bin_centers(
            0.0, self.max_pae_error, self.num_pae_bins, device=device
        )
        out["pae"] = {"logits": pae_logits, "bin_centers": pae_bin_centers}

        return out


class ConfidenceHead_Multimer(nn.Module):
    def __init__(
        self,
        channel_s: int = 384,
        channel_z: int = 128,
        num_heads: int = 16,
        num_tri_heads: int = 4,
        num_blocks: int = 4,
        dropout_z: float = 0.25,
        # distogram bins
        num_bins: int = 39,
        min_dist: float = 3.25,
        max_dist: float = 50.75,
        # head dimensions
        num_plddt_bins: int = 50,
        num_pae_bins: int = 64,
        max_pae_error: float = 32.0,
        num_pde_bins: int = 64,
        max_pde_error: float = 32.0,
        # for train
        blocks_per_ckpt: int | None = None,
    ) -> None:
        super().__init__()
        self.proj_s = nn.Sequential(
            LayerNorm(channel_s),
            LinearNoBias(channel_s, channel_s),
        )
        self.embed_aa = LinearNoBias(21, channel_z)

        # Prepare pair representation with distogram features
        self.num_bins: int = num_bins
        self.min_dist: float = min_dist
        self.max_dist: float = max_dist
        self.linear_distogram = LinearNoBias(num_bins, channel_z, init="default")

        # Attention stack for confidence prediction
        self.stack: PairStack = PairStack(
            channel_s=channel_s,
            channel_z=channel_z,
            num_heads_attn=num_heads,
            num_heads_tri_attn=num_tri_heads,
            dropout_z=dropout_z,
            num_blocks=num_blocks,
            single_to_pair=False,
            pair_to_pair=True,
            pair_to_single=True,
            blocks_per_ckpt=blocks_per_ckpt,
        )

        # plddt
        self.num_plddt_bins: int = num_plddt_bins
        self.plddt_head = PredictedLDDTHead(channel_s, num_plddt_bins)
        self.experimentally_resolved_head = ExperimentallyResolvedHead(channel_s)

        # pae
        self.num_pae_bins: int = num_pae_bins
        self.max_pae_error: float = max_pae_error
        self.pae_head = PredictedAlignedErrorHead(channel_z, num_pae_bins)

        # pde
        self.num_pde_bins: int = num_pde_bins
        self.max_pde_error: float = max_pde_error
        self.pde_head = PredictedDistanceErrorHead(channel_z, num_pde_bins)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        s: torch.Tensor,
        z: torch.Tensor,
        x_pred: torch.Tensor,
        kernel_backend: str = "torch",
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Forward pass of confidence head module.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            The input batch.
        s : torch.Tensor
            The single representation, shape [B, L, c_s].
        z : torch.Tensor
            The pair representation, shape [B, L, L, c_z].
        x_pred: torch.Tensor
            The predicted coordinates, shape (B, N, L, 14, 3).
        kernel_backend : str
            Triangle operation backend: "torch" or "cuequiv".
            Defaults to "torch".

        Returns
        -------
        plddt_logits: torch.Tensor
            The predicted local distance difference test (lDDT-Calpha) logits,
            shape [B, N, L, num_plddt_bins].
        experimentally_resolved_logits: torch.Tensor
            The predicted resolved atom logits, shape [B, N, L, 37].
        pae_logits: torch.Tensor
            The predicted aligned error (PAE) logits, shape [B, N, L, L, num_pae_bins].
        pde_logits: torch.Tensor
            The predicted distance error (PDE) logits, shape [B, N, L, L, num_pde_bins].
        """
        # Detach the inputs to prevent gradients from flowing into the trunk
        s, z, x_pred = map(lambda x: x.detach(), (s, z, x_pred))

        # Cast the inputs to the appropriate dtype
        device = s.device
        dtype = get_context_dtype(device.type)
        s, z = s.to(dtype, copy=True), z.to(dtype, copy=True)

        s = self.proj_s(s)  # [B, L, c_a]

        aa_emb = self.embed_aa(batch["aatype"])  # [B, L, c_z]
        z = z + aa_emb[:, :, None, :] + aa_emb[:, None, :, :]

        # Prepare the mask
        mask = batch["seq_mask"]  # [B, L]

        distogram_boundaries = torch.linspace(
            self.min_dist, self.max_dist, self.num_bins - 1, device=device
        )

        def compute_confidences_single(
            s: torch.Tensor,
            z: torch.Tensor,
            mask: torch.Tensor,
            coords: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            # Clone the representations to avoid in-place modifications
            s = s.clone()

            # Prepare pair representation with distogram features
            # Since z is updated with distogram features, we don't need to clone it
            cbeta_idx = batch["pseudo_beta"]  # [B, L]
            distogram = get_distogram(coords, cbeta_idx, distogram_boundaries)
            z = z + self.linear_distogram(distogram.to(z.dtype))  # [B, L, L, c_z]

            # Run the stack
            s, z = self.stack(s, z, mask, kernel_backend=kernel_backend)
            s, z = s.float(), z.float()

            # Compute the confidence logits
            logits = {}
            with torch.autocast(device.type, enabled=False):
                logits["plddt"] = self.plddt_head(s)
                logits["experimentally_resolved"] = self.experimentally_resolved_head(s)
                logits["pae"] = self.pae_head(z)
                logits["pde"] = self.pde_head(z)
            return logits

        B, N, L, _, _ = x_pred.shape
        if N == 1:
            logits = compute_confidences_single(s, z, mask, x_pred[:, 0])
            plddt_logits = logits["plddt"].unsqueeze(1)
            exp_resolved_logits = logits["experimentally_resolved"].unsqueeze(1)
            pae_logits = logits["pae"].unsqueeze(1)
            pde_logits = logits["pde"].unsqueeze(1)
        else:
            plddt_logits = torch.zeros(B, N, L, self.num_plddt_bins, device=device)
            exp_resolved_logits = torch.zeros(B, N, L, 37, device=device)
            pae_logits = torch.zeros(B, N, L, L, self.num_pae_bins, device=device)
            pde_logits = torch.zeros(B, N, L, L, self.num_pde_bins, device=device)

            for i in range(N):
                logits = compute_confidences_single(s, z, mask, x_pred[:, i])
                plddt_logits[:, i] = logits["plddt"]
                exp_resolved_logits[:, i] = logits["experimentally_resolved"]
                pae_logits[:, i] = logits["pae"]
                pde_logits[:, i] = logits["pde"]

        out: dict[str, dict[str, torch.Tensor]] = {}
        out["experimentally_resolved"] = {"logits": exp_resolved_logits}
        plddt_bin_centers = get_bin_centers(0.0, 1.0, self.num_plddt_bins, device=device)
        out["plddt"] = {"logits": plddt_logits, "bin_centers": plddt_bin_centers}
        pae_bin_centers = get_bin_centers(
            0.0, self.max_pae_error, self.num_pae_bins, device=device
        )
        out["pae"] = {"logits": pae_logits, "bin_centers": pae_bin_centers}
        pde_bin_centers = get_bin_centers(
            0.0, self.max_pde_error, self.num_pde_bins, device=device
        )
        out["pde"] = {"logits": pde_logits, "bin_centers": pde_bin_centers}

        return out
