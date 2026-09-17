from functools import partial

import torch

from atlasfold.train.utils.kernels.cdist import cdist as kernel_cdist
from atlasfold.utils.checkpointing import checkpoint_fn
from atlasfold.utils.geometry.rigid_align import rigid_align_atom14_torch


def safe_cdist(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute pairwise distances between two sets of points."""
    d = x[..., :, None, :] - y[..., None, :, :]  # [*, Lx, Ly, 3]
    return torch.sqrt(d.pow(2).sum(-1) + eps)


class MSELoss(torch.nn.Module):
    def forward(
        self,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the MSE loss.

        Parameters
        ----------
        x_pred : torch.Tensor
            Predicted coordinates from the model. Shape [B, N, L, 14, 3].
        x_gt : torch.Tensor
            Ground truth coordinates. Shape [B, L, 14, 3].
        mask : torch.Tensor
            Resolved mask per atom of shape [B, L, 14].

        Returns
        -------
        mse_loss: torch.Tensor
            Computed MSE loss. Shape [B, N].
        """
        x_gt = x_gt.unsqueeze(1).expand(x_pred.shape)  # [B, N, L, 14, 3]
        mask = mask.unsqueeze(1)  # [B, 1, L, 14]

        # Align predicted coordinates to ground truth coordinates using Kabsch algorithm.
        with torch.no_grad():
            x_gt = rigid_align_atom14_torch(x_gt, x_pred, mask)

        # Compute mse loss
        w = mask.float()
        w_sum = w.sum((-1, -2)).clamp_min(1)  # [B, 1]
        d_sq = (x_pred - x_gt).pow(2).sum(-1)  # [B, N, L, 14]
        mse_loss = (1 / 3) * (d_sq * w).sum((-1, -2)) / w_sum  # [B, N]
        return mse_loss


class SmoothLDDTLoss(torch.nn.Module):
    """Smooth LDDT loss of denoised atom positions
    See Section 3.7.1 Algorithm 27 Smooth LDDT Loss of the AlphaFold 3 paper."""

    def __init__(
        self,
        cutoff: float = 15.0,
        calpha_only: bool = False,
        chunk_size: int | None = 1,
        use_kernel: bool = False,
    ):
        """Initialize SmoothLDDTLoss.

        Parameters
        ----------
        cutoff: float
            The distance cutoff.
        calpha_only: bool
            Whether to compute LDDT-Calpha loss instead of full-atom LDDT loss.
        chunk_size: int | None
            The chunk size for computing LDDT loss.
        use_kernel: bool
            Whether to use triton implementation for pairwise distance calculation.
        """

        super().__init__()
        self.cutoff: float = cutoff
        self.calpha_only: bool = calpha_only
        self.chunk_size: int | None = chunk_size
        self.use_kernel: bool = use_kernel

    def forward(
        self,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted alignment.

        Parameters
        ----------
        x_pred : torch.Tensor
            Predicted coordinates from the model. Shape [B, N, L, 14, 3].
        x_gt : torch.Tensor
            Ground truth coordinates. Shape [B, L, 14, 3].
        mask : torch.Tensor
            Resolved mask per atom of shape [B, L, 14].

        Returns
        -------
        lddt_loss: torch.Tensor
            Computed LDDT loss. Shape (B, N).
        """
        batch_size = x_pred.shape[0]
        losses = []
        for b_i in range(batch_size):
            losses.append(
                self._forward_single(
                    x_pred[b_i],  # [N, L, 14, 3]
                    x_gt[b_i],  # [L, 14, 3]
                    mask[b_i],  # [L, 14]
                )
            )
        lddt_loss = torch.stack(losses, dim=0)  # [B, N]
        return lddt_loss

    def _forward_single(
        self,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        N, L, _, _ = x_pred.shape

        # Create pair mask
        atom_mask = mask.view(L * 14)  # [L*14]
        pair_mask = atom_mask[:, None] & atom_mask[None, :]  # [L*14, L*14]
        # Mask out self-term
        pair_mask.diagonal(dim1=0, dim2=1).fill_(False)

        _cdist = kernel_cdist if self.use_kernel else safe_cdist

        x_gt_flat = x_gt.view(L * 14, 3)  # [L*14, 3]
        if self.calpha_only:
            # Extract representative atom indices
            x_gt_calpha = x_gt[:, 1, :]  # [L, 3]
            d_gt = _cdist(x_gt_calpha, x_gt_flat)  # [L, L*14]
            pair_mask = pair_mask.view(L, 14, L * 14)[:, 1, :]  # [L, L*14]
        else:
            d_gt = _cdist(x_gt_flat, x_gt_flat)  # [L*14, L*14]

        # Mask out invalid distances
        pair_mask &= d_gt < self.cutoff  # [L*14, L*14] or [L, L*14]

        loss_fn = partial(
            self._chunk_forward,
            d_gt=d_gt,  # [L, L*14] or [L*14, L*14]
            pair_mask=pair_mask.float(),  # [L, L] or [L, L*14]
            calpha_only=self.calpha_only,
            use_kernel=self.use_kernel,
        )

        if self.chunk_size is None:
            loss = loss_fn(x_pred)  # [N,]
        else:
            losses = []
            for i in range(0, N, self.chunk_size):
                st, end = i, i + self.chunk_size
                x_chunk = x_pred[st:end]  # [chunk_size, L, 14, 3]
                loss_chunk = checkpoint_fn(
                    loss_fn,
                    x_chunk,
                    use_reentrant=False,
                    determinism_check="none",  # No randomness in loss function
                )
                losses.append(loss_chunk)
            loss = torch.cat(losses, dim=0)  # [N,]
        return loss

    @staticmethod
    def _chunk_forward(
        x_pred: torch.Tensor,
        d_gt: torch.Tensor,
        pair_mask: torch.Tensor,
        calpha_only: bool = False,
        use_kernel: bool = False,
    ) -> torch.Tensor:
        _cdist = kernel_cdist if use_kernel else safe_cdist
        N, L, _, _ = x_pred.shape

        x_pred_flat = x_pred.view(N, L * 14, 3)  # [N, L*14, 3]
        if calpha_only:
            # Compute predicted distances between representative atoms and all atoms
            x_pred_calpha = x_pred[:, :, 1, :]  # [N, L, 3]
            d_pred = _cdist(x_pred_calpha, x_pred_flat)  # [N, L, L*14]
        else:
            # Compute predicted pairwise distances (original AF3)
            d_pred = _cdist(x_pred_flat, x_pred_flat)  # [N, L*14, L*14]

        d_diff = torch.abs(d_pred - d_gt[None, ...])  # [N, L, L*14] or [N, L*14, L*14]

        lddt_score = (1 / 4) * (
            torch.sigmoid(0.5 - d_diff)
            + torch.sigmoid(1.0 - d_diff)
            + torch.sigmoid(2.0 - d_diff)
            + torch.sigmoid(4.0 - d_diff)
        )  # [N, L, L] or [N, L, L*14]

        n_pair = pair_mask.sum((-1, -2)).clamp(min=1)  # scalar
        lddt = (lddt_score * pair_mask[None, ...]).sum((-1, -2)) / n_pair  # [N,]

        lddt_loss = 1.0 - lddt  # [N,]
        return lddt_loss
