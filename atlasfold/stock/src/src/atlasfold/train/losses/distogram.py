import torch

from atlasfold.utils.torch_utils import index_select_dim


class DistogramLoss(torch.nn.Module):
    def forward(
        self,
        logits: torch.Tensor,
        boundaries: torch.Tensor,
        x_gt: torch.Tensor,
        mask_gt: torch.Tensor,
        cbeta_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the  distogram loss.

        Parameters
        ----------
        logits : torch.Tensor
            Tensor of shape (B, L, L, num_bins) containing distogram logits.
        boundaries : torch.Tensor
            Tensor of shape (num_bins - 1,) containing the boundaries of the bins for
            the distogram.
        x_gt : torch.Tensor
            Tensor of shape (B, L, 14, 3) containing ground truth coordinates.
        mask_gt : torch.Tensor
            Tensor of shape (B, L, 14) containing boolean mask for valid coordinates.
        cbeta_idx : torch.Tensor
            Tensor of shape (B, L) containing the index of the C-beta atom
            for each residue (or C-alpha for glycine).

        Returns
        -------
        distogram_loss : torch.Tensor
            The computed distogram loss of shape (B,).
        """

        # Gather the C-beta atom coordinates and mask (CB, or CA for Gly)
        x_gt_beta = index_select_dim(x_gt, dim=-2, index=cbeta_idx)  # [B, L, 3]
        mask_beta = index_select_dim(mask_gt, dim=-1, index=cbeta_idx)  # [B, L]

        # NOTE: This doesn't require backprop, so use norm().
        d = (x_gt_beta[..., None, :, :] - x_gt_beta[..., :, None, :]).norm(dim=-1)
        target_distogram = (d[..., None] > boundaries).sum(-1).long()

        # Compute the distogram loss
        B, L, _, _ = x_gt.shape
        distogram_loss = torch.nn.functional.cross_entropy(
            logits.view(B * L * L, -1),
            target_distogram.view(B * L * L),
            reduction="none",
        ).view(B, L, L)

        # Mask out invalid distogram
        pair_mask = mask_beta[..., :, None] & mask_beta[..., None, :]  # [B, L, L]
        pair_mask.diagonal(dim1=-2, dim2=-1).fill_(False)  # Mask out self-distances

        # Compute mean loss
        w = pair_mask.float()
        sum_loss = (distogram_loss * w).sum((-1, -2))  # [B,]
        n_valid = w.sum((-1, -2)).clamp(1)  # [B,]
        return sum_loss / n_valid  # [B,]
