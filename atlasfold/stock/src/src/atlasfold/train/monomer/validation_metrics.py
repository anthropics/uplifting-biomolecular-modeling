"""Model validation metrics for structure prediction tasks."""

import numpy as np
import torch

from atlasfold.train.monomer.structure_alignment import get_aligned_gt_structure
from atlasfold.utils.geometry.metrics import (
    compute_lddt_ca,
    compute_lddt_fullatom,
    compute_rmsd_atom14,
)


def compute_validation_metric(
    x_pred: torch.Tensor,
    batch: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
    rank_idx: int | None = None,
) -> dict[str, float]:
    """Get structure prediction metrics.

    Parameters
    ----------
    x_pred : torch.Tensor
        Predicted coordinates from the model. Shape (N, L, 14, 3).
    batch : dict[str, torch.Tensor]
        Input features.
    label : dict[str, torch.Tensor]
        Ground truth labels.
    rank_idx : int | None
        Top-plddt ranking index for each sample.

    Returns
    -------
    metrics: dict
        The computed metrics
    """
    N, L, _, _ = x_pred.shape
    aatype = batch["aatype_int"]  # [L,]
    x_gt = label["coordinates"]  # [L, 14, 3]
    mask = label["resolved_mask"]  # [L, 14]

    all_metrics = {}

    rmsds = []
    lddts = []
    lddt_cas = []
    for n_i in range(N):
        x_pred_i = x_pred[n_i]  # [L, 14, 3]

        # Align the predicted structure
        x_gt_i, m_i = get_aligned_gt_structure(x_gt, x_pred_i, aatype, mask)

        # Compute metrics for this prediction
        rmsds.append(compute_rmsd_atom14(x_pred_i, x_gt_i, m_i).item())
        lddts.append(compute_lddt_fullatom(x_pred_i, x_gt_i, m_i).item())
        lddt_cas.append(compute_lddt_ca(x_pred_i, x_gt_i, m_i).item())

    # Aggregate metrics across N predictions
    all_metrics["avg/rmsd"] = np.mean(rmsds)
    all_metrics["avg/lddt"] = np.mean(lddts)
    all_metrics["avg/lddt-ca"] = np.mean(lddt_cas)
    all_metrics["top/rmsd"] = min(rmsds)
    all_metrics["top/lddt"] = max(lddts)
    all_metrics["top/lddt-ca"] = max(lddt_cas)

    if rank_idx is not None:
        all_metrics["rank/rmsd"] = rmsds[rank_idx]
        all_metrics["rank/lddt"] = lddts[rank_idx]
        all_metrics["rank/lddt-ca"] = lddt_cas[rank_idx]

    all_metrics = {k: float(v) for k, v in all_metrics.items()}

    return all_metrics
