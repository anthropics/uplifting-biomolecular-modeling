from typing import Any

import torch
import torch.nn.functional as F

from atlasfold.common import residue_constants


def atom14_to_atom37(atom14: torch.Tensor, restype: torch.Tensor) -> torch.Tensor:
    """Convert atom14 representation to atom37 representation.

    Parameters
    ----------
    atom14 : torch.Tensor
        The atom14 representation of shape (*, 14, dim)
    restype : torch.Tensor
        The residue type indices of shape (*)

    Returns
    -------
    torch.Tensor
        The atom37 representation of shape (*, 37, dim)
    """
    gather_indices = torch.as_tensor(
        residue_constants._gather_indices, device=atom14.device
    )
    gather_mask = torch.as_tensor(residue_constants._gather_mask, device=atom14.device)
    atom_map = gather_indices[restype]  # shape: (*, 37)
    atom_mask = gather_mask[restype]  # shape: (*, 37)
    expand_shape = atom_map.shape + (atom14.shape[-1],)
    atom37 = atom14.gather(-2, index=atom_map[..., None].expand(*expand_shape))
    atom37 = atom37.masked_fill(~atom_mask[..., None], 0.0)
    return atom37


class LocalAttentionIndex:
    def __init__(
        self,
        res_idx: torch.Tensor,
        chain_idx: torch.Tensor,
        pad_mask: torch.Tensor,
        window_size: int = 4,
        max_r: int = 4,
    ) -> None:
        """Build indices for window-based local attention from atoms to query windows.

        Parameters
        ----------
        res_idx: int
            The residue indices of shape (*, L).
        chain_idx: int
            The chain indices (= asym_id) of shape (*, L).
        pad_mask: torch.Tensor
            The mask tensor of shape (*, L) indicating valid positions
        window_size: int
            The number of query windows (W) per block.
        max_r: int
            The maximum residue distance for local attention
        """
        assert window_size % 2 == 0, "window_size must be even."
        assert (2 * max_r) % (window_size // 2) == 0, (
            "max_r must be divisible by half the window size."
        )
        self.L: int = res_idx.shape[-1]
        self.device: torch.device = res_idx.device
        self.max_r: int = max_r
        self.Lq: int = window_size  # = 4
        self.Lk: int = window_size + 2 * max_r  # = 12

        assert self.L % self.Lq == 0, (
            f"Length {self.L} must be divisible by window_size {self.Lq}."
        )

        self.W: int = self.L // self.Lq

        # Compute the attention mask for local attention
        res_i_q, res_i_k = self.to_qk(res_idx, -1, v_pad=int(1e6))  # [*, W, Lq/Lk]
        attn_mask = torch.abs(res_i_q[..., :, None] - res_i_k[..., None, :]) <= max_r

        # Mask token pairs that are not in the same chain
        chain_id_q, chain_id_k = self.to_qk(chain_idx, -1, v_pad=-1)  # [*, W, Lq/Lk]
        attn_mask &= chain_id_q[..., :, None] == chain_id_k[..., None, :]

        # Mask out padded positions
        mask_k = self.to_k(pad_mask, -1, v_pad=False)  # [*, W, Lk]
        attn_mask &= mask_k.unsqueeze(-2)  # [*, W, Lq, Lk]

        self.attn_mask: torch.Tensor = attn_mask

    def __call__(
        self, x: torch.Tensor, dim: int, v_pad: Any = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Window-based indexing for local attention.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape [*, L, *]
        dim: int, optional
            Dimension along which to unflatten into query/key windows.

        Returns
        -------
        q: torch.Tensor
            Query tensor of shape [*, W, Lq, *]
        k: torch.Tensor
            Key tensor of shape [*, W, Lk, *]
        """
        return self.to_qk(x, dim, v_pad)

    def to_qk(
        self, x: torch.Tensor, dim: int, v_pad: Any = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-copy windowing for query and key states."""
        return self.to_q(x, dim), self.to_k(x, dim, v_pad)

    def to_q(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Extract query windows."""
        return x.unflatten(dim, sizes=(self.W, self.Lq))

    def to_k(self, x: torch.Tensor, dim: int, v_pad: Any = 0) -> torch.Tensor:
        """Extract key windows."""
        # Pad the input tensor for key extraction
        pad_config = [0] * (2 * x.ndim)
        dim = dim % x.ndim
        target_dim_from_back = x.ndim - 1 - dim
        pad_config[2 * target_dim_from_back] = self.max_r
        pad_config[2 * target_dim_from_back + 1] = self.max_r
        x_padded = F.pad(x, pad_config, value=v_pad)

        # Unfold to get key windows.
        x_k = x_padded.unfold(dim, size=self.Lk, step=self.Lq)
        # Permute to bring the window dimension next to the batch dimension.
        dims_order = list(range(x_k.ndim))
        dims_order.insert(dim + 1, dims_order.pop(-1))
        x_k = x_k.permute(dims_order)

        return x_k
