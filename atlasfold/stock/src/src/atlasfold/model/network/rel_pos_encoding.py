import einops
import numpy as np
import torch
import torch.nn as nn

from atlasfold.common import residue_constants
from atlasfold.model.network.misc import LocalAttentionIndex


class RelativePositionEncoding(torch.nn.Module):
    def __init__(
        self,
        r_max: int = 32,
        s_max: int = 2,
    ) -> None:
        """Relative position encoding module for pair representation.

        Parameters
        ----------
        r_max: int
            The maximum residue distance for relative position encoding.
        s_max: int
            The maximum chain distance for relative position encoding.
        """
        super().__init__()
        self.r_max: int = r_max
        self.s_max: int = s_max
        self.dim: int = 2 * r_max + 2 * s_max + 5  # = 64 + 4 + 5 = 73

    def forward(
        self,
        feat: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the relative position encoding for the pair representation.

        Parameters
        ----------
        feat: dict[str, torch.Tensor]
            A dictionary containing the following keys:
            - res_idx: torch.Tensor of shape (*, L) containing residue indices.
            - asym_id: torch.Tensor of shape (*, L) containing chain indices (asym_id).
            - entity_id: torch.Tensor of shape (*, L) containing entity indices.
            - sym_id: torch.Tensor of shape (*, L) containing symmetry group indices.

        Returns
        -------
        torch.Tensor
            The relative position encoding tensor of shape (*, L, L, bins).
        """
        r_max, s_max = self.r_max, self.s_max
        res_idx = feat["res_idx"]
        asym_id = feat["asym_id"]
        entity_id = feat["entity_id"]
        sym_id = feat["sym_id"]

        is_same_chain = torch.eq(asym_id[..., :, None], asym_id[..., None, :])
        rel_pos = res_idx[..., :, None] - res_idx[..., None, :]
        rel_pos = torch.clamp(rel_pos + r_max, min=0, max=2 * r_max)
        rel_pos = torch.where(is_same_chain, rel_pos, 2 * r_max + 1)
        a_rel_pos = torch.nn.functional.one_hot(rel_pos, 2 * r_max + 2).float()

        is_same_entity = torch.eq(entity_id[..., :, None], entity_id[..., None, :])
        rel_chain = sym_id[..., :, None] - sym_id[..., None, :]
        rel_chain = torch.clip(rel_chain + s_max, min=0, max=2 * s_max)
        rel_chain = torch.where(is_same_entity, rel_chain, 2 * s_max + 1)
        a_rel_chain = torch.nn.functional.one_hot(rel_chain, 2 * s_max + 2).float()
        a_entity = is_same_entity.float().unsqueeze(-1)

        rel_position_encoding = torch.cat(
            [a_rel_pos, a_entity.float(), a_rel_chain], dim=-1
        )
        return rel_position_encoding


class AtomRelativePositionEncoding(nn.Module):
    def __init__(self, max_r: int = 4) -> None:
        """Initialize the Atom Relative Position Encoding layer."""
        super().__init__()
        self.max_r: int = max_r
        res_pos_dim = 2 * max_r + 2
        atom_pos_dim = 3 + 1
        self.dim = res_pos_dim + atom_pos_dim  # = 14
        self.init_buffers()

    def init_buffers(self) -> None:
        ref_d = np.zeros((21, 14, 14, 4))
        for i, aa in enumerate(residue_constants.restypes):
            aa3 = residue_constants.restype_1to3[aa]
            pos = residue_constants.restype_atom14_positions[aa3]
            natom = residue_constants.num_residue_atoms[aa3]
            offset = pos[:, None, :] - pos[None, :, :]
            dist = np.linalg.norm(offset, axis=-1)
            relpos = np.concatenate([offset, dist[..., None]], axis=-1)
            relpos[natom:] = 0.0
            relpos[:, natom:] = 0.0
            ref_d[i] = relpos
        ref_d = ref_d.reshape(21, 14 * 14 * 4)
        self.register_buffer("atom_rel_pos", torch.from_numpy(ref_d).float())

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        res_idx = batch["res_idx"]  # (B, L)
        asym_id = batch["asym_id"]  # (B, L)
        seq_mask = batch["seq_mask"]  # (B, L)

        local_attn_index = LocalAttentionIndex(
            res_idx, asym_id, seq_mask, window_size=4, max_r=self.max_r
        )

        # Compute the residue-wise relative positional encodings
        # NOTE: >max_r, different chain pairs would be masked out in local-attention.
        res_idx_q, res_idx_k = local_attn_index(res_idx, -1, v_pad=int(1e6))
        asym_id_q, asym_id_k = local_attn_index(asym_id, -1, v_pad=-1)

        # 1. Residue-Level Encoding
        # Shape: (B, W, Lq, Lk, 10)
        pad_r = 2 * self.max_r + 1
        rel_pos = res_idx_q[..., :, None] - res_idx_k[..., None, :]
        is_same_chain = asym_id_q[..., :, None] == asym_id_k[..., None, :]

        a_rel_pos = torch.clamp(rel_pos + self.max_r, 0, pad_r)
        a_rel_pos = torch.nn.functional.one_hot(a_rel_pos, pad_r + 1).float()
        a_rel_pos = a_rel_pos * is_same_chain.unsqueeze(-1)
        res_pos_encoding = a_rel_pos.float()  # (B, W, Lq, Lk, 2*r_max+2)
        res_pos_encoding = einops.repeat(res_pos_encoding, "b w q k d -> b w q 14 k 14 d")

        # 2. Atom-Level Intra-Residue Encoding
        aatype = batch["aatype"]  # (B, L, 21)
        # We only need 'q' since intra-bias is only applied when q == k
        aatype_q, _ = local_attn_index(aatype, -2)  # (B, W, Lq, 21)

        is_same_res = (rel_pos == 0) & is_same_chain  # (B, W, Lq, Lk)
        a_atom_dist = aatype_q @ self.atom_rel_pos  # (B, W, Lq, 14*14*4)
        a_atom_dist = a_atom_dist.unsqueeze(-2) * is_same_res.unsqueeze(-1)
        atom_pos_encoding = a_atom_dist  # (B, W, Lq, Lk, 14*14*4)
        atom_pos_encoding = einops.rearrange(
            atom_pos_encoding, "b w q k (a1 a2 d) -> b w q a1 k a2 d", a1=14, a2=14
        )

        # 3. Combine encodings
        pos_encoding = torch.cat([res_pos_encoding, atom_pos_encoding], dim=-1)
        return pos_encoding
