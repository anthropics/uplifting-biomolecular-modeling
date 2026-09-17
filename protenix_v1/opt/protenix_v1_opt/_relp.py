"""The relative-position one-hot statements of protenix 1.1.0 (`RelativePositionEncoding.generate_relp`, embedders.py:150-201) for a block
of query rows [i, j), written into a caller-provided [.., j-i, N, 139] fp32 buffer. ONE statement set, two consumers: the P=1 memory line's
`relp_lean` lever (big.py: the whole plane filled block by block into its output) and the n_gpu>1 adapter (tp.RelpRows: a rank's row block,
the plane never materialised)."""
from __future__ import annotations


def relp_width(rpe) -> int:
    """Channel count of the one-hot plane: 2*(r_max+1) residue offsets + 2*(r_max+1) token offsets + 1 same-entity + 2*(s_max+1) chain offsets."""
    n_pos, n_chain = 2 * (rpe.r_max + 1), 2 * (rpe.s_max + 1)
    return n_pos + n_pos + 1 + n_chain


def relp_rows_into(rpe, input_feature_dict, i: int, j: int, out) -> None:
    """embedders.py:150-201 for query rows [i, j): `out[..., :, :, :]` ([.., j-i, N, relp_width]) receives the four one-hot groups."""
    import torch
    import torch.nn.functional as F
    asym_id = input_feature_dict["asym_id"]; residue_index = input_feature_dict["residue_index"]
    entity_id = input_feature_dict["entity_id"]; token_index = input_feature_dict["token_index"]; sym_id = input_feature_dict["sym_id"]
    n_pos, n_chain = 2 * (rpe.r_max + 1), 2 * (rpe.s_max + 1)
    with torch.no_grad():
        b_same_chain = (asym_id[..., i:j, None] == asym_id[..., None, :]).long()
        b_same_residue = (residue_index[..., i:j, None] == residue_index[..., None, :]).long()
        b_same_entity = (entity_id[..., i:j, None] == entity_id[..., None, :]).long()
        d_residue = torch.clip(input=residue_index[..., i:j, None] - residue_index[..., None, :] + rpe.r_max, min=0, max=2 * rpe.r_max) * b_same_chain + (1 - b_same_chain) * (2 * rpe.r_max + 1)
        out[..., :, :, 0:n_pos] = F.one_hot(d_residue, n_pos)
        d_token = torch.clip(input=token_index[..., i:j, None] - token_index[..., None, :] + rpe.r_max, min=0, max=2 * rpe.r_max) * b_same_chain * b_same_residue + (1 - b_same_chain * b_same_residue) * (2 * rpe.r_max + 1)
        out[..., :, :, n_pos:2 * n_pos] = F.one_hot(d_token, n_pos)
        out[..., :, :, 2 * n_pos] = b_same_entity
        d_chain = torch.clip(input=sym_id[..., i:j, None] - sym_id[..., None, :] + rpe.s_max, min=0, max=2 * rpe.s_max) * b_same_entity + (1 - b_same_entity) * (2 * rpe.s_max + 1)
        out[..., :, :, 2 * n_pos + 1:] = F.one_hot(d_chain, n_chain)
