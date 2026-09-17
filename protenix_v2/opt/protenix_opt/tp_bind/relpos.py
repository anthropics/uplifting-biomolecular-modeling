"""Protenix-v2's relative-position pair features on ROWS: ``RelativePositionEncoding.generate_relp`` (``protenix/model/modules/embedders.py``,
AF3 Alg. 3) evaluated for token rows ``[g0, g1)`` against all ``N`` columns from the per-token index features only — the ONE producer both the
input embedder's pair-init rows and the diffusion conditioning rows import (an ``[N, N, 139]`` tensor never exists under row sharding). The
per-pair statements are the core's ``opt_core.mem.rowpair.trunk.same_rows`` / ``relpos_onehot_rows``; this module holds Protenix's feature
order and bins."""
from typing import Any, Dict

import torch

from opt_core.mem.rowpair import trunk as T

__all__ = ["relp_rows", "relp_dim", "INDEX_KEYS"]

INDEX_KEYS = ("asym_id", "residue_index", "entity_id", "token_index", "sym_id")     # RelativePositionEncoding.input_feature


def relp_dim(r_max: int, s_max: int) -> int:
    """``4 r_max + 2 s_max + 7``: the feature width ``RelativePositionEncoding.linear_no_bias`` expects."""
    return 4 * int(r_max) + 2 * int(s_max) + 7


def relp_rows(feats: Dict[str, Any], g0: int, g1: int, r_max: int, s_max: int, dtype=torch.float32):
    """Rows ``[g0, g1)`` of ``generate_relp``: ``[a_rel_pos (2 r_max + 2) | a_rel_token (2 r_max + 2) | b_same_entity (1) | a_rel_chain (2 s_max + 2)]``
    -> ``[*, g1 - g0, N, relp_dim]`` in ``dtype`` (stock: ``.float()``). ``a_rel_pos``: residue offset clipped to ``r_max`` within a chain, else the
    bin ``2 r_max + 1``; ``a_rel_token``: token offset clipped to ``r_max`` within a chain AND residue, else ``2 r_max + 1``; ``a_rel_chain``:
    ``sym_id`` offset clipped to ``s_max`` within an entity, else ``2 s_max + 1`` — integer-exact with the dense statement."""
    asym, res, ent, tok, sym = (feats[k] for k in INDEX_KEYS)
    same_chain, same_res, same_ent = T.same_rows(asym, g0, g1), T.same_rows(res, g0, g1), T.same_rows(ent, g0, g1)
    a_pos = T.relpos_onehot_rows(res, g0, g1, r_max, condition=same_chain, dtype=dtype)
    a_tok = T.relpos_onehot_rows(tok, g0, g1, r_max, condition=same_chain & same_res, dtype=dtype)
    a_chain = T.relpos_onehot_rows(sym, g0, g1, s_max, condition=same_ent, dtype=dtype)
    return torch.cat([a_pos, a_tok, same_ent.to(dtype).unsqueeze(-1), a_chain], dim=-1)
