"""The structural-token stage on rows (opendde_opt.tp_struct) vs the DENSE engine statements, CPU, gloo, P in {2, 3} through the kit launcher
(``opt_core.mem.rowpair.launch.run_sharded``), ragged layouts (N_st and N not divisible by the grid): a miniature StructuralTokenExpander with
OpenDDE 1.1.1's ATTRIBUTE NAMES and DENSE statements (structural_tokens.py: the single expansions, the pair context, the per-row pair features,
the role-pair projection in ``full`` and ``factorized`` modes, the pair init bias, the attention bias, the row-chunked pair activations,
``forward``) + a 4-block ``PairformerStack`` refiner WITH the single track (``_tp_stub_engine``) + ``OpenDDE.expand_to_structural_tokens``
verbatim is the reference; the same weights run through tp_struct's row path in every rank. Checked per rank (verdicts AND-ed onto rank 0):
the expansion rows and the extra-bias rows bit for bit, the refined ``z_struct`` rows / ``s_struct`` vs dense (fp32 max|diff| <= 1e-5;
torch.equal REPORTED), the structural feature dict (keys, values, rows marker, residue-only keys dropped), ``fetch_rows_ring`` == the dense
rows for arbitrary requests (rows of every rank, an empty request), the census words, nothing pair-TRACK shaped (``[N_st, N_st, >=c]``,
``[N, N, >=c]``, a whole ``[N_st, N_st]`` bias, or two N_st / N dims in any other arrangement) while the stage runs — the triangle
attention's replicated bias ``[H, N_st, N_st]`` and its per-row-batch logits ``[rows x H, N_st, N_st]`` (rows <= the attention chunk) are the
trunk binding's own bounded transients: census'd by shape and bounded, not exempted silently — and the refusals by name (``P = 1`` layout, a ``z_res`` that is not a RowShard,
``parent_residue_idx`` out of range). Tiny dims (N=66 residues -> N_st=105 structural tokens, c_z=8, c_s=6): the point is the statement
algebra, the seams and the census."""
from __future__ import annotations

import os
import copy
import sys
import types

import pytest

torch = pytest.importorskip("torch", reason="the seam test runs the statements: torch (CPU) required")
pytest.importorskip("opt_core.mem.rowpair.pairstack", reason="the pinned core lacks the rowpair pair-stack driver")

from torch import nn  # noqa: E402

from opendde_opt.tests._tp_stub_engine import C_S, C_SIN, C_Z, LN, Lin, PairformerStack, install_modules  # noqa: E402

N_RES = 66                                   # residues (2 chains: 40 + 26); grid P=2 -> rows (0,64),(64,66); P=3 -> (0,32),(32,64),(64,66)
N_REF_BLOCKS = 4                             # the pin's structural refiner: 4 Pairformer blocks with the single track
EXP_CHUNK = 20                               # the expander's pair_chunk_size (row blocks of the expansion; several per rank, last partial)
TOL = 1e-5
STRUCTURAL_TOKEN_ROLES = {"atom": 0, "protein_bb": 1, "protein_sc": 2, "dna_bb": 3, "dna_base": 4, "rna_bb": 5, "rna_base": 6}   # data/tokenizer.py
N_ROLES = 7
RESIDUE_ONLY = ("msa", "has_deletion", "msa_mask", "token_bonds", "template_foo")


# ============================================================================================================ the stub expander (1.1.1 statements)
class StructuralTokenExpander(nn.Module):
    """== opendde.model.modules.structural_tokens.StructuralTokenExpander (1.1.1): the attribute names and the DENSE / row-chunked statements
    the row path binds (``_gather_parent_single``, ``_build_structural_pair_context``, ``_build_structural_pair_features_for_rows``,
    ``_pair_project_by_role`` full / factorized, ``_make_pair_init_bias``, ``_make_attention_bias``, ``_gather_parent_pair_rows``,
    ``_make_structural_pair_activations_chunked``, ``forward``). Weights are random (the stock's zero init would make the deltas vanish)."""

    def __init__(self, c_s, c_z, c_s_inputs, n_roles=N_ROLES, pair_projection_mode="full", pair_chunk_size=None, gen=None):
        super().__init__()
        self.c_s, self.c_z, self.c_s_inputs, self.n_roles = c_s, c_z, c_s_inputs, n_roles
        self.pair_chunk_size = None if pair_chunk_size is None else int(pair_chunk_size)
        self.pair_projection_mode = pair_projection_mode
        self.single_split_mlp = nn.Sequential(LN(c_s), Lin(c_s, 2 * c_s), nn.SiLU(), Lin(2 * c_s, c_s))
        self.single_input_role_embedding = nn.Embedding(n_roles, c_s_inputs)
        self.single_role_embedding = nn.Embedding(n_roles, c_s)
        if pair_projection_mode == "full":
            self.pair_block_proj = nn.ModuleList([Lin(c_z, c_z) for _ in range(n_roles * n_roles)])
        elif pair_projection_mode == "factorized":
            self.shared_pair_proj = Lin(c_z, c_z)
            self.role_pair_gate = nn.Embedding(8, c_z)
            self.role_pair_delta_bias = nn.Embedding(8, c_z)
        self.same_parent_embedding, self.same_residue_twin_embedding = nn.Embedding(2, c_z), nn.Embedding(2, c_z)
        self.prev_bb_chain_embedding, self.next_bb_chain_embedding = nn.Embedding(2, c_z), nn.Embedding(2, c_z)
        self.role_pair_type_embedding = nn.Embedding(8, c_z)
        self.attn_bias_same_parent, self.attn_bias_same_residue_twin = nn.Parameter(torch.zeros(())), nn.Parameter(torch.zeros(()))
        self.attn_bias_prev_bb_chain, self.attn_bias_next_bb_chain = nn.Parameter(torch.zeros(())), nn.Parameter(torch.zeros(()))
        self.attn_bias_role_pair_type = nn.Parameter(torch.zeros(8))
        self.backbone_role_ids = (STRUCTURAL_TOKEN_ROLES["protein_bb"], STRUCTURAL_TOKEN_ROLES["dna_bb"], STRUCTURAL_TOKEN_ROLES["rna_bb"])
        self.sidechain_role_id = STRUCTURAL_TOKEN_ROLES["protein_sc"]
        self.base_role_ids = (STRUCTURAL_TOKEN_ROLES["dna_base"], STRUCTURAL_TOKEN_ROLES["rna_base"])
        with torch.no_grad():                                                     # deterministic, non-trivial weights
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=gen) * 0.3)

    # ---- structural_tokens.py:196-211
    @staticmethod
    def _gather_parent_single(x, parent):
        return x.index_select(dim=-2, index=parent)

    @staticmethod
    def _gather_parent_pair(z, parent):
        return z.index_select(dim=-3, index=parent).index_select(dim=-2, index=parent)

    @staticmethod
    def _gather_parent_pair_rows(z, parent, row_index):
        row_parent = parent.index_select(dim=0, index=row_index)
        return z.index_select(dim=-3, index=row_parent).index_select(dim=-2, index=parent)

    # ---- :289-319 / :353-379
    def _pair_project_by_role_full_chunk(self, z, role, row_index):
        n_struct = role.shape[-1]
        chunk_len = row_index.numel()
        batch_shape = z.shape[:-3]
        flat_z = z.reshape(*batch_shape, chunk_len * n_struct, self.c_z)
        flat_delta = torch.zeros_like(flat_z)
        row_role = role.index_select(dim=0, index=row_index)
        role_i = row_role[:, None].expand(chunk_len, n_struct).reshape(-1)
        role_j = role[None, :].expand(chunk_len, n_struct).reshape(-1)
        dummy_input = flat_z[..., :1, :]
        dummy_use = flat_z.new_zeros(())
        for role_i_value in range(self.n_roles):
            for role_j_value in range(self.n_roles):
                flat_mask = (role_i == role_i_value) & (role_j == role_j_value)
                projection = self.pair_block_proj[role_i_value * self.n_roles + role_j_value]
                if torch.any(flat_mask):
                    flat_delta[..., flat_mask, :] = projection(flat_z[..., flat_mask, :])
                else:
                    dummy_use = dummy_use + projection(dummy_input).sum() * 0.0
        return flat_delta.reshape(*batch_shape, chunk_len, n_struct, self.c_z) + dummy_use

    def _pair_project_by_role(self, z, role, pair_features, row_index=None, col_index=None):
        if self.pair_projection_mode == "none":
            return None
        if self.pair_projection_mode == "full":
            assert col_index is None
            if row_index is None:
                row_index = torch.arange(role.shape[-1], device=role.device)      # == _pair_project_by_role_full on all rows
            return self._pair_project_by_role_full_chunk(z, role, row_index)
        role_pair_type = pair_features["role_pair_type"]
        base_delta = self.shared_pair_proj(z)
        gate = self.role_pair_gate(role_pair_type).to(dtype=z.dtype)
        bias = self.role_pair_delta_bias(role_pair_type).to(dtype=z.dtype)
        return base_delta * gate + bias

    # ---- :430-544
    def _build_structural_pair_context(self, input_feature_dict, role, parent):
        n_struct = role.shape[-1]
        residue_index = input_feature_dict["residue_index"].index_select(dim=-1, index=parent)
        asym_id = input_feature_dict["asym_id"].index_select(dim=-1, index=parent)
        polymer_type = input_feature_dict.get("structural_polymer_type")
        if polymer_type is None:
            polymer_type = torch.zeros_like(role)
        polymer_type = polymer_type.long()
        is_backbone = (role == self.backbone_role_ids[0]) | (role == self.backbone_role_ids[1]) | (role == self.backbone_role_ids[2])
        is_sidechain = role == self.sidechain_role_id
        is_base = (role == self.base_role_ids[0]) | (role == self.base_role_ids[1])
        prev_parent = input_feature_dict.get("prev_parent_residue_idx")
        next_parent = input_feature_dict.get("next_parent_residue_idx")
        if prev_parent is None:
            prev_parent = parent.new_full((n_struct,), -1)
        if next_parent is None:
            next_parent = parent.new_full((n_struct,), -1)
        return {"parent": parent, "role": role, "residue_index": residue_index, "asym_id": asym_id, "polymer_type": polymer_type,
                "is_backbone": is_backbone, "is_sidechain": is_sidechain, "is_base": is_base, "prev_parent": prev_parent, "next_parent": next_parent}

    def _build_structural_pair_features_for_rows(self, context, row_index):
        parent, residue_index, asym_id, polymer_type = context["parent"], context["residue_index"], context["asym_id"], context["polymer_type"]
        is_backbone, is_sidechain, is_base, prev_parent, next_parent = (context[k] for k in ("is_backbone", "is_sidechain", "is_base", "prev_parent", "next_parent"))
        n_struct = parent.shape[-1]
        row_parent = parent.index_select(dim=0, index=row_index)
        row_asym_id = asym_id.index_select(dim=0, index=row_index)
        row_polymer_type = polymer_type.index_select(dim=0, index=row_index)
        row_is_backbone = is_backbone.index_select(dim=0, index=row_index)
        row_is_sidechain = is_sidechain.index_select(dim=0, index=row_index)
        row_is_base = is_base.index_select(dim=0, index=row_index)
        row_prev_parent = prev_parent.index_select(dim=0, index=row_index)
        row_next_parent = next_parent.index_select(dim=0, index=row_index)
        same_parent_residue = row_parent[:, None] == parent[None, :]
        same_chain = row_asym_id[:, None] == asym_id[None, :]
        same_polymer_type = (row_polymer_type[:, None] == polymer_type[None, :]) & (row_polymer_type[:, None] > 0)
        same_residue_twin = same_parent_residue & ((row_is_backbone[:, None] & (is_sidechain[None, :] | is_base[None, :]))
                                                   | (is_backbone[None, :] & (row_is_sidechain[:, None] | row_is_base[:, None])))
        prev_bb_chain = row_is_backbone[:, None] & is_backbone[None, :] & same_chain & (row_prev_parent[:, None] == parent[None, :])
        next_bb_chain = row_is_backbone[:, None] & is_backbone[None, :] & same_chain & (row_next_parent[:, None] == parent[None, :])
        chunk_len = row_index.numel()
        role_pair_type = torch.full((chunk_len, n_struct), 7, dtype=torch.long, device=parent.device)
        role_pair_type[row_is_backbone[:, None] & is_backbone[None, :]] = 0
        role_pair_type[row_is_backbone[:, None] & is_sidechain[None, :]] = 1
        role_pair_type[row_is_sidechain[:, None] & is_backbone[None, :]] = 2
        role_pair_type[row_is_sidechain[:, None] & is_sidechain[None, :]] = 3
        role_pair_type[row_is_backbone[:, None] & is_base[None, :]] = 4
        role_pair_type[row_is_base[:, None] & is_backbone[None, :]] = 5
        role_pair_type[row_is_base[:, None] & is_base[None, :]] = 6
        return {"same_parent_residue": same_parent_residue, "same_residue_twin": same_residue_twin, "prev_bb_chain": prev_bb_chain,
                "next_bb_chain": next_bb_chain, "role_pair_type": role_pair_type, "same_chain": same_chain, "same_polymer_type": same_polymer_type,
                "residue_index": residue_index}

    def build_structural_pair_features(self, input_feature_dict, role, parent):
        context = self._build_structural_pair_context(input_feature_dict=input_feature_dict, role=role, parent=parent)
        return self._build_structural_pair_features_for_rows(context=context, row_index=torch.arange(parent.shape[-1], device=parent.device))

    # ---- :646-678 / :798-814
    def _make_pair_init_bias(self, pair_features, dtype):
        pair_bias = self.same_parent_embedding(pair_features["same_parent_residue"].long()).to(dtype=dtype)
        pair_bias = pair_bias + self.same_residue_twin_embedding(pair_features["same_residue_twin"].long()).to(dtype=dtype)
        pair_bias = pair_bias + self.prev_bb_chain_embedding(pair_features["prev_bb_chain"].long()).to(dtype=dtype)
        pair_bias = pair_bias + self.next_bb_chain_embedding(pair_features["next_bb_chain"].long()).to(dtype=dtype)
        return pair_bias + self.role_pair_type_embedding(pair_features["role_pair_type"]).to(dtype=dtype)

    def _make_attention_bias(self, pair_features, dtype):
        role_pair_bias = self.attn_bias_role_pair_type[pair_features["role_pair_type"]].to(dtype=dtype)
        return (self.attn_bias_same_parent.to(dtype=dtype) * pair_features["same_parent_residue"].to(dtype)
                + self.attn_bias_same_residue_twin.to(dtype=dtype) * pair_features["same_residue_twin"].to(dtype)
                + self.attn_bias_prev_bb_chain.to(dtype=dtype) * pair_features["prev_bb_chain"].to(dtype)
                + self.attn_bias_next_bb_chain.to(dtype=dtype) * pair_features["next_bb_chain"].to(dtype) + role_pair_bias)

    # ---- :726-796 (inference branch)
    def _make_structural_pair_activations_chunked(self, input_feature_dict, z_res, role, parent):
        n_struct = role.shape[-1]
        chunk_size = min(self.pair_chunk_size or n_struct, n_struct)
        context = self._build_structural_pair_context(input_feature_dict=input_feature_dict, role=role, parent=parent)
        z_out = attn_bias_out = None
        for start in range(0, n_struct, chunk_size):
            end = min(start + chunk_size, n_struct)
            row_index = torch.arange(start, end, device=parent.device)
            pair_features = self._build_structural_pair_features_for_rows(context=context, row_index=row_index)
            z_chunk = self._gather_parent_pair_rows(z=z_res, parent=parent, row_index=row_index)
            delta = self._pair_project_by_role(z=z_chunk, role=role, pair_features=pair_features, row_index=row_index)
            if delta is not None:
                z_chunk = z_chunk + delta
            z_chunk = z_chunk + self._make_pair_init_bias(pair_features, dtype=z_chunk.dtype)
            attn_bias_chunk = self._make_attention_bias(pair_features, dtype=z_chunk.dtype)
            if z_out is None:
                z_out = z_chunk.new_empty((*z_chunk.shape[:-3], n_struct, *z_chunk.shape[-2:]))
                attn_bias_out = attn_bias_chunk.new_empty((n_struct, *attn_bias_chunk.shape[1:]))
            z_out[..., start:end, :, :].copy_(z_chunk)
            attn_bias_out[start:end].copy_(attn_bias_chunk)
        return z_out, {"structural_pair_attn_bias": attn_bias_out}

    # ---- :842-883
    def forward(self, input_feature_dict, s_inputs_res, s_res, z_res):
        parent = input_feature_dict["parent_residue_idx"].long()
        role = input_feature_dict["subtoken_role_id"].long()
        s_inputs_struct = self._gather_parent_single(s_inputs_res, parent) + self.single_input_role_embedding(role).to(dtype=s_inputs_res.dtype)
        s_parent = self._gather_parent_single(s_res, parent)
        s_struct = s_parent + self.single_split_mlp(s_parent) + self.single_role_embedding(role).to(dtype=s_parent.dtype)
        if self.pair_chunk_size is None:
            pair_features = self.build_structural_pair_features(input_feature_dict=input_feature_dict, role=role, parent=parent)
            z_parent = self._gather_parent_pair(z_res, parent)
            delta = self._pair_project_by_role(z_parent, role, pair_features)
            z_struct = z_parent if delta is None else z_parent + delta
            z_struct = z_struct + self._make_pair_init_bias(pair_features, dtype=z_parent.dtype)
            pair_features = {"structural_pair_attn_bias": self._make_attention_bias(pair_features, dtype=z_parent.dtype)}   # the chunked statement's returned set
        else:
            z_struct, pair_features = self._make_structural_pair_activations_chunked(input_feature_dict=input_feature_dict, z_res=z_res, role=role, parent=parent)
        return s_inputs_struct, s_struct, z_struct, pair_features


class LazyRelp(object):
    """== embedders.LazyRelativePositionEncodingFeatures: the O(N_st) index vectors; rows are materialised by the consumers (never here)."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


class RelPE(nn.Module):
    """== RelativePositionEncoding.generate_relp (embedders.py:237-258), the lazy branch (the loop passes lazy_relp=True)."""
    r_max, s_max = 32, 2

    def generate_relp(self, input_feature_dict, lazy=False):
        assert lazy, "the structural stage generates the relp lazy (the loop's lazy_relp=True; the [N_st, N_st, 139] one-hot is never built)"
        with torch.no_grad():
            input_feature_dict["relp"] = LazyRelp(asym_id=input_feature_dict["asym_id"], residue_index=input_feature_dict["residue_index"],
                                                  entity_id=input_feature_dict["entity_id"], token_index=input_feature_dict["token_index"],
                                                  sym_id=input_feature_dict["sym_id"], r_max=self.r_max, s_max=self.s_max)
            return input_feature_dict


class Model(nn.Module):
    """The structural stage's model surface: ``structural_token_expander``, ``structural_token_refiner`` (a 4-block PairformerStack WITH the single
    track, as the pin's), ``relative_position_encoding``, ``configs.triangle_attention``, the enable flags, ``drop_residue_only_features_...``,
    and ``expand_to_structural_tokens`` = the pin's statements verbatim (opendde.py:422-538, single device) = the DENSE reference."""

    def __init__(self, mode: str, gen):
        super().__init__()
        self.enable_structural_token_expansion, self.enable_structural_token_refiner, self.pair_output_space = True, True, "structural"
        self.structural_token_expander = StructuralTokenExpander(C_S, C_Z, C_SIN, pair_projection_mode=mode, pair_chunk_size=EXP_CHUNK, gen=gen)
        self.structural_token_refiner = PairformerStack(N_REF_BLOCKS, C_Z, C_S)
        self.relative_position_encoding = RelPE()
        self.configs = types.SimpleNamespace(triangle_attention="torch", triangle_multiplicative="torch")

    def _maybe_foldcp_mesh(self):
        return None

    @staticmethod
    def drop_residue_only_features_for_structural_branch(input_feature_dict):
        residue_only_keys = {"msa", "has_deletion", "deletion_value", "msa_mask", "profile", "deletion_mean", "token_bonds"}
        for key in list(input_feature_dict.keys()):
            if key in residue_only_keys or key.startswith("template_"):
                input_feature_dict.pop(key, None)

    def expand_to_structural_tokens(self, input_feature_dict, s_inputs, s, z, inplace_safe=False, chunk_size=None, lazy_relp=True):
        required = ["parent_residue_idx", "subtoken_role_id", "structural_token_index", "atom_to_structural_token_idx", "atom_to_structural_tokatom_idx",
                    "structural_distogram_rep_atom_mask", "structural_pae_rep_atom_mask", "structural_has_frame", "structural_frame_atom_index"]
        missing = [key for key in required if key not in input_feature_dict]
        if missing:
            raise KeyError("missing structural feature(s): " + ", ".join(missing))
        structural_feature_dict = dict(input_feature_dict)
        for residue_feature in ["token_index", "asym_id", "residue_index", "entity_id", "sym_id", "atom_to_token_idx", "atom_to_tokatom_idx",
                                "has_frame", "frame_atom_index", "pae_rep_atom_mask", "distogram_rep_atom_mask"]:
            structural_feature_dict[f"residue_level_{residue_feature}"] = input_feature_dict[residue_feature]
        parent = input_feature_dict["parent_residue_idx"].long()
        s_inputs, s, z, structural_pair_features = self.structural_token_expander(input_feature_dict=input_feature_dict, s_inputs_res=s_inputs, s_res=s, z_res=z)
        structural_feature_dict["token_index"] = input_feature_dict["structural_token_index"].long()
        structural_feature_dict["atom_to_token_idx"] = input_feature_dict["atom_to_structural_token_idx"].long()
        structural_feature_dict["atom_to_tokatom_idx"] = input_feature_dict["atom_to_structural_tokatom_idx"].long()
        for token_feature in ["asym_id", "residue_index", "entity_id", "sym_id"]:
            structural_feature_dict[token_feature] = input_feature_dict[token_feature].index_select(dim=-1, index=parent)
        structural_feature_dict["has_frame"] = input_feature_dict["structural_has_frame"]
        structural_feature_dict["frame_atom_index"] = input_feature_dict["structural_frame_atom_index"]
        structural_feature_dict["pae_rep_atom_mask"] = input_feature_dict["structural_pae_rep_atom_mask"].long()
        structural_feature_dict["distogram_rep_atom_mask"] = input_feature_dict["structural_distogram_rep_atom_mask"].long()
        for feature_name, feature_value in structural_pair_features.items():
            structural_feature_dict[feature_name] = feature_value
        structural_feature_dict = self.relative_position_encoding.generate_relp(structural_feature_dict, lazy=lazy_relp)
        if self.enable_structural_token_refiner:
            s, z = self.structural_token_refiner(s=s, z=z, pair_mask=None, triangle_multiplicative=self.configs.triangle_multiplicative,
                                                 triangle_attention=self.configs.triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size,
                                                 extra_attn_bias=structural_feature_dict.get("structural_pair_attn_bias", None))
        self.drop_residue_only_features_for_structural_branch(structural_feature_dict)
        return structural_feature_dict, s_inputs, s, z


# ============================================================================================================ deterministic inputs
def _tokenize():
    """Two chains (40 + 26 residues); per residue (by index mod 5): a single 'atom' token, protein bb+sc, protein bb+sc, protein bb only (GLY-like),
    RNA bb+base -> N_st = 105 structural tokens (8 per 5 residues + 1). Returns per-structural-token vectors + the residue features."""
    asym = torch.tensor([0] * 40 + [1] * 26)
    residue_index = torch.cat([torch.arange(40), torch.arange(26)])
    parent, role, poly, prev_p, next_p = [], [], [], [], []
    for r in range(N_RES):
        k = r % 5
        toks = {0: [("atom", 0)], 1: [("protein_bb", 1), ("protein_sc", 1)], 2: [("protein_bb", 1), ("protein_sc", 1)], 3: [("protein_bb", 1)],
                4: [("rna_bb", 3), ("rna_base", 3)]}[k]
        for name, pt in toks:
            parent.append(r)
            role.append(STRUCTURAL_TOKEN_ROLES[name])
            poly.append(pt)
            same_prev = r - 1 >= 0 and int(asym[r - 1]) == int(asym[r])
            same_next = r + 1 < N_RES and int(asym[r + 1]) == int(asym[r])
            prev_p.append(r - 1 if (name.endswith("_bb") and same_prev) else -1)
            next_p.append(r + 1 if (name.endswith("_bb") and same_next) else -1)
    t = lambda x: torch.tensor(x, dtype=torch.long)  # noqa: E731
    return asym, residue_index, t(parent), t(role), t(poly), t(prev_p), t(next_p)


def build(seed: int, mode: str):
    """``(model, feats, s_inputs, s, z)``: deterministic weights and inputs; N=66 residues, N_st=105 structural tokens, 2 atoms per structural token."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = Model(mode, g).eval()
    asym, residue_index, parent, role, poly, prev_p, next_p = _tokenize()
    n_st = int(parent.numel())
    n_atom = 2 * n_st
    a2st = torch.arange(n_st).repeat_interleave(2)
    feats = {
        "asym_id": asym, "residue_index": residue_index, "entity_id": asym.clone(), "sym_id": torch.zeros(N_RES, dtype=torch.long), "token_index": torch.arange(N_RES),
        "parent_residue_idx": parent, "subtoken_role_id": role, "structural_polymer_type": poly, "prev_parent_residue_idx": prev_p, "next_parent_residue_idx": next_p,
        "structural_token_index": torch.arange(n_st), "atom_to_structural_token_idx": a2st, "atom_to_structural_tokatom_idx": torch.arange(n_atom) % 2,
        "atom_to_token_idx": parent[a2st], "atom_to_tokatom_idx": torch.arange(n_atom) % 4,
        "structural_distogram_rep_atom_mask": torch.ones(n_atom), "structural_pae_rep_atom_mask": torch.ones(n_atom),
        "structural_has_frame": torch.ones(n_st, dtype=torch.long), "structural_frame_atom_index": torch.zeros(n_st, 3, dtype=torch.long),
        "has_frame": torch.ones(N_RES, dtype=torch.long), "frame_atom_index": torch.zeros(N_RES, 3, dtype=torch.long),
        "pae_rep_atom_mask": torch.ones(n_atom), "distogram_rep_atom_mask": torch.ones(n_atom),
        "msa": torch.zeros(3, N_RES), "has_deletion": torch.zeros(3, N_RES), "msa_mask": torch.ones(3, N_RES), "token_bonds": torch.zeros(N_RES, N_RES), "template_foo": torch.zeros(2),
    }
    s_inputs = torch.randn((N_RES, C_SIN), generator=g)
    s = torch.randn((N_RES, C_S), generator=g)
    z = torch.randn((N_RES, N_RES, C_Z), generator=g)
    return model, feats, s_inputs, s, z


# ============================================================================================================ the shape scan
def _pair_guard(n_st: int, n_res: int, c_z: int, heads: int):
    """A TorchDispatchMode classifying EVERY op output while the row stage runs, for n in {N_st, N}. OFFENDERS (must be empty): a 2-D ``[n, n]``
    (a whole extra bias / mask), a 3-D ``[n, n, c >= c_z]`` (a whole pair TRACK), and any other output carrying two dims == n that is not
    one of the two named triangle-attention transients of the trunk binding — (a) the triangle bias ``[n, n, H]`` / ``[H, n, n]`` /
    ``[1, H, n, n]`` (``H < c_z``; replicated by design: ``triatt_bias[N,N,H]`` in tp.REPLICATED_BY_DESIGN) and (b) ONE ROW BATCH's attention
    logits / weights ``[..., n, n]`` whose leading dims multiply to at most ``rows x H`` (the dense statement holds ``[N, H, N, N]``; the row
    statement holds ``rows <= chunk`` of them at a time). Both are CENSUS'D by shape with counts, and the largest leading product of (b) is
    recorded so the test asserts it ``<= attention chunk x H``. ``max_numel`` is tracked over everything that is NOT (a)/(b) (a fact; the
    core's fixed-size ring buffers exceed a toy-sized whole pair tensor, so numel alone cannot be the rule at N_st=105)."""
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    class Guard(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.offenders, self.max_numel, self.max_shape, self.censused, self.attn_lead_max = [], 0, None, {}, 0

        def _census(self, kind, shape):
            self.censused[(kind, shape)] = self.censused.get((kind, shape), 0) + 1

        def _see(self, func, leaf):
            shape = tuple(int(d) for d in leaf.shape)
            for n in (n_st, n_res):
                if sum(1 for d in shape if d == n) < 2:
                    continue
                if len(shape) == 2:
                    self.offenders.append(("whole_2d", str(func), shape))                       # [n, n]: a whole bias / mask
                elif len(shape) == 3 and shape[0] == n and shape[1] == n:
                    if shape[2] < c_z:
                        self._census("triatt_bias", shape)                                      # [n, n, H]: the triangle bias before its permute
                    else:
                        self.offenders.append(("whole_pair_track", str(func), shape))           # [n, n, c]: a whole pair track
                elif shape[-1] == n and shape[-2] == n:                                         # [..., n, n]: the triangle bias [H,n,n]/[1,H,n,n] or one row batch's logits
                    lead = 1
                    for d in shape[:-2]:
                        lead *= d
                    if lead <= heads and len(shape) in (3, 4) and (len(shape) == 3 or shape[0] == 1):
                        self._census("triatt_bias", shape)
                    else:
                        self._census("triatt_logits", shape)
                        self.attn_lead_max = max(self.attn_lead_max, lead)
                else:
                    self.offenders.append(("pair_shaped", str(func), shape))                    # two n dims in any other arrangement
                return
            numel = int(leaf.numel())
            if numel > self.max_numel:
                self.max_numel, self.max_shape = numel, (str(func), shape)

        def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for leaf in tree_flatten(out)[0]:
                if isinstance(leaf, torch.Tensor):
                    self._see(func, leaf)
            return out
    return Guard()


# ============================================================================================================ per-rank entry
def _rank_entry(seed: int, mode: str, chunk):
    """Every rank: dense reference (whole tensors, the stub engine's own statements) -> the same model through tp_struct on this rank's rows ->
    compare HERE; every rank's verdicts reach rank 0 (the core's allgather_obj)."""
    install_modules()                                                             # opendde.model.utils (permute_final_dims) for tp's lazy imports
    torch.set_num_threads(1)
    from opendde_opt import tp, tp_diffusion as TD, tp_struct as TS
    from opt_core.mem.rowpair import RowpairRefused, dist as D
    model, feats, s_inputs, s, z = build(seed, mode)
    n_st = int(feats["parent_residue_idx"].numel())
    res = {"checks": {}, "metrics": {}, "bitwise": {}, "facts": {}}
    chk, met, bit, facts = res["checks"], res["metrics"], res["bitwise"], res["facts"]

    def cmp(name, got, ref, tol=TOL):
        d = float((got.contiguous().to(torch.float32) - ref.contiguous().to(torch.float32)).abs().max()) if got.numel() else 0.0
        met[name], bit[name], chk[name] = d, bool(torch.equal(got.contiguous(), ref.contiguous())), d <= tol

    def refuses(fn, *needles):
        try:
            fn()
        except RowpairRefused as e:
            return all(n in str(e) for n in needles) or f"wrong message: {e}"
        except Exception as e:  # noqa: BLE001
            return f"wrong exception type {type(e).__name__}: {e}"
        return False

    with torch.no_grad():
        # ---------------------------------------------------------------- dense reference (single device statements)
        sfd_d, si_d, s_d, z_d = model.expand_to_structural_tokens(copy.copy(feats), s_inputs, s, z.clone(), chunk_size=chunk)
        exp = model.structural_token_expander
        _si_e, _s_e, zexp_d, pf_d = exp(input_feature_dict=feats, s_inputs_res=s_inputs, s_res=s, z_res=z)       # the expansion alone (before the refiner)
        bias_d = pf_d["structural_pair_attn_bias"]
        # ---------------------------------------------------------------- sharded: this rank's rows through tp_struct
        P, r = tp._group()
        lay_res = tp._layout(N_RES)
        lay_st = TD.struct_layout(n_st)
        facts.update(P=P, rank=r, res_rows=(lay_res.r0, lay_res.r1), struct_rows=(lay_st.r0, lay_st.r1), R_st=lay_st.R, layout_res=lay_res.facts(), layout_st=lay_st.facts(),
                     ring_slab=tp._hostgather_rows(lay_res))
        chk["layouts_ragged"] = (lay_res.N % lay_res.Rmax != 0) and (lay_st.N % lay_st.Rmax != 0)
        rs = tp.RowShard(z[lay_res.r0:lay_res.r1].contiguous(), lay_res)
        heads = int(model.structural_token_refiner.blocks[0].tri_att_start.mha.H)
        guard = _pair_guard(n_st, N_RES, C_Z, heads)
        with guard:
            park0, unpark0 = tp.STATS["zres_park"], tp.STATS["zres_unpark"]
            sfd_s, si_s, s_s, zsh = TS.expand_to_structural_tokens_rows(model, copy.copy(feats), s_inputs, s, rs, chunk_size=chunk)
            # ROWPAIR_PARK_ZRES (set by _mp for every rank): the residue shard was PARKED once inside the stage, after fetch_rows_ring read the residue
            # rows the expansion needs, and NOTHING in the stage read it afterwards (no unpark until a downstream reader touches .z) — on a GPU the
            # device storage is released for the structural stage + roll-out; a CPU shard stays resident (parking is a GPU mechanism)
            chk["zres_parked_once_in_stage"] = (tp.STATS["zres_park"] - park0 == 1) and rs.parked and (tp.STATS["zres_unpark"] - unpark0 == 0)
            _ = rs.z                                                              # the distogram / confidence rows' read: unparks exactly once
            chk["zres_unparked_by_first_reader"] = (not rs.parked) and (tp.STATS["zres_unpark"] - unpark0 == 1) and torch.equal(rs.z, z[lay_res.r0:lay_res.r1])
        attn_chunk = tp._attn_rows(chunk, lay_st)
        facts["pair_shaped_offenders"] = guard.offenders[:8]
        facts["n_offenders"] = len(guard.offenders)
        facts["max_op_numel_outside_triatt"], facts["max_op_outside_triatt"] = guard.max_numel, guard.max_shape
        facts["triatt_censused"] = {f"{k[0]}{list(k[1])}": v for k, v in sorted(guard.censused.items(), key=lambda kv: -kv[1])[:8]}
        facts["triatt_lead_max"], facts["attn_chunk"], facts["triatt_heads"] = guard.attn_lead_max, attn_chunk, heads
        chk["never_pair_track_shaped"] = len(guard.offenders) == 0 or guard.offenders[:4]
        chk["triatt_logits_bounded_by_chunk_x_heads"] = 0 < guard.attn_lead_max <= attn_chunk * heads or (guard.attn_lead_max, attn_chunk, heads)
        chk["zstruct_is_rowshard_of_struct_layout"] = isinstance(zsh, tp.RowShard) and zsh.lay is lay_st and tuple(zsh.z.shape) == (lay_st.R, n_st, C_Z)
        # ---------------------------------------------------------------- rows vs dense
        r0, r1 = lay_st.r0, lay_st.r1
        pair_dtype = os.environ.get("ODDE_TP_STRUCT_PAIR_DTYPE", "fp32")                # struct_pair_bf16: the refined shard is bf16 -> a scale-aware tier-2 tolerance, reported
        tol_ref = TOL if pair_dtype != "bf16" else 2e-2 * float(z_d.abs().max()) + 2e-2
        cmp("z_struct_rows_refined", zsh.z, z_d[r0:r1], tol=tol_ref)
        cmp("s_struct_refined", s_s, s_d, tol=TOL if pair_dtype != "bf16" else 2e-2 * float(s_d.abs().max()) + 2e-2)
        chk["zstruct_dtype"] = zsh.z.dtype == (torch.bfloat16 if pair_dtype == "bf16" else torch.float32)
        chk["stats_struct_pair_dtype"] = tp.STATS.get("struct_pair_dtype") == pair_dtype and tp.STATS.get("struct_bf16_calls") == (1 if pair_dtype == "bf16" else 0)
        facts["refined_max_abs_diff_z"], facts["refined_ref_max_abs_z"] = None, float(z_d.abs().max())
        cmp("s_inputs_struct", si_s, si_d)
        cmp("extra_bias_rows", sfd_s[TD.EXTRA_BIAS_KEY], bias_d[r0:r1])
        chk["extra_bias_rows_marker"] = sfd_s.get(TD.EXTRA_BIAS_ROWS_KEY) is True and tuple(sfd_s[TD.EXTRA_BIAS_KEY].shape) == (lay_st.R, n_st)
        # the expansion alone: bit for bit (index_select copies + the same per-row statements)
        rows_needed = TS.struct_rows_needed(feats["parent_residue_idx"], lay_st)
        zres_rows, zres_index = TS.fetch_rows_ring(rs.z, rows_needed, lay_res)
        K = int(zres_index.numel())
        facts["K"] = K
        chk["rows_needed_are_the_local_parents"] = torch.equal(rows_needed, torch.unique(feats["parent_residue_idx"][r0:r1])) and torch.equal(zres_index, rows_needed)
        cmp("fetched_parent_rows", zres_rows, z[zres_index])
        bias_rows = torch.empty((lay_st.R, n_st))
        zexp_rows = TS.expand_rows(exp, feats, s_inputs, s, zres_rows, zres_index, lay_st, rows=7, attn_bias_out=bias_rows)   # a block height that does not divide R_st
        cmp("z_struct_rows_expanded", zexp_rows, zexp_d[r0:r1])
        cmp("extra_bias_rows_expanded", bias_rows, bias_d[r0:r1])
        si_e, s_e = TS.expand_single(exp, feats, s_inputs, s)
        cmp("s_struct_expanded", s_e, _s_e)
        # refine_rows alone on the dense expansion's rows == the dense refiner (same tolerance; s carried)
        z_in = zexp_d[r0:r1].clone().contiguous()
        s_ref_out, z_ref_out = model.structural_token_refiner(s=_s_e.clone(), z=zexp_d.clone(), pair_mask=None, extra_attn_bias=bias_d)
        s_out, z_out = TS.refine_rows(model.structural_token_refiner, z_in, lay_st, s=_s_e.clone(), extra_attn_bias=bias_rows, triangle_attention="torch", chunk_size=chunk)
        cmp("refine_rows_z", z_out, z_ref_out[r0:r1])
        cmp("refine_rows_s", s_out, s_ref_out)
        chk["refine_rows_in_place"] = z_out.data_ptr() == z_in.data_ptr()
        # a WHOLE extra bias handed to refine_rows == the rows form (tp slices what it is handed)
        s_out_w, z_out_w = TS.refine_rows(model.structural_token_refiner, zexp_d[r0:r1].clone().contiguous(), lay_st, s=_s_e.clone(), extra_attn_bias=bias_d,
                                          triangle_attention="torch", chunk_size=chunk)
        bit["extra_bias_whole_vs_rows_z"] = chk["extra_bias_whole_vs_rows_z"] = bool(torch.equal(z_out_w, z_out))
        bit["extra_bias_whole_vs_rows_s"] = chk["extra_bias_whole_vs_rows_s"] = bool(torch.equal(s_out_w, s_out))
        # ---------------------------------------------------------------- the structural feature dict == the dense one
        keys_d, keys_s = set(sfd_d), set(sfd_s) - {TD.EXTRA_BIAS_ROWS_KEY}
        chk["feature_dict_keys"] = keys_d == keys_s or f"dense-only={sorted(keys_d - keys_s)} rows-only={sorted(keys_s - keys_d)}"
        same = {}
        for k in sorted(keys_d & keys_s):
            a, b = sfd_d[k], sfd_s[k]
            if k == TD.EXTRA_BIAS_KEY:
                continue
            if torch.is_tensor(a):
                same[k] = torch.is_tensor(b) and a.shape == b.shape and torch.equal(a, b)
            elif isinstance(a, LazyRelp):
                same[k] = isinstance(b, LazyRelp) and all(torch.equal(getattr(a, f), getattr(b, f)) for f in ("asym_id", "residue_index", "entity_id", "token_index", "sym_id"))
            else:
                same[k] = a == b
        chk["feature_dict_values"] = all(same.values()) or sorted(k for k, v in same.items() if not v)
        chk["residue_only_dropped"] = not any(k in sfd_s for k in RESIDUE_ONLY) and all(k in feats for k in RESIDUE_ONLY)
        chk["relp_lazy_structural"] = isinstance(sfd_s.get("relp"), LazyRelp) and int(sfd_s["relp"].token_index.numel()) == n_st
        # ---------------------------------------------------------------- census words
        facts["stats"] = {k: tp.STATS.get(k) for k in ("struct_rows", "zres_ring_rows", "struct_presharded_calls", "struct_stage")}
        chk["stats_struct_rows"] = tp.STATS.get("struct_rows") == lay_st.R
        chk["stats_zres_ring_rows"] = tp.STATS.get("zres_ring_rows") == K and 1 <= K <= lay_st.R
        chk["stats_struct_presharded_calls"] = tp.STATS.get("struct_presharded_calls") == 3          # the stage entry's refiner call + the two refine_rows calls above
        chk["stats_struct_stage"] = tp.STATS.get("struct_stage") == "rowshard"
        fl = dict(TS.fields())
        chk["fields_named"] = fl["struct_stage"] == "rowshard" and fl["struct_R"] == lay_st.R and "struct_rows" not in fl and "s_struct" in fl["struct_replicated_by_design"]
        from opt_core.mem.rowpair import evidence as _ev                            # the pinned core ships the schedule record: an absent one is a named failure, not a pass
        sch = dict(_ev.schedule_fields())
        facts["schedule"] = {k: v for k, v in sch.items() if k.startswith("struct_")}
        chk["schedule_recorded"] = (int(sch.get("struct_expand_rows", 0)) >= 1 and sch.get("struct_expand_stmt") == "chunked_rows" and int(sch.get("struct_ring_slabs", 0)) >= 2
                                    and int(sch.get("struct_ring_slab_rows", 0)) == tp._hostgather_rows(lay_res) and int(sch.get("struct_refiner_blocks", 0)) == N_REF_BLOCKS) or facts["schedule"]
        # ---------------------------------------------------------------- fetch_rows_ring: arbitrary rows (every owner), an empty request, duplicates / unsorted
        gq = torch.Generator().manual_seed(77 + r)
        want = torch.cat([torch.tensor([q0 for q0, q1 in lay_res.bounds if q1 > q0]),                       # the first row of every rank
                          torch.tensor([q1 - 1 for q0, q1 in lay_res.bounds if q1 > q0]),                    # the last row of every rank
                          torch.randint(0, N_RES, (9,), generator=gq)])                                       # rank-dependent random rows (duplicates allowed, unsorted)
        rows_a, idx_a = TS.fetch_rows_ring(rs.z, want, lay_res)
        chk["fetch_arbitrary_index"] = torch.equal(idx_a, torch.unique(want))
        cmp("fetch_arbitrary_rows", rows_a, z[idx_a])
        empty = torch.empty(0, dtype=torch.long) if r == 0 else torch.arange(N_RES)                         # rank 0 asks for nothing while the others ask for everything
        rows_b, idx_b = TS.fetch_rows_ring(rs.z, empty, lay_res)
        chk["fetch_empty_or_all_shape"] = tuple(rows_b.shape) == (int(empty.numel()), N_RES, C_Z) and torch.equal(idx_b, empty)
        cmp("fetch_empty_or_all_rows", rows_b, z[idx_b])
        # ---------------------------------------------------------------- refusals by name (every rank refuses alike, before any collective: no rank is left waiting)
        chk["refuses_non_rowshard"] = refuses(lambda: TS.expand_to_structural_tokens_rows(model, copy.copy(feats), s_inputs, s, z), "RowShard") is True
        bad = copy.copy(feats)
        bad["parent_residue_idx"] = feats["parent_residue_idx"].clone()
        bad["parent_residue_idx"][3] = N_RES
        chk["refuses_parent_out_of_range"] = refuses(lambda: TS.expand_to_structural_tokens_rows(model, bad, s_inputs, s, rs), "parent_residue_idx", f"N={N_RES}") is True
        chk["refuses_rows_out_of_range"] = refuses(lambda: TS.fetch_rows_ring(rs.z, torch.tensor([0, N_RES]), lay_res), "outside [0, N=") is True
        chk["refuses_parent_wrong_length"] = refuses(lambda: TS.struct_rows_needed(feats["parent_residue_idx"][:-1], lay_st), "N_st=") is True
        chk["refuses_zres_shape"] = refuses(lambda: TS.fetch_rows_ring(z, want, lay_res), "[R, N, C] expected") is True
        chk["refuses_s_missing"] = refuses(lambda: TS.refine_rows(model.structural_token_refiner, zexp_d[r0:r1].clone(), lay_st, s=None, extra_attn_bias=None,
                                                                  triangle_attention="torch"), "single track") is True
        off = dict(feats)
        m_off = copy.copy(model)
        m_off.enable_structural_token_expansion = False
        chk["refuses_expansion_disabled"] = refuses(lambda: TS.expand_to_structural_tokens_rows(m_off, off, s_inputs, s, rs), "disabled") is True
    # -------------------------------------------------------------------- every rank's verdict reaches rank 0
    allres = D.comm().allgather_obj(res)
    out = dict(res)
    out["all_ok"] = all(all(v is True for v in x["checks"].values()) for x in allres)
    out["failed"] = sorted({f"rank{i}:{k}={v}" for i, x in enumerate(allres) for k, v in x["checks"].items() if v is not True})
    out["bitwise_all_ranks"] = {k: all(x["bitwise"].get(k) for x in allres) for k in res["bitwise"]}
    out["metrics_max"] = {k: max(float(x["metrics"].get(k, 0.0)) for x in allres) for k in res["metrics"]}
    out["facts_all"] = [x["facts"] for x in allres]
    return out


def _mp(P, *args):
    os.environ.setdefault("ROWPAIR_PARK_ZRES", "1")                                 # the BIG_TP line's exports, inherited by the rank processes
    os.environ.setdefault("ODDE_TP_STRUCT_TRIMUL_RB", "8")                           # the refiner's explicit streamed-slab rows (production 128; 8 at the stub's N_st)
    from opendde_opt.tests.conftest import run_sharded_or_skip
    return run_sharded_or_skip(P, _rank_entry, *args, mode="big", backend="gloo", cpu_ok=True, nccl_timeout_s=120, run_timeout_s=900)


@pytest.mark.parametrize("P,mode,chunk", [(2, "full", 8), (3, "factorized", None), (3, "full", 5)],
                         ids=["P2_full_chunk8", "P3_factorized_attnrows", "P3_full_chunk5"])
def test_mp_structural_stage_rows_vs_dense(P, mode, chunk):
    res = _mp(P, 0, mode, chunk)
    print(f"\nP={P} mode={mode} chunk={chunk}:\n  facts(rank0)={res['facts']}\n  facts(all)={res['facts_all']}\n  metrics_max={res['metrics_max']}\n"
          f"  bitwise_all_ranks={res['bitwise_all_ranks']}\n  failed={res['failed']}")
    assert res["all_ok"], res["failed"]
    for k in ("extra_bias_rows_expanded", "extra_bias_rows", "s_inputs_struct", "fetched_parent_rows", "fetch_arbitrary_rows", "fetch_empty_or_all_rows"):
        assert res["bitwise_all_ranks"][k], (k, res["metrics_max"][k])                   # copies and elementwise row statements: bit for bit on every rank
    # z_struct_rows_expanded / the refined tensors: TOL-gated above, torch.equal REPORTED (nn.Linear on a different row-batch composition and the
    # tiled tri-mult contractions may reorder sums; observed bitwise for the expansion on torch 2.7.1+cpu)


def test_p1_refuses_by_name():
    """At --n_gpu 1 the engine's own structural stage runs: every entry of the row stage refuses a P = 1 layout by name (never a dense fallback)."""
    install_modules()
    from opendde_opt import tp, tp_struct as TS
    from opt_core.mem.rowpair import RowpairRefused, dist as D
    model, feats, s_inputs, s, z = build(1, "full")
    n_st = int(feats["parent_residue_idx"].numel())
    lay1_res, lay1_st = D.Layout(N_RES, 1, 0), D.Layout(n_st, 1, 0)
    pat = "n_gpu=1|P=1|P == 1|single rank|not sharded"
    with torch.no_grad():
        with pytest.raises(RowpairRefused, match=pat):
            TS.fetch_rows_ring(z, torch.tensor([0, 1]), lay1_res)
        with pytest.raises(RowpairRefused, match=pat):
            TS.refine_rows(model.structural_token_refiner, z.new_zeros((n_st, n_st, C_Z)), lay1_st, s=s.new_zeros((n_st, C_S)), extra_attn_bias=None, triangle_attention="torch")
        with pytest.raises(RowpairRefused, match=pat):
            TS.expand_to_structural_tokens_rows(model, dict(feats), s_inputs, s, tp.RowShard(z, lay1_res))


def test_struct_rows_needed_is_pure_indexing():
    """``struct_rows_needed`` = sorted unique parents of the local structural rows (no group needed); negative parents refused by name."""
    from opendde_opt import tp_struct as TS
    from opt_core.mem.rowpair import RowpairRefused, dist as D
    parent = torch.tensor([0, 0, 1, 2, 2, 2, 5, 4, 4, 3])
    lay = D.Layout(10, 2, 1, B=4)                                                     # grid: Rmax=8 -> rank 1 owns rows [8, 10)
    assert (lay.r0, lay.r1) == (8, 10)
    assert torch.equal(TS.struct_rows_needed(parent, lay), torch.tensor([3, 4]))
    assert torch.equal(TS.struct_rows_needed(parent[None], D.Layout(10, 2, 0, B=4)), torch.tensor([0, 1, 2, 4, 5]))
    with pytest.raises(RowpairRefused, match="negative"):
        TS.struct_rows_needed(torch.tensor([0, -1, 1, 2, 2, 2, 5, 4, 4, 3]), D.Layout(10, 2, 0, B=4))


def test_module_is_engine_free_at_import():
    """Importing the stage imports neither torch nor the engine nor the core's distributed drivers (lazy imports; the kit's --help path stays
    light) — checked in a FRESH interpreter: this process has long imported them and registered the stub engine modules."""
    import subprocess
    probe = ("import sys, opendde_opt.tp_struct as m; "
             "bad = sorted(k for k in sys.modules if k.startswith(('opendde.', 'torch', 'opt_core.mem.rowpair.dist', 'opt_core.mem.rowpair.pairstack'))); "
             "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert r.returncode == 0, f"opendde_opt.tp_struct imported heavy modules at import: {r.stdout.strip()} {r.stderr[-300:]}"


def test_mp_structural_stage_bf16_shard_within_tier2(monkeypatch):
    """Lever struct_pair_bf16 (ODDE_TP_STRUCT_PAIR_DTYPE=bf16, inherited by the rank processes): the structural shard is STORED bf16 and the
    refiner runs under bf16 autocast on it; expansion rows / extra-bias rows / fetched rows stay bit for bit (they precede the cast); the refined
    z_struct rows and s_struct agree with the fp32 dense reference within a scale-aware tier-2 tolerance (2e-2 x max|ref| + 2e-2; max|diff|
    reported); the census names the dtype; the finite gate passed (the stage returned)."""
    monkeypatch.setenv("ODDE_TP_STRUCT_PAIR_DTYPE", "bf16")
    res = _mp(2, 0, "full", 8)
    print(f"\nbf16 P=2: metrics_max={res['metrics_max']}\n  failed={res['failed']}\n  facts={res['facts']}")
    assert res["all_ok"], res["failed"]
    for k in ("extra_bias_rows_expanded", "extra_bias_rows", "fetched_parent_rows"):
        assert res["bitwise_all_ranks"][k], (k, res["metrics_max"][k])


@pytest.mark.parametrize("rows", [1, 7, 8, 64])
def test_finite_rows_matches_whole_tensor_verdict(rows):
    """The post-refiner finite gate runs per row block (no shard-sized bool temporary): its verdict equals torch.isfinite(t).all() on random
    tensors and on injected NaN / +-Inf in the first, a middle and the (ragged) last block, for fp32 and bf16."""
    from opendde_opt import tp_struct as TS
    g = torch.Generator().manual_seed(0)
    for dtype in (torch.float32, torch.bfloat16):
        t = torch.randn(37, 5, 3, generator=g).to(dtype)                        # 37 rows: ragged last block for rows in {7, 8, 64}
        assert TS.finite_rows(t, rows) is True and bool(torch.isfinite(t).all()) is True
        for where, val in ((0, float("nan")), (18, float("inf")), (36, float("-inf")), (36, float("nan"))):
            bad = t.clone(); bad[where, 2, 1] = val
            assert bool(torch.isfinite(bad).all()) is False
            assert TS.finite_rows(bad, rows) is False, (dtype, where, val, rows)
        s = torch.randn(11, generator=g)                                         # 1-D single-track-like input
        assert TS.finite_rows(s, rows) is True
        s[10] = float("nan")
        assert TS.finite_rows(s, rows) is False
    assert TS.finite_rows(torch.tensor(1.0), rows) is True and TS.finite_rows(torch.tensor(float("nan")), rows) is False
