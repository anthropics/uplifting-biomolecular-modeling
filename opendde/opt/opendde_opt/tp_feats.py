"""The data-pipeline side of the row-sharded line (``--mode big --n_gpu P``, P > 1): every N²-shaped INPUT feature is born as THIS
RANK'S ROWS on the host, never whole — on rank 0 by its featurizer (the patches below), on ranks > 0 from the template precursors of the tree
rank 0 broadcast (:func:`rows_for_rank`; the ``rank0_bcast`` data form, ``tp._rank0_item``).

OpenDDE's inference featurizer builds the template pair features on the host for every query, dummies included:
``template_distogram [T, N, N, 39]``, ``template_unit_vector [T, N, N, 3]``, ``template_pseudo_beta_mask [T, N, N]``,
``template_backbone_frame_mask [T, N, N]`` (``opendde/data/template/template_featurizer.py`` ``Templates.as_opendde_dict``; ``T = 4`` slots)
= ``704 · N²`` bytes per rank process (43 GB at 7,824 tokens, 8 ranks = 8 copies on one host). They are functions of the O(T·N) precursors
the same object carries (``template_aatype [T, N]``, ``template_atom_positions [T, N, 24, 3]``, ``template_atom_mask [T, N, 24]``). In a
rank process :func:`as_opendde_dict_rows` runs the featurizer's own statements RESTRICTED TO ROWS ``r0:r1`` of this rank's layout
(``opendde_opt.tp.layout_of``: the trunk's ``Layout.auto(N, P, rank)``, one function) — ``TemplateFeatures.pseudo_beta_fn`` unchanged, the
distogram and unit-vector statements of ``opendde/data/template/template_utils.py`` (``dgram_from_positions``,
``compute_template_unit_vector``) with the row index restricted, same numpy dtype chain, same einsum subscripts — so every element equals the
dense feature's element (the CPU test compares against the stock's dense output: ``torch.equal``). The dict carries the marker
:data:`ROWS_KEY` ``= [r0, R, N]``; the runner's feature move (``opendde_opt.tp``: ``to_device``) keeps the row slabs on the host and the trunk
entry moves ``[T, R, N(, *)]`` to the card. A whole ``[T, N, N(, *)]`` tensor reaching the move under ``P > 1`` is refused by name there.

Replicated by design (named in the census word ``inputs``): ``token_bonds [N, N]`` fp32 (``4 · N²`` B; the featurizer's bond filter and its
dense adjacency statement are one block — the trunk reads its rows), the MSA features ``[S, N]``, every O(N) feature. Out of scope, named: the
atom-level ``bond_mask [N_atom, N_atom]`` host transient of ``Featurizer.get_mask_features`` (read by no consumer at inference; the kit's
``drop_bond_mask`` lever drops the key).

The empty-template-dict path (``opendde/data/utils.py`` ``make_dummy_feature(dummy_feats=['template'])``: dense ``[T, N, N(, *)]`` zeros)
is refused by name in a rank process: the inference featurizer always assembles the template slots (dummies are assembled, not synthesised
there), so the path is unreachable on the kit's routes and must not silently allocate the dense zeros if upstream changes that.
"""
from __future__ import annotations

import numpy as np

TEMPL_TARGET = "opendde.data.template.template_featurizer"   # Templates.as_opendde_dict
UTILS_TARGET = "opendde.data.utils"                          # make_dummy_feature
ROWS_KEY = "_rowpair_input_rows"                             # marker in the feature dict: int64 [r0, R, N] — the template pair features are rows r0:r0+R of N
PAIR_KEYS = ("template_distogram", "template_unit_vector", "template_pseudo_beta_mask", "template_backbone_frame_mask")   # [T, N, N(, F)] in the stock
PAIR_CHANNELS = {"template_distogram": 39, "template_unit_vector": 3, "template_pseudo_beta_mask": 1, "template_backbone_frame_mask": 1}
STATS = {"rows_calls": 0, "rows_born": 0, "rows_bytes": 0, "full_bytes": 0, "dummy_refused": 0}


def full_pair_bytes(T: int, N: int) -> int:
    """Bytes of the stock's dense fp32 template pair features ``[T, N, N, 39 + 3 + 1 + 1]``."""
    return int(T) * int(N) * int(N) * sum(PAIR_CHANNELS.values()) * 4


def rows_pair_bytes(T: int, R: int, N: int) -> int:
    return int(T) * int(R) * int(N) * sum(PAIR_CHANNELS.values()) * 4


# ----------------------------------------------------------------------------------------------------------------- stock statements, rows r0:r1
def dgram_rows(positions: np.ndarray, r0: int, r1: int, config) -> np.ndarray:
    """Rows ``r0:r1`` of ``TemplateFeatures.dgram_from_positions(positions, config)`` (template_utils.py): the same fp32 statements with the
    first pair index restricted — ``[r1 - r0, N, num_bins]``. The bin breaks are the stock's own cached pair (``TemplateFeatures._dgram_cache``)."""
    from opendde.data.template.template_utils import TemplateFeatures
    key = (config.min_bin, config.max_bin, config.num_bins)
    if key not in TemplateFeatures._dgram_cache:
        TemplateFeatures.dgram_from_positions(positions[:1], config=config)          # one position: fills the stock's break cache, O(1)
    lower_breaks, upper_breaks = TemplateFeatures._dgram_cache[key]
    pos = positions.astype(np.float32, copy=False)
    diff = pos[r0:r1, np.newaxis, :] - pos[np.newaxis, :, :]
    dist2 = np.einsum("ijk,ijk->ij", diff, diff)[..., np.newaxis]
    return ((dist2 > lower_breaks) & (dist2 < upper_breaks)).astype(np.float32)


def unit_vector_rows(aatype: np.ndarray, atom_positions: np.ndarray, atom_mask: np.ndarray, r0: int, r1: int, epsilon: float = 1e-6):
    """Rows ``r0:r1`` of ``TemplateFeatures.compute_template_unit_vector(aatype, atom_positions, atom_mask)`` (template_utils.py):
    ``(unit_vector [r1 - r0, N, 3], mask_2d [r1 - r0, N])``. The per-residue backbone frames are the stock's O(N) statements; the pair statements
    (``diff``, the frame projection, the normalisation, the 2-D mask) run with the first index restricted to the rows."""
    from opendde.data.constants import RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX
    backbone_indices = RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX[aatype, 0]                  # [num_res, 3]: C, CA, N
    c_idx, ca_idx, n_idx = backbone_indices[:, 0], backbone_indices[:, 1], backbone_indices[:, 2]
    num_res = aatype.shape[0]
    res_indices = np.arange(num_res)
    c_pos = atom_positions[res_indices, c_idx].astype(np.float32, copy=False)
    ca_pos = atom_positions[res_indices, ca_idx].astype(np.float32, copy=False)
    n_pos = atom_positions[res_indices, n_idx].astype(np.float32, copy=False)
    mask = (atom_mask[res_indices, c_idx] * atom_mask[res_indices, ca_idx] * atom_mask[res_indices, n_idx]).astype(np.float32)
    v1 = c_pos - ca_pos
    v2 = n_pos - ca_pos
    v1_norm = np.sqrt(np.einsum("ij,ij->i", v1, v1))[:, np.newaxis] + epsilon
    e1 = v1 / v1_norm
    e2 = v2 - np.einsum("ij,ij->i", v2, e1)[:, np.newaxis] * e1
    e2_norm = np.sqrt(np.einsum("ij,ij->i", e2, e2))[:, np.newaxis] + epsilon
    e2 = e2 / e2_norm
    e3 = np.cross(e1, e2)
    R = np.stack([e1, e2, e3], axis=-1)                                               # [num_res, 3, 3]
    diff = ca_pos[np.newaxis, :, :] - ca_pos[r0:r1, np.newaxis, :]                    # rows i in r0:r1 of  ca[j] - ca[i]
    unit_vector = np.einsum("ilk,ijl->ijk", R[r0:r1], diff)
    uv_norm = np.sqrt(np.einsum("ijk,ijk->ij", unit_vector, unit_vector))[..., np.newaxis]
    unit_vector = unit_vector / (uv_norm + epsilon)
    mask_2d = mask[r0:r1, None] * mask[None, :]
    return unit_vector, mask_2d


PRECURSOR_KEYS = ("template_aatype", "template_atom_positions", "template_atom_mask")   # upstream's TEMPLATE_FEATURES: the O(T·N) precursors the four pair features are functions of (``Templates.as_data_dict``)


def pair_rows_from(aatype: np.ndarray, atom_positions: np.ndarray, atom_mask: np.ndarray, r0: int, r1: int, config) -> dict:
    """The four pair features for rows ``r0:r1`` only — ``[T, r1 - r0, N(, F)]`` — from the precursors ``template_aatype [T, N]``,
    ``template_atom_positions [T, N, 24, 3]``, ``template_atom_mask [T, N, 24]`` (numpy in, numpy out): the statements of
    ``Templates.as_opendde_dict`` with the row index restricted (``config`` = its distogram bins, ``Templates._DGRAM_CONFIG``)."""
    from opendde.data.template.template_utils import TemplateFeatures
    num_templates, num_res = int(aatype.shape[0]), int(aatype.shape[1])
    R = int(r1) - int(r0)
    all_pb_masks = np.empty((num_templates, R, num_res), dtype=np.float32)
    all_dgrams = np.empty((num_templates, R, num_res, 39), dtype=np.float32)
    all_unit_vectors = np.empty((num_templates, R, num_res, 3), dtype=np.float32)
    all_bb_masks = np.empty((num_templates, R, num_res), dtype=np.float32)
    for i in range(num_templates):
        aa = aatype[i]
        mask = atom_mask[i]
        pos = atom_positions[i] * mask[..., None]
        pb_pos, pb_mask = TemplateFeatures.pseudo_beta_fn(aa, pos, mask)             # O(N): the stock statement
        pb_mask_2d = pb_mask[r0:r1, None] * pb_mask[None, :]
        dgram = dgram_rows(pb_pos, r0, r1, config)
        all_dgrams[i] = dgram * pb_mask_2d[..., None]
        all_pb_masks[i] = pb_mask_2d
        uv, bb_mask_2d = unit_vector_rows(aa, pos, mask, r0, r1)
        all_unit_vectors[i] = uv * bb_mask_2d[..., None]
        all_bb_masks[i] = bb_mask_2d
    return {"template_pseudo_beta_mask": all_pb_masks, "template_distogram": all_dgrams,
            "template_unit_vector": all_unit_vectors, "template_backbone_frame_mask": all_bb_masks}


def pair_feature_rows(templates, r0: int, r1: int) -> dict:
    """``Templates.as_opendde_dict()`` with the four pair features computed for rows ``r0:r1`` only (:func:`pair_rows_from` on the object's
    precursors); the token-level keys are the stock's ``as_data_dict()``."""
    features = templates.as_data_dict()
    features.update(pair_rows_from(templates.aatype, templates.atom_positions, templates.atom_mask, r0, r1, type(templates)._DGRAM_CONFIG))
    return features


def rows_for_rank(fd) -> dict:
    """This rank's rows of the four pair features + the marker :data:`ROWS_KEY`, built IN ``fd`` from the template precursors the feature dict
    carries (:data:`PRECURSOR_KEYS`, tensors as upstream's ``dict_to_tensor`` left them) and converted as the featurising path converts them
    (``dict_to_tensor``: float32 / int64) — what a rank that did not featurise (the ``rank0_bcast`` data form, ``tp._rank0_item``) adds to the tree
    it received so the feature move and the trunk see the dict the featurising path hands them. Refuses by name when a precursor is absent."""
    from . import tp as _tp
    from opendde.data.template.template_featurizer import Templates
    from opendde.utils.torch_utils import dict_to_tensor
    missing = [k for k in PRECURSOR_KEYS if k not in fd]
    if missing:                                                                  # a receive-side refusal: the core's `feats_bcast_malformed` words (tp._malformed)
        raise _tp._malformed(missing[0], f"the received features carry no template precursors {missing}; this rank cannot build its rows of the "
                                         f"template pair features ({', '.join(PAIR_KEYS)}) without them")
    aatype, pos, mask = (np.asarray(fd[k].cpu() if hasattr(fd[k], "cpu") else fd[k]) for k in PRECURSOR_KEYS)
    T, N = int(aatype.shape[0]), int(aatype.shape[1])
    lay = _tp.layout_of(N, _tp.rank_world(), _tp.rank())                              # the trunk's layout arithmetic (one function); no group needed
    r0, R = int(lay.r0), int(lay.R)
    rows = pair_rows_from(aatype, pos, mask, r0, r0 + R, Templates._DGRAM_CONFIG)
    rows[ROWS_KEY] = np.array([r0, R, N], dtype=np.int64)
    fd.update(dict_to_tensor(rows))
    STATS["rows_calls"] += 1
    STATS["rows_born"] += len(PAIR_KEYS)
    STATS["rows_bytes"] = rows_pair_bytes(T, R, N)
    STATS["full_bytes"] = full_pair_bytes(T, N)
    return fd


# ----------------------------------------------------------------------------------------------------------------- patches (rank process only)
def _make_as_opendde_dict_rows(orig):
    """``Templates.as_opendde_dict`` in a rank process: this rank's rows of the four pair features + the marker :data:`ROWS_KEY`."""
    def as_opendde_dict(self):
        from . import tp as _tp
        if not _tp.in_rank_process():
            return orig(self)
        N = int(self.aatype.shape[1])
        lay = _tp.layout_of(N, _tp.rank_world(), _tp.rank())                          # the trunk's layout arithmetic (one function); no group needed
        r0, R = int(lay.r0), int(lay.R)
        feats = pair_feature_rows(self, r0, r0 + R)
        feats[ROWS_KEY] = np.array([r0, R, N], dtype=np.int64)
        T = int(self.aatype.shape[0])
        STATS["rows_calls"] += 1
        STATS["rows_born"] += len(PAIR_KEYS)
        STATS["rows_bytes"] = rows_pair_bytes(T, R, N)
        STATS["full_bytes"] = full_pair_bytes(T, N)
        return feats
    return as_opendde_dict


def _make_dummy_feature_refusing(orig):
    """``make_dummy_feature`` in a rank process: the ``'template'`` dummy (dense ``[T, N, N(, *)]`` zeros) is refused by name; every other
    dummy (``'msa'``) is the stock statement."""
    def make_dummy_feature(features_dict, dummy_feats=("msa",)):
        from . import tp as _tp
        if _tp.in_rank_process() and "template" in list(dummy_feats):
            from opt_core.mem.rowpair import RowpairRefused
            STATS["dummy_refused"] += 1
            raise RowpairRefused(f"{_tp.LEVER}: refused: make_dummy_feature(dummy_feats={list(dummy_feats)}) would allocate dense [T, N, N, *] "
                                 "template zeros in a rank process (n_gpu>1); the inference featurizer assembles the template slots "
                                 "(InferenceTemplateFeaturizer.make_template_feature) — an empty template dict is not a row-sharded input")
        return orig(features_dict, dummy_feats=dummy_feats)
    return make_dummy_feature


def install_patches(autoload, tag: str, lever: str) -> list:
    """The two featurizer patches (armed at import of their modules); returned for the adapter's ``input_patches`` census."""
    return [autoload.patch_attr_at_import(TEMPL_TARGET, "Templates.as_opendde_dict", _make_as_opendde_dict_rows, tag=tag, name=lever + "_templ_rows"),
            autoload.patch_attr_at_import(UTILS_TARGET, "make_dummy_feature", _make_dummy_feature_refusing, tag=tag, name=lever + "_templ_dummy")]


def born_rows(fd):
    """``(r0, R, N)`` from the marker when the feature dict carries row-born template pair features, else None."""
    m = fd.get(ROWS_KEY) if isinstance(fd, dict) else None
    if m is None:
        return None
    vals = np.asarray(m.cpu() if hasattr(m, "cpu") else m).reshape(-1).astype(np.int64).tolist()   # tensor / array / list, any leading batch dims
    if len(vals) != 3:
        from opt_core.mem.rowpair import RowpairRefused
        raise RowpairRefused(f"rowpair_tp: refused: malformed {ROWS_KEY} marker {vals} (expected [r0, R, N])")
    return tuple(vals)
