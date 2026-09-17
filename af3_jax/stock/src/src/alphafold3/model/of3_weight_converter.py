"""Convert OpenFold3 (PyTorch) weights to AlphaFold3 (JAX/Haiku) format.

Implements the 5 systematic mapping rules:
  1. Parameter name renames (weight→weights, LayerNorm weight/bias→scale/offset)
  2. Linear weight transpositions (PyTorch out×in → JAX in×out)
  3. Attention projection reshaping (Q/K trunk transposed vs others non-transposed)
  4. SwiGLU weight concatenation (linear_a ++ linear_b → transition1)
  5. Layer stack aggregation (N separate blocks → (N, ...) stacked tensors)

No dependency on OpenFold3 source code — only PyTorch and NumPy are required
to run the conversion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ─── OF3→AF3 aatype remap ─────────────────────────────────────────────────────
# OF3 uses 32 residue types; AF3 uses 31.  For each AF3 type index a (0-30),
# _AF3_TO_OF3_AATYPE[a] gives the corresponding OF3 type index.
# AF3: 0-20=protein+UNK, 21=GAP, 22-25=A/G/C/U, 26-29=DA/DG/DC/DT, 30=N
# OF3: 0-20=protein+UNK, 21-25=A/G/C/U/N, 26-30=DA/DG/DC/DT/DN, 31=GAP
_AF3_TO_OF3_AATYPE = np.array(
    list(range(21)) + [31] + [21, 22, 23, 24] + [26, 27, 28, 29] + [25],
    dtype=np.int32,
)


# AF3's MSA one-hot has 32 classes: the same 31 aatype classes plus one spare
# (trailing) class that AF3's featurizer never emits (see msa_features.py, where
# '-'→21, RNA A/G/C/U→22-25, DNA A/G/C/T→26-29, unknown nucleic→30).  OF3 encodes
# the MSA over its own 32-residue alphabet, so the rows must be permuted just
# like aatype.  The unused AF3 class 31 is pointed at OF3's DN row (also never
# activated under AF3 featurization).
_AF3_TO_OF3_MSA = np.concatenate([_AF3_TO_OF3_AATYPE, [30]]).astype(np.int32)

# OF3's unknown-DNA class ('DN'), the one class AF3's 31-way alphabet has no slot
# for (AF3 folds unknown DNA into the shared unknown-nucleic class 'N').
_OF3_UNK_DNA_AATYPE = 30


def _pad_element_weights(w: np.ndarray) -> np.ndarray:
    """Pad OF3 element embedding weights from 119 rows to 128 (AF3 uses 128 classes).

    w shape: (119, c_a) → (128, c_a) by appending 9 zero rows.
    """
    return np.concatenate([w, np.zeros((9,) + w.shape[1:], dtype=w.dtype)], axis=0)


def _reorder_target_feat_weights(w: np.ndarray) -> np.ndarray:
    """Transform target_feat input weights from OF3 layout (449 rows) to AF3 (447 rows).

    OF3 input order: [atom_cross_att(384), OF3_aatype(32), OF3_profile(32), del_mean(1)]
    AF3 input order: [AF3_aatype(31),      AF3_profile(31), del_mean(1),    atom_cross_att(384)]

    For AF3 type a the weight row is taken from OF3 type _AF3_TO_OF3_AATYPE[a].
    Works for 2-D weight matrices (shape 449 × c_out).
    """
    remap = _AF3_TO_OF3_AATYPE
    return np.concatenate([
        w[384 + remap],   # aatype  (31 rows from OF3 aatype block at row 384)
        w[416 + remap],   # profile (31 rows from OF3 profile block at row 416)
        w[448:449],       # del_mean (row 448)
        w[0:384],         # atom_cross_att (rows 0-383)
    ], axis=0)


def _reorder_aatype_weights(w: np.ndarray) -> np.ndarray:
    """Reorder OF3 32-class aatype weight rows to AF3 31-class ordering.

    w shape: (32, c_out) indexed by OF3 aatype class.
    Returns: (31, c_out) indexed by AF3 aatype class.
    """
    return w[_AF3_TO_OF3_AATYPE]


def _reorder_msa_feat_weights(w: np.ndarray) -> np.ndarray:
    """Reorder msa_feat input weights from OF3 to AF3 residue ordering.

    Both layouts are [restype one-hot(32), has_deletion(1), deletion_value(1)],
    so the shape is unchanged (34 rows) — only the 32 restype rows are permuted.

    w shape: (34, c_out) indexed by OF3 MSA class → (34, c_out) in AF3 order.
    """
    return np.concatenate([w[_AF3_TO_OF3_MSA], w[32:34]], axis=0)


def _reorder_features_1d(arr: np.ndarray, c_single: int = 384) -> np.ndarray:
    """Reorder OF3 features_1d (single_emb + target_feat) along axis 0 to AF3 layout.

    OF3 layout: [single(c_single), atom_cross_att(384), OF3_aatype(32), OF3_profile(32), del_mean(1)]
    AF3 layout: [single(c_single), AF3_aatype(31), UNK_DNA(1), AF3_profile(31), UNK_DNA(1),
                 del_mean(1), atom_cross_att(384)]

    Unlike every other residue-indexed matrix, the two aatype classes AF3 lacks
    (OF3's unknown-DNA slot in restype and in profile) may NOT be dropped here:
    this block feeds `single_cond_initial_norm`, a LayerNorm, which turns a zero
    input into -mean/std. OF3 therefore always contributes those two columns
    through their trained weights, and the normalisation statistics run over 833
    channels rather than 831. Both effects are reproduced by keeping the columns
    (diffusion_head inserts matching zero columns when of3_weights is set);
    dropping them costs ~1.6e-3 relative error in the single conditioning.

    Works for 1-D (LayerNorm scale, shape c_single+449) and 2-D (Linear weights,
    shape (c_single+449, c_out)) arrays — reorders along axis 0.
    """
    remap = _AF3_TO_OF3_AATYPE
    unk_dna = _OF3_UNK_DNA_AATYPE
    return np.concatenate([
        arr[0:c_single],
        arr[c_single + 384 + remap],                        # aatype rows
        arr[c_single + 384 + unk_dna: c_single + 385 + unk_dna],  # aatype UNK_DNA
        arr[c_single + 416 + remap],                        # profile rows
        arr[c_single + 416 + unk_dna: c_single + 417 + unk_dna],  # profile UNK_DNA
        arr[c_single + 448: c_single + 449],                # del_mean
        arr[c_single: c_single + 384],                       # atom_cross_att
    ], axis=0)


# ─── Primitive transforms ──────────────────────────────────────────────────────

def _t(w: np.ndarray) -> np.ndarray:
    """Standard linear transpose: PyTorch (out, in) → JAX (in, out)."""
    return w.T


def _q_k_trunk(w: np.ndarray, H: int, D: int) -> np.ndarray:
    """Trunk GridSelfAttention Q or K: (H*D, in) → (H, D, in)."""
    return w.reshape(H, D, -1)


def _v_standard(w: np.ndarray, H: int, D: int) -> np.ndarray:
    """V projection (all attention types): (H*D, in) → (in, H, D)."""
    return w.T.reshape(-1, H, D)


def _gating_trunk(w: np.ndarray) -> np.ndarray:
    """Trunk GridSelfAttention gating: keep as (H*D, in) — already transposed."""
    return w


def _gating_standard(w: np.ndarray) -> np.ndarray:
    """Standard self_attention gating: (H*D, in) → (in, H*D)."""
    return w.T


def _q_k_standard(w: np.ndarray, H: int, D: int) -> np.ndarray:
    """Standard (non-transposed) Q/K: (H*D, in) → (in, H, D)."""
    return w.T.reshape(-1, H, D)


# ─── State dict helpers ────────────────────────────────────────────────────────

def _pfx(prefix: str, field: str) -> str:
    return f'{prefix}.{field}' if prefix else field


def _get(sd: dict, key: str) -> np.ndarray:
    v = sd[key]
    if hasattr(v, 'numpy'):
        return v.detach().float().numpy()
    return np.array(v, dtype=np.float32)


def _has(sd: dict, key: str) -> bool:
    return key in sd


# ─── Parameter dict helpers ───────────────────────────────────────────────────

def _set(params: dict, scope: str, name: str, arr: np.ndarray) -> None:
    params.setdefault(scope, {})[name] = arr


def _populate_scope(params: dict, scope: str, local_dict: dict[str, np.ndarray]) -> None:
    """Add local_dict entries to params dict under given scope.

    Keys may be:
      'param_name'        → params[scope]['param_name']
      'sub/param_name'    → params[scope/sub]['param_name']
      'key:value_name'    → params[scope/key]['value_name']
    """
    for local_key, arr in local_dict.items():
        if ':' in local_key:
            parts = local_key.split(':')
            sub_path, name = '/'.join(parts[:-1]), parts[-1]
            full_scope = f'{scope}/{sub_path}' if sub_path else scope
        elif '/' in local_key:
            sub_path, name = local_key.rsplit('/', 1)
            full_scope = f'{scope}/{sub_path}'
        else:
            full_scope = scope
            name = local_key
        params.setdefault(full_scope, {})[name] = arr


# ─── Module-level converters ───────────────────────────────────────────────────

def convert_layernorm(sd: dict, prefix: str) -> dict[str, np.ndarray]:
    return {
        'scale': _get(sd, _pfx(prefix, 'weight')),
        'offset': _get(sd, _pfx(prefix, 'bias')),
    }


def convert_swiglu_transition(sd: dict, prefix: str) -> dict[str, np.ndarray]:
    d = {}
    d['input_layer_norm/scale'] = _get(sd, _pfx(prefix, 'layer_norm.weight'))
    d['input_layer_norm/offset'] = _get(sd, _pfx(prefix, 'layer_norm.bias'))
    wa = _get(sd, _pfx(prefix, 'swiglu.linear_a.weight'))
    wb = _get(sd, _pfx(prefix, 'swiglu.linear_b.weight'))
    d['transition1/weights'] = np.concatenate([wa.T, wb.T], axis=-1)
    d['transition2/weights'] = _t(_get(sd, _pfx(prefix, 'linear_out.weight')))
    return d


def convert_triangle_mul(sd: dict, prefix: str, outgoing: bool = True) -> dict[str, np.ndarray]:
    d = {}
    d['left_norm_input/scale'] = _get(sd, _pfx(prefix, 'layer_norm_in.weight'))
    d['left_norm_input/offset'] = _get(sd, _pfx(prefix, 'layer_norm_in.bias'))
    d['center_norm/scale'] = _get(sd, _pfx(prefix, 'layer_norm_out.weight'))
    d['center_norm/offset'] = _get(sd, _pfx(prefix, 'layer_norm_out.bias'))
    ap = _get(sd, _pfx(prefix, 'linear_a_p.weight'))
    bp = _get(sd, _pfx(prefix, 'linear_b_p.weight'))
    ag = _get(sd, _pfx(prefix, 'linear_a_g.weight'))
    bg = _get(sd, _pfx(prefix, 'linear_b_g.weight'))
    if not outgoing:
        ap, bp = bp, ap
        ag, bg = bg, ag
    d['projection/weights'] = np.stack([ap.T, bp.T], axis=-1).reshape(ap.shape[1], -1)
    d['gate/weights'] = np.stack([ag.T, bg.T], axis=-1).reshape(ag.shape[1], -1)
    d['gating_linear/weights'] = _t(_get(sd, _pfx(prefix, 'linear_g.weight')))
    d['output_projection/weights'] = _t(_get(sd, _pfx(prefix, 'linear_z.weight')))
    return d


def convert_grid_attention(sd: dict, prefix: str, H: int, D: int) -> dict[str, np.ndarray]:
    d = {}
    d['act_norm/scale'] = _get(sd, _pfx(prefix, 'layer_norm.weight'))
    d['act_norm/offset'] = _get(sd, _pfx(prefix, 'layer_norm.bias'))
    d['pair_bias_projection/weights'] = _t(_get(sd, _pfx(prefix, 'linear_z.weight')))
    d['q_projection/weights'] = _q_k_trunk(_get(sd, _pfx(prefix, 'mha.linear_q.weight')), H, D)
    d['k_projection/weights'] = _q_k_trunk(_get(sd, _pfx(prefix, 'mha.linear_k.weight')), H, D)
    d['v_projection/weights'] = _v_standard(_get(sd, _pfx(prefix, 'mha.linear_v.weight')), H, D)
    d['gating_query/weights'] = _gating_trunk(_get(sd, _pfx(prefix, 'mha.linear_g.weight')))
    d['output_projection/weights'] = _t(_get(sd, _pfx(prefix, 'mha.linear_o.weight')))
    return d


def convert_msa_attention(sd: dict, prefix: str, H: int, D: int) -> dict[str, np.ndarray]:
    d = {}
    d['act_norm/scale'] = _get(sd, _pfx(prefix, 'layer_norm_m.weight'))
    d['act_norm/offset'] = _get(sd, _pfx(prefix, 'layer_norm_m.bias'))
    d['pair_norm/scale'] = _get(sd, _pfx(prefix, 'layer_norm_z.weight'))
    d['pair_norm/offset'] = _get(sd, _pfx(prefix, 'layer_norm_z.bias'))
    d['pair_logits/weights'] = _t(_get(sd, _pfx(prefix, 'linear_z.weight')))
    d['v_projection/weights'] = _v_standard(_get(sd, _pfx(prefix, 'linear_v.weight')), H, D)
    d['gating_query/weights'] = _t(_get(sd, _pfx(prefix, 'linear_g.weight')))
    d['output_projection/weights'] = _t(_get(sd, _pfx(prefix, 'linear_o.weight')))
    return d


def convert_outer_product_mean(sd: dict, prefix: str, c_hidden: int, c_z: int) -> dict[str, np.ndarray]:
    d = {}
    d['layer_norm_input/scale'] = _get(sd, _pfx(prefix, 'layer_norm.weight'))
    d['layer_norm_input/offset'] = _get(sd, _pfx(prefix, 'layer_norm.bias'))
    d['left_projection/weights'] = _t(_get(sd, _pfx(prefix, 'linear_1.weight')))
    d['right_projection/weights'] = _t(_get(sd, _pfx(prefix, 'linear_2.weight')))
    lo_w = _get(sd, _pfx(prefix, 'linear_out.weight'))
    d['__top__/output_w'] = lo_w.T.reshape(c_hidden, c_hidden, c_z)
    d['__top__/output_b'] = _get(sd, _pfx(prefix, 'linear_out.bias'))
    return d


def convert_single_attention(sd: dict, prefix: str, H: int, D: int) -> dict[str, np.ndarray]:
    d = {}
    d['single_pair_logits_norm/scale'] = _get(sd, _pfx(prefix, 'layer_norm_z.weight'))
    d['single_pair_logits_norm/offset'] = _get(sd, _pfx(prefix, 'layer_norm_z.bias'))
    d['single_pair_logits_projection/weights'] = _t(_get(sd, _pfx(prefix, 'linear_z.weight')))
    d['single_attention_layer_norm/scale'] = _get(sd, _pfx(prefix, 'layer_norm_a.weight'))
    d['single_attention_layer_norm/offset'] = _get(sd, _pfx(prefix, 'layer_norm_a.bias'))
    d['single_attention_q_projection/weights'] = _q_k_standard(
        _get(sd, _pfx(prefix, 'mha.linear_q.weight')), H, D)
    if _has(sd, _pfx(prefix, 'mha.linear_q.bias')):
        d['single_attention_q_projection/bias'] = (
            _get(sd, _pfx(prefix, 'mha.linear_q.bias')).reshape(H, D))
    d['single_attention_k_projection/weights'] = _q_k_standard(
        _get(sd, _pfx(prefix, 'mha.linear_k.weight')), H, D)
    d['single_attention_v_projection/weights'] = _q_k_standard(
        _get(sd, _pfx(prefix, 'mha.linear_v.weight')), H, D)
    d['single_attention_gating_query/weights'] = _gating_standard(
        _get(sd, _pfx(prefix, 'mha.linear_g.weight')))
    d['single_attention_transition2/weights'] = _t(_get(sd, _pfx(prefix, 'mha.linear_o.weight')))
    return d


# ─── Block composers ──────────────────────────────────────────────────────────

def _pairblock_params(sd: dict, prefix: str,
                      pair_H: int, pair_D: int) -> dict[str, np.ndarray]:
    d = {}
    for tag, of3_name, outgoing in [
        ('triangle_multiplication_outgoing', 'tri_mul_out', True),
        ('triangle_multiplication_incoming', 'tri_mul_in', False),
    ]:
        for k, v in convert_triangle_mul(sd, f'{prefix}.{of3_name}', outgoing=outgoing).items():
            d[f'{tag}/{k}'] = v
    for tag, of3_name in [('pair_attention1', 'tri_att_start'),
                           ('pair_attention2', 'tri_att_end')]:
        for k, v in convert_grid_attention(sd, f'{prefix}.{of3_name}', pair_H, pair_D).items():
            d[f'{tag}/{k}'] = v
    for k, v in convert_swiglu_transition(sd, f'{prefix}.pair_transition').items():
        d[f'pair_transition/{k}'] = v
    return d


def pairformer_block_params(sd: dict, block_idx: int,
                             pair_H: int = 4, pair_D: int = 32,
                             single_H: int = 16, single_D: int = 24) -> dict[str, np.ndarray]:
    prefix = f'pairformer_stack.blocks.{block_idx}'
    d = {}
    for k, v in _pairblock_params(sd, f'{prefix}.pair_stack', pair_H, pair_D).items():
        d[k] = v
    for k, v in convert_single_attention(sd, f'{prefix}.attn_pair_bias', single_H, single_D).items():
        d[k] = v
    for k, v in convert_swiglu_transition(sd, f'{prefix}.single_transition').items():
        d[f'single_transition/{k}'] = v
    return d


def msa_block_params(sd: dict, block_idx: int,
                     msa_H: int = 8, msa_D: int = 8,
                     pair_H: int = 4, pair_D: int = 32,
                     opm_hidden: int = 32, c_z: int = 128) -> dict[str, np.ndarray]:
    prefix = f'msa_module.blocks.{block_idx}'
    d = {}
    if _has(sd, f'{prefix}.msa_att_row.layer_norm_m.weight'):
        for k, v in convert_msa_attention(sd, f'{prefix}.msa_att_row', msa_H, msa_D).items():
            d[f'msa_attention1/{k}'] = v
    if _has(sd, f'{prefix}.msa_transition.layer_norm.weight'):
        for k, v in convert_swiglu_transition(sd, f'{prefix}.msa_transition').items():
            d[f'msa_transition/{k}'] = v
    for k, v in convert_outer_product_mean(sd, f'{prefix}.outer_product_mean', opm_hidden, c_z).items():
        if k.startswith('__top__/'):
            d[f'outer_product_mean:{k[len("__top__/"):]}'] = v
        else:
            d[f'outer_product_mean/{k}'] = v
    for k, v in _pairblock_params(sd, f'{prefix}.pair_stack', pair_H, pair_D).items():
        d[k] = v
    return d


# ─── Layer stack aggregation ──────────────────────────────────────────────────

def _stack_blocks(per_block_fn, n_blocks: int, **kwargs) -> dict[str, np.ndarray]:
    all_dicts = [per_block_fn(block_idx=i, **kwargs) for i in range(n_blocks)]
    result = {}
    for key in all_dicts[0]:
        reference = all_dicts[0][key]
        arrays = []
        for d in all_dicts:
            arrays.append(d[key] if key in d else np.zeros_like(reference))
        result[key] = np.stack(arrays, axis=0)
    return result


# ─── Top-level section mappers ────────────────────────────────────────────────

def map_pairformer_stack(sd: dict, params: dict,
                          n_blocks: int = 48,
                          pair_H: int = 4, pair_D: int = 32,
                          single_H: int = 16, single_D: int = 24) -> None:
    scope = 'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer'
    stacked = _stack_blocks(
        lambda block_idx: pairformer_block_params(sd, block_idx, pair_H, pair_D, single_H, single_D),
        n_blocks)
    _populate_scope(params, scope, stacked)


def map_msa_stack(sd: dict, params: dict,
                   n_blocks: int = 4,
                   msa_H: int = 8, msa_D: int = 8,
                   pair_H: int = 4, pair_D: int = 32,
                   opm_hidden: int = 32, c_z: int = 128) -> None:
    scope = 'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack'
    stacked = _stack_blocks(
        lambda block_idx: msa_block_params(sd, block_idx, msa_H, msa_D, pair_H, pair_D, opm_hidden, c_z),
        n_blocks)
    _populate_scope(params, scope, stacked)


def map_evoformer_input_embeddings(sd: dict, params: dict) -> None:
    scope = 'diffuser/evoformer'
    params.setdefault(f'{scope}/prev_embedding_layer_norm', {}).update({
        'scale': _get(sd, 'layer_norm_z.weight'),
        'offset': _get(sd, 'layer_norm_z.bias'),
    })
    params.setdefault(f'{scope}/prev_embedding', {})['weights'] = _t(_get(sd, 'linear_z.weight'))
    params.setdefault(f'{scope}/prev_single_embedding_layer_norm', {}).update({
        'scale': _get(sd, 'layer_norm_s.weight'),
        'offset': _get(sd, 'layer_norm_s.bias'),
    })
    params.setdefault(f'{scope}/prev_single_embedding', {})['weights'] = _t(_get(sd, 'linear_s.weight'))
    if _has(sd, 'input_embedder.linear_s.weight'):
        params.setdefault(f'{scope}/single_activations', {})['weights'] = _reorder_target_feat_weights(
            _t(_get(sd, 'input_embedder.linear_s.weight')))
    if _has(sd, 'input_embedder.linear_z_i.weight'):
        params.setdefault(f'{scope}/left_single', {})['weights'] = _reorder_target_feat_weights(
            _t(_get(sd, 'input_embedder.linear_z_i.weight')))
    if _has(sd, 'input_embedder.linear_z_j.weight'):
        params.setdefault(f'{scope}/right_single', {})['weights'] = _reorder_target_feat_weights(
            _t(_get(sd, 'input_embedder.linear_z_j.weight')))
    if _has(sd, 'input_embedder.linear_relpos.weight'):
        params.setdefault(f'{scope}/~_relative_encoding/position_activations', {})['weights'] = _t(
            _get(sd, 'input_embedder.linear_relpos.weight'))
    if _has(sd, 'input_embedder.linear_token_bonds.weight'):
        params.setdefault(f'{scope}/bond_embedding', {})['weights'] = _t(
            _get(sd, 'input_embedder.linear_token_bonds.weight'))
    if _has(sd, 'msa_module_embedder.linear_m.weight'):
        params.setdefault(f'{scope}/msa_activations', {})['weights'] = _reorder_msa_feat_weights(
            _t(_get(sd, 'msa_module_embedder.linear_m.weight')))
    if _has(sd, 'msa_module_embedder.linear_s_input.weight'):
        params.setdefault(f'{scope}/extra_msa_target_feat', {})['weights'] = _reorder_target_feat_weights(
            _t(_get(sd, 'msa_module_embedder.linear_s_input.weight')))


def map_distogram_head(sd: dict, params: dict) -> None:
    if _has(sd, 'aux_heads.distogram.linear.weight'):
        _set(params, 'diffuser/distogram_head/half_logits', 'weights',
             _t(_get(sd, 'aux_heads.distogram.linear.weight')))


def map_confidence_head(sd: dict, params: dict,
                         n_layers: int = 4,
                         pair_H: int = 4, pair_D: int = 32,
                         single_H: int = 16, single_D: int = 24,
                         c_z: int = 128, c_s: int = 384,
                         max_atoms_per_token: int = 24) -> None:
    scope_base = 'diffuser/confidence_head'
    stack_scope = f'{scope_base}/__layer_stack_no_per_layer/confidence_pairformer'
    sentinel = 'aux_heads.pairformer_embedding.pairformer_stack.blocks.0.pair_stack.tri_mul_out.layer_norm_in.weight'
    if _has(sd, sentinel):
        def _block_fn(block_idx):
            prefix = f'aux_heads.pairformer_embedding.pairformer_stack.blocks.{block_idx}'
            d = {}
            for k, v in _pairblock_params(sd, f'{prefix}.pair_stack', pair_H, pair_D).items():
                d[k] = v
            for k, v in convert_single_attention(sd, f'{prefix}.attn_pair_bias', single_H, single_D).items():
                d[k] = v
            for k, v in convert_swiglu_transition(sd, f'{prefix}.single_transition').items():
                d[f'single_transition/{k}'] = v
            return d
        _populate_scope(params, stack_scope, _stack_blocks(_block_fn, n_layers))
    pe = 'aux_heads.pairformer_embedding'
    embed_scope = f'{scope_base}/~_embed_features'
    # NOTE: AF3's left/right naming is TRANSPOSED between its two pair-embedding
    # sites, so linear_i/linear_j cannot be mapped by name alone:
    #   evoformer._seq_pair_embedding: left_single[:, None] + right_single[None]
    #       -> left  indexes row i, right indexes column j
    #   confidence_head._embed_features: left(tf) + right(tf)[:, None]
    #       -> left  indexes column j, right indexes row i
    # OF3 uses the evoformer convention in both places
    # (z = emb_i[..., None, :] + emb_j[..., None, :, :], so linear_i is row i),
    # hence linear_i -> right and linear_j -> left in the confidence head.
    if _has(sd, f'{pe}.linear_i.weight'):
        _set(params, f'{embed_scope}/right_target_feat_project', 'weights',
             _reorder_target_feat_weights(_t(_get(sd, f'{pe}.linear_i.weight'))))
    if _has(sd, f'{pe}.linear_j.weight'):
        _set(params, f'{embed_scope}/left_target_feat_project', 'weights',
             _reorder_target_feat_weights(_t(_get(sd, f'{pe}.linear_j.weight'))))
    if _has(sd, f'{pe}.linear_distance.weight'):
        _set(params, f'{embed_scope}/distogram_feat_project', 'weights', _t(_get(sd, f'{pe}.linear_distance.weight')))
    s = 'aux_heads'
    if _has(sd, f'{s}.pde.layer_norm.weight'):
        params.setdefault(f'{scope_base}/logits_ln', {}).update(convert_layernorm(sd, f'{s}.pde.layer_norm'))
        _set(params, f'{scope_base}/left_half_distance_logits', 'weights', _t(_get(sd, f'{s}.pde.linear.weight')))
    if _has(sd, f'{s}.pae.layer_norm.weight'):
        params.setdefault(f'{scope_base}/pae_logits_ln', {}).update(convert_layernorm(sd, f'{s}.pae.layer_norm'))
        _set(params, f'{scope_base}/pae_logits', 'weights', _t(_get(sd, f'{s}.pae.linear.weight')))
    if _has(sd, f'{s}.plddt.layer_norm.weight'):
        params.setdefault(f'{scope_base}/plddt_logits_ln', {}).update(convert_layernorm(sd, f'{s}.plddt.layer_norm'))
        w = _get(sd, f'{s}.plddt.linear.weight')
        # w shape (PyTorch): (of3_atoms_per_token * 50, c_s); after .T: (c_s, of3_atoms*50)
        of3_atoms = w.shape[0] // 50
        logits = w.T.reshape(c_s, of3_atoms, 50)
        if of3_atoms < max_atoms_per_token:
            pad = np.zeros((c_s, max_atoms_per_token - of3_atoms, 50), dtype=logits.dtype)
            logits = np.concatenate([logits, pad], axis=1)
        _set(params, f'{scope_base}/plddt_logits', 'weights', logits)
    if _has(sd, f'{s}.experimentally_resolved.layer_norm.weight'):
        params.setdefault(f'{scope_base}/experimentally_resolved_ln', {}).update(
            convert_layernorm(sd, f'{s}.experimentally_resolved.layer_norm'))
        w = _get(sd, f'{s}.experimentally_resolved.linear.weight')
        # w shape (PyTorch): (of3_atoms_per_token * 2, c_s); after .T: (c_s, of3_atoms*2)
        of3_atoms = w.shape[0] // 2
        logits = w.T.reshape(c_s, of3_atoms, 2)
        if of3_atoms < max_atoms_per_token:
            pad = np.zeros((c_s, max_atoms_per_token - of3_atoms, 2), dtype=logits.dtype)
            logits = np.concatenate([logits, pad], axis=1)
        _set(params, f'{scope_base}/experimentally_resolved_logits', 'weights', logits)


# ─── Diffusion transformer block converters ──────────────────────────────────

def _diff_adaln_params(sd: dict, prefix: str, name: str) -> dict[str, np.ndarray]:
    return {
        f'{name}single_cond_layer_norm/scale': _get(sd, f'{prefix}.layer_norm_s.weight'),
        f'{name}single_cond_scale/weights':    _t(_get(sd, f'{prefix}.linear_g.weight')),
        f'{name}single_cond_scale/bias':       _get(sd, f'{prefix}.linear_g.bias'),
        f'{name}single_cond_bias/weights':     _t(_get(sd, f'{prefix}.linear_s.weight')),
    }


def _diff_cond_transition_params(sd: dict, prefix: str, name: str) -> dict[str, np.ndarray]:
    d = {}
    d.update(_diff_adaln_params(sd, f'{prefix}.layer_norm', f'{name}ffw_'))
    wa = _get(sd, f'{prefix}.swiglu.linear_a.weight')
    wb = _get(sd, f'{prefix}.swiglu.linear_b.weight')
    d[f'{name}ffw_transition1/weights'] = np.concatenate([wa.T, wb.T], axis=-1)
    d[f'{name}ffw_transition2/weights'] = _t(_get(sd, f'{prefix}.linear_out.weight'))
    d[f'{name}ffw_adaptive_zero_cond/weights'] = _t(_get(sd, f'{prefix}.linear_g.weight'))
    d[f'{name}ffw_adaptive_zero_cond/bias'] = _get(sd, f'{prefix}.linear_g.bias')
    return d


def convert_diff_self_attn_block(sd: dict, block_idx: int, prefix_base: str,
                                  name: str, H: int, D: int) -> dict[str, np.ndarray]:
    pa = f'{prefix_base}.blocks.{block_idx}.attention_pair_bias'
    pt = f'{prefix_base}.blocks.{block_idx}.conditioned_transition'
    d = {}
    d.update(_diff_adaln_params(sd, f'{pa}.layer_norm_a', name))
    # Per-block pair LayerNorm + projection stored without name prefix to match
    # bare hm.LayerNorm(name='pair_input_layer_norm') in diffusion_transformer.py
    d['pair_input_layer_norm/scale'] = _get(sd, f'{pa}.layer_norm_z.weight')
    d['pair_logits_projection/weights'] = _t(_get(sd, f'{pa}.linear_z.weight'))
    qw = _get(sd, f'{pa}.mha.linear_q.weight')
    d[f'{name}q_projection/weights'] = qw.T.reshape(-1, H, D)
    d[f'{name}q_projection/bias'] = _get(sd, f'{pa}.mha.linear_q.bias').reshape(H, D)
    d[f'{name}k_projection/weights'] = _get(sd, f'{pa}.mha.linear_k.weight').T.reshape(-1, H, D)
    d[f'{name}v_projection/weights'] = _get(sd, f'{pa}.mha.linear_v.weight').T.reshape(-1, H, D)
    d[f'{name}gating_query/weights'] = _t(_get(sd, f'{pa}.mha.linear_g.weight'))
    d[f'{name}transition2/weights'] = _t(_get(sd, f'{pa}.mha.linear_o.weight'))
    d[f'{name}adaptive_zero_cond/weights'] = _t(_get(sd, f'{pa}.linear_ada_out.weight'))
    d[f'{name}adaptive_zero_cond/bias'] = _get(sd, f'{pa}.linear_ada_out.bias')
    d.update(_diff_cond_transition_params(sd, pt, name))
    return d


def convert_diff_cross_attn_block(sd: dict, block_idx: int, prefix_base: str,
                                   name: str, H: int, D: int) -> dict[str, np.ndarray]:
    pa = f'{prefix_base}.blocks.{block_idx}.attention_pair_bias'
    pt = f'{prefix_base}.blocks.{block_idx}.conditioned_transition'
    d = {}
    d.update(_diff_adaln_params(sd, f'{pa}.layer_norm_a_q', f'{name}q'))
    d.update(_diff_adaln_params(sd, f'{pa}.layer_norm_a_k', f'{name}k'))
    qw = _get(sd, f'{pa}.mha.linear_q.weight')
    d[f'{name}q_projection/weights'] = qw.T.reshape(-1, H, D)
    d[f'{name}q_projection/bias'] = _get(sd, f'{pa}.mha.linear_q.bias').reshape(H, D)
    d[f'{name}k_projection/weights'] = _get(sd, f'{pa}.mha.linear_k.weight').T.reshape(-1, H, D)
    d[f'{name}v_projection/weights'] = _get(sd, f'{pa}.mha.linear_v.weight').T.reshape(-1, H, D)
    d[f'{name}gating_query/weights'] = _t(_get(sd, f'{pa}.mha.linear_g.weight'))
    d[f'{name}transition2/weights'] = _t(_get(sd, f'{pa}.mha.linear_o.weight'))
    d[f'{name}adaptive_zero_cond/weights'] = _t(_get(sd, f'{pa}.linear_ada_out.weight'))
    d[f'{name}adaptive_zero_cond/bias'] = _get(sd, f'{pa}.linear_ada_out.bias')
    d.update(_diff_cond_transition_params(sd, pt, name))
    return d


def _stack_diff_transformer(sd: dict, prefix_base: str, n_blocks: int,
                              name: str, H: int, D: int,
                              is_cross_attn: bool = False) -> dict[str, np.ndarray]:
    fn = convert_diff_cross_attn_block if is_cross_attn else convert_diff_self_attn_block
    all_blocks = [fn(sd, i, prefix_base, name, H, D) for i in range(n_blocks)]
    result = {}
    for key in all_blocks[0]:
        result[key] = np.stack([b[key] for b in all_blocks], axis=0)
    return result


def _pair_logits_flat(sd: dict, prefix_base: str, n_blocks: int) -> np.ndarray:
    return np.stack(
        [_get(sd, f'{prefix_base}.blocks.{j}.attention_pair_bias.linear_z.weight').T
         for j in range(n_blocks)],
        axis=1,
    )


def _stack_main_transformer(sd: dict, prefix_base: str, name: str,
                              n_blocks: int, n_super: int, H: int, D: int) -> dict[str, np.ndarray]:
    super_size = n_blocks // n_super
    all_blocks = [convert_diff_self_attn_block(sd, i, prefix_base, name, H, D)
                  for i in range(n_blocks)]
    result = {}
    for key in all_blocks[0]:
        flat = np.stack([b[key] for b in all_blocks], axis=0)
        result[key] = flat.reshape((n_super, super_size) + flat.shape[1:])
    return result


def map_evoformer_conditioning(sd: dict, params: dict, *,
                                n_atom_blocks: int = 3,
                                atom_H: int = 4, atom_D: int = 32) -> None:
    ie_enc = 'input_embedder.atom_attn_enc'
    if not _has(sd, f'{ie_enc}.ref_atom_feature_embedder.linear_ref_pos.weight'):
        return
    scope = 'diffuser'
    rfe = f'{ie_enc}.ref_atom_feature_embedder'
    pfx = 'evoformer_conditioning_'

    def _ec(of3_key, af3_name):
        _set(params, f'{scope}/{pfx}{af3_name}', 'weights', _t(_get(sd, f'{ie_enc}.{of3_key}')))

    _set(params, f'{scope}/{pfx}embed_ref_pos',       'weights', _t(_get(sd, f'{rfe}.linear_ref_pos.weight')))
    _set(params, f'{scope}/{pfx}embed_ref_mask',      'weights', _t(_get(sd, f'{rfe}.linear_ref_mask.weight')))
    _set(params, f'{scope}/{pfx}embed_ref_element',   'weights', _pad_element_weights(_t(_get(sd, f'{rfe}.linear_ref_element.weight'))))
    _set(params, f'{scope}/{pfx}embed_ref_charge',    'weights', _t(_get(sd, f'{rfe}.linear_ref_charge.weight')))
    _set(params, f'{scope}/{pfx}embed_ref_atom_name', 'weights', _t(_get(sd, f'{rfe}.linear_ref_atom_chars.weight')))
    for suf, attr in [('embed_pair_offsets', 'linear_ref_offset'),
                       ('embed_pair_distances', 'linear_inv_sq_dists'),
                       ('embed_pair_offsets_valid', 'linear_valid_mask')]:
        w = _t(_get(sd, f'{rfe}.{attr}.weight'))
        _set(params, f'{scope}/{pfx}{suf}',   'weights', w)
        if suf != 'embed_pair_offsets_valid':
            _set(params, f'{scope}/{pfx}{suf}_1', 'weights', w)
    for suf, attr in [('single_to_pair_cond_row', 'linear_l'),
                       ('single_to_pair_cond_col', 'linear_m')]:
        w = _t(_get(sd, f'{ie_enc}.{attr}.weight'))
        _set(params, f'{scope}/{pfx}{suf}',   'weights', w)
        _set(params, f'{scope}/{pfx}{suf}_1', 'weights', w)
    _ec('pair_mlp.1.weight', 'pair_mlp_1')
    _ec('pair_mlp.3.weight', 'pair_mlp_2')
    _ec('pair_mlp.5.weight', 'pair_mlp_3')
    _ec('linear_q.0.weight', 'project_atom_features_for_aggr')
    enc_base  = f'{ie_enc}.atom_transformer'
    enc_name  = f'{pfx}atom_transformer_encoder'
    enc_scope = f'{scope}/{enc_name}'
    _set(params, f'{enc_scope}/pair_input_layer_norm', 'scale',
         _get(sd, f'{enc_base}.layer_norm_z.weight'))
    enc_stacked = _stack_diff_transformer(sd, enc_base, n_atom_blocks, enc_name, atom_H, atom_D, is_cross_attn=True)
    _populate_scope(params, f'{enc_scope}/__layer_stack_with_per_layer', enc_stacked)
    _set(params, f'{enc_scope}/pair_logits_projection', 'weights',
         _pair_logits_flat(sd, enc_base, n_atom_blocks))


def map_template_embedder(sd: dict, params: dict, *,
                           n_templ_blocks: int = 2,
                           templ_H: int = 4, templ_D: int = 16) -> None:
    te = 'template_embedder'
    if not _has(sd, f'{te}.template_pair_embedder.dgram_linear.weight'):
        return
    tpe = f'{te}.template_pair_embedder'
    tps = f'{te}.template_pair_stack'
    scope_te  = 'diffuser/evoformer/template_embedding'
    scope_ste = f'{scope_te}/single_template_embedding'
    for idx, attr in [(0, 'dgram_linear'), (8, 'linear_z')]:
        _set(params, f'{scope_ste}/template_pair_embedding_{idx}', 'weights',
             _t(_get(sd, f'{tpe}.{attr}.weight')))
    # AF3's to_concat order puts aatype[None, :, :] (column j) at index 2 and
    # aatype[:, None, :] (row i) at index 3, while OF3's aatype_linear_1 is the
    # row-i projection (template_restype[..., None, :]) and aatype_linear_2 the
    # column-j one. The indices are therefore crossed.
    for idx, attr in [(3, 'aatype_linear_1'), (2, 'aatype_linear_2')]:
        _set(params, f'{scope_ste}/template_pair_embedding_{idx}', 'weights',
             _reorder_aatype_weights(_t(_get(sd, f'{tpe}.{attr}.weight'))))
    for idx, attr in [(1, 'pseudo_beta_mask_linear'), (4, 'x_linear'),
                      (5, 'y_linear'), (6, 'z_linear'), (7, 'backbone_mask_linear')]:
        _set(params, f'{scope_ste}/template_pair_embedding_{idx}', 'weights',
             _get(sd, f'{tpe}.{attr}.weight').squeeze(-1))
    _set(params, f'{scope_ste}/query_embedding_norm', 'scale',  _get(sd, f'{tpe}.layer_norm_z.weight'))
    _set(params, f'{scope_ste}/query_embedding_norm', 'offset', _get(sd, f'{tpe}.layer_norm_z.bias'))
    _set(params, f'{scope_ste}/output_layer_norm', 'scale',  _get(sd, f'{tps}.layer_norm.weight'))
    _set(params, f'{scope_ste}/output_layer_norm', 'offset', _get(sd, f'{tps}.layer_norm.bias'))
    stacked = _stack_blocks(
        lambda block_idx: _pairblock_params(sd, f'{tps}.blocks.{block_idx}', templ_H, templ_D),
        n_templ_blocks)
    _populate_scope(params, f'{scope_ste}/__layer_stack_no_per_layer/template_embedding_iteration', stacked)
    _set(params, f'{scope_te}/output_linear', 'weights', _t(_get(sd, f'{te}.linear_t.weight')))


def map_diffusion_head(sd: dict, params: dict, *,
                        n_diff_blocks: int = 24, diff_H: int = 16, diff_D: int = 48,
                        n_super_blocks: int = 6, n_atom_blocks: int = 3,
                        atom_H: int = 4, atom_D: int = 32) -> None:
    scope = 'diffuser/~/diffusion_head'
    dm = 'diffusion_module'
    if not _has(sd, f'{dm}.diffusion_conditioning.layer_norm_z.weight'):
        return
    dc = f'{dm}.diffusion_conditioning'

    def _swiglu_flat(of3_prefix, af3_name):
        _set(params, f'{scope}/{af3_name}ffw_layer_norm', 'scale', _get(sd, f'{of3_prefix}.layer_norm.weight'))
        _set(params, f'{scope}/{af3_name}ffw_layer_norm', 'offset', _get(sd, f'{of3_prefix}.layer_norm.bias'))
        wa = _get(sd, f'{of3_prefix}.swiglu.linear_a.weight')
        wb = _get(sd, f'{of3_prefix}.swiglu.linear_b.weight')
        _set(params, f'{scope}/{af3_name}ffw_transition1', 'weights', np.concatenate([wa.T, wb.T], axis=-1))
        _set(params, f'{scope}/{af3_name}ffw_transition2', 'weights', _t(_get(sd, f'{of3_prefix}.linear_out.weight')))

    _set(params, f'{scope}/pair_cond_initial_norm', 'scale', _get(sd, f'{dc}.layer_norm_z.weight'))
    _set(params, f'{scope}/pair_cond_initial_projection', 'weights', _t(_get(sd, f'{dc}.linear_z.weight')))
    _swiglu_flat(f'{dc}.transition_z.0', 'pair_transition_0')
    _swiglu_flat(f'{dc}.transition_z.1', 'pair_transition_1')
    _set(params, f'{scope}/single_cond_initial_norm', 'scale',
         _reorder_features_1d(_get(sd, f'{dc}.layer_norm_s.weight')))
    _set(params, f'{scope}/single_cond_initial_projection', 'weights',
         _reorder_features_1d(_t(_get(sd, f'{dc}.linear_s.weight'))))
    _set(params, f'{scope}/noise_embedding_initial_norm', 'scale', _get(sd, f'{dc}.layer_norm_n.weight'))
    _set(params, f'{scope}/noise_embedding_initial_projection', 'weights', _t(_get(sd, f'{dc}.linear_n.weight')))
    # Fourier embedding constants (PyTorch RNG differs from JAX RNG at seed=42)
    _set(params, scope, 'fourier_embedding_weight', _get(sd, f'{dc}.fourier_emb.w').astype(np.float32))
    _set(params, scope, 'fourier_embedding_bias',   _get(sd, f'{dc}.fourier_emb.b').astype(np.float32))
    _swiglu_flat(f'{dc}.transition_s.0', 'single_transition_0')
    _swiglu_flat(f'{dc}.transition_s.1', 'single_transition_1')

    ae = f'{dm}.atom_attn_enc'

    def _enc(of3_key, af3_name):
        _set(params, f'{scope}/{af3_name}', 'weights', _t(_get(sd, f'{ae}.{of3_key}')))

    _enc('ref_atom_feature_embedder.linear_ref_pos.weight',        'diffusion_embed_ref_pos')
    _enc('ref_atom_feature_embedder.linear_ref_mask.weight',       'diffusion_embed_ref_mask')
    _set(params, f'{scope}/diffusion_embed_ref_element', 'weights',
         _pad_element_weights(_t(_get(sd, f'{ae}.ref_atom_feature_embedder.linear_ref_element.weight'))))
    _enc('ref_atom_feature_embedder.linear_ref_charge.weight',     'diffusion_embed_ref_charge')
    _enc('ref_atom_feature_embedder.linear_ref_atom_chars.weight', 'diffusion_embed_ref_atom_name')
    for suf, attr in [('diffusion_embed_pair_offsets',   'linear_ref_offset'),
                       ('diffusion_embed_pair_distances', 'linear_inv_sq_dists'),
                       ('diffusion_embed_pair_offsets_valid', 'linear_valid_mask')]:
        w = _t(_get(sd, f'{ae}.ref_atom_feature_embedder.{attr}.weight'))
        _set(params, f'{scope}/{suf}', 'weights', w)
        if suf != 'diffusion_embed_pair_offsets_valid':
            _set(params, f'{scope}/{suf}_1', 'weights', w)
    for suf, attr in [('diffusion_single_to_pair_cond_row', 'linear_l'),
                       ('diffusion_single_to_pair_cond_col', 'linear_m')]:
        w = _t(_get(sd, f'{ae}.{attr}.weight'))
        _set(params, f'{scope}/{suf}',   'weights', w)
        _set(params, f'{scope}/{suf}_1', 'weights', w)
    _enc('pair_mlp.1.weight', 'diffusion_pair_mlp_1')
    _enc('pair_mlp.3.weight', 'diffusion_pair_mlp_2')
    _enc('pair_mlp.5.weight', 'diffusion_pair_mlp_3')
    _enc('noisy_position_embedder.linear_s.weight',  'diffusion_embed_trunk_single_cond')
    _set(params, f'{scope}/diffusion_lnorm_trunk_single_cond', 'scale',
         _get(sd, f'{ae}.noisy_position_embedder.layer_norm_s.weight'))
    _enc('noisy_position_embedder.linear_z.weight',  'diffusion_embed_trunk_pair_cond')
    _set(params, f'{scope}/diffusion_lnorm_trunk_pair_cond', 'scale',
         _get(sd, f'{ae}.noisy_position_embedder.layer_norm_z.weight'))
    _enc('noisy_position_embedder.linear_r.weight',  'diffusion_atom_positions_to_features')
    _enc('linear_q.0.weight', 'diffusion_project_atom_features_for_aggr')

    ad = f'{dm}.atom_attn_dec'
    _set(params, f'{scope}/diffusion_project_token_features_for_broadcast', 'weights',
         _t(_get(sd, f'{ad}.linear_q_in.weight')))
    _set(params, f'{scope}/diffusion_atom_features_layer_norm', 'scale',
         _get(sd, f'{ad}.layer_norm.weight'))
    _set(params, f'{scope}/diffusion_atom_features_to_position_update', 'weights',
         _t(_get(sd, f'{ad}.linear_q_out.weight')))

    for base, name, scope_name in [
        (f'{dm}.atom_attn_enc.atom_transformer', 'diffusion_atom_transformer_encoder',
         'diffusion_atom_transformer_encoder'),
        (f'{dm}.atom_attn_dec.atom_transformer', 'diffusion_atom_transformer_decoder',
         'diffusion_atom_transformer_decoder'),
    ]:
        enc_scope = f'{scope}/{scope_name}'
        _set(params, f'{enc_scope}/pair_input_layer_norm', 'scale',
             _get(sd, f'{base}.layer_norm_z.weight'))
        stacked = _stack_diff_transformer(sd, base, n_atom_blocks, name, atom_H, atom_D, is_cross_attn=True)
        _populate_scope(params, f'{enc_scope}/__layer_stack_with_per_layer', stacked)
        _set(params, f'{enc_scope}/pair_logits_projection', 'weights',
             _pair_logits_flat(sd, base, n_atom_blocks))

    tr_base = f'{dm}.diffusion_transformer'
    tr_name = 'transformer'
    tr_scope = f'{scope}/{tr_name}'
    tr_stacked = _stack_main_transformer(sd, tr_base, tr_name, n_diff_blocks, n_super_blocks, diff_H, diff_D)
    tr_inner = f'{tr_scope}/__layer_stack_no_per_layer/__layer_stack_no_per_layer'
    _populate_scope(params, tr_inner, tr_stacked)

    _set(params, f'{scope}/single_cond_embedding_norm', 'scale', _get(sd, f'{dm}.layer_norm_s.weight'))
    _set(params, f'{scope}/single_cond_embedding_projection', 'weights', _t(_get(sd, f'{dm}.linear_s.weight')))
    _set(params, f'{scope}/output_norm', 'scale', _get(sd, f'{dm}.layer_norm_a.weight'))


# ─── Top-level conversion entry point ────────────────────────────────────────

def map_of3_to_af3(
    state_dict: dict,
    *,
    n_pairformer_blocks: int = 48,
    n_msa_blocks: int = 4,
    pair_H: int = 4, pair_D: int = 32,
    single_H: int = 16, single_D: int = 24,
    msa_H: int = 8, msa_D: int = 8,
    opm_hidden: int = 32,
    c_z: int = 128, c_s: int = 384,
    n_diff_blocks: int = 24, diff_H: int = 16, diff_D: int = 48,
    n_super_blocks: int = 6,
    n_atom_blocks: int = 3, atom_H: int = 4, atom_D: int = 32,
) -> dict[str, dict[str, np.ndarray]]:
    """Convert an OF3 state dict to AF3 Haiku params dict.

    Returns nested dict {scope: {param_name: np.ndarray}} compatible with
    alphafold3.model.params for saving and loading.
    """
    params: dict[str, dict[str, np.ndarray]] = {}
    map_evoformer_input_embeddings(state_dict, params)
    map_msa_stack(state_dict, params, n_blocks=n_msa_blocks,
                   msa_H=msa_H, msa_D=msa_D, pair_H=pair_H, pair_D=pair_D,
                   opm_hidden=opm_hidden, c_z=c_z)
    map_pairformer_stack(state_dict, params, n_blocks=n_pairformer_blocks,
                          pair_H=pair_H, pair_D=pair_D, single_H=single_H, single_D=single_D)
    map_confidence_head(state_dict, params,
                         pair_H=pair_H, pair_D=pair_D, single_H=single_H, single_D=single_D,
                         c_z=c_z, c_s=c_s)
    map_distogram_head(state_dict, params)
    map_template_embedder(state_dict, params)
    map_evoformer_conditioning(state_dict, params, n_atom_blocks=n_atom_blocks,
                                atom_H=atom_H, atom_D=atom_D)
    map_diffusion_head(state_dict, params, n_diff_blocks=n_diff_blocks,
                        diff_H=diff_H, diff_D=diff_D, n_super_blocks=n_super_blocks,
                        n_atom_blocks=n_atom_blocks, atom_H=atom_H, atom_D=atom_D)
    return params


# ─── Checkpoint I/O ───────────────────────────────────────────────────────────

def load_of3_checkpoint(ckpt_path: Path | str, use_ema: bool = True) -> dict:
    """Load an OF3 checkpoint (.pt file) and return the model state dict.

    Handles three checkpoint formats:
      - Pre-trained model: direct state dict (no 'ema'/'state_dict' keys)
      - Training checkpoint: has 'ema' key with EMA weights
      - DeepSpeed: has 'module' key
      - PyTorch-Lightning: has 'state_dict' key

    Always prefer EMA weights (use_ema=True) for inference quality.
    No OpenFold3 source code dependency.
    """
    import torch
    ckpt_path = Path(ckpt_path)

    if ckpt_path.is_dir():
        raise ValueError(
            f'{ckpt_path} is a directory. DeepSpeed checkpoints require the '
            'openfold3 package. Pass a single .pt file instead.'
        )

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    is_pretrained = 'module' not in ckpt and 'state_dict' not in ckpt and 'ema' not in ckpt
    if is_pretrained:
        state_dict = {'model.' + k: v for k, v in ckpt.items()}
    elif use_ema and 'ema' in ckpt:
        state_dict = {'model.' + k: v for k, v in ckpt['ema']['params'].items()}
    elif 'module' in ckpt:
        state_dict = ckpt['module']
    else:
        state_dict = ckpt.get('state_dict', ckpt)

    # Strip 'model.' prefix
    return {
        (k[len('model.'):] if k.startswith('model.') else k): v
        for k, v in state_dict.items()
    }


def save_af3_params(params: dict[str, dict[str, np.ndarray]], output_dir: Path | str) -> Path:
    """Save converted params in AF3 binary format (.bin.zst).

    Returns the output file path.
    """
    import zstandard
    from alphafold3.model.params import encode_record

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'of3_ported_weights.bin.zst'

    with zstandard.open(output_path, 'wb') as f:
        meta_arr = np.zeros(64, dtype=np.uint8)
        f.write(encode_record('__meta__', '__identifier__', meta_arr))
        for scope, scope_params in sorted(params.items()):
            for name, arr in sorted(scope_params.items()):
                f.write(encode_record(scope, name, np.asarray(arr, dtype=np.float32)))

    return output_path
