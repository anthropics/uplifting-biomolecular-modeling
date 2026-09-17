"""Compare OF3 (PyTorch reference) vs AF3-with-OF3-weights on the pre-diffusion
embeddings, using the real of3-p2-155k checkpoint.

Every quantity below is an *input-boundary* embedding: each one is a linear map
applied to features whose layout differs between the two codebases. Everything
downstream (evoformer trunk, pairformer, diffusion conditioning) is byte-identical
code operating on these tensors, so if all of these agree then the pair features
entering the diffusion module agree.

The same molecule / MSA is expressed twice — once in AF3's feature convention
(via AF3's real featurizers) and once in OF3's — then pushed through the OF3
checkpoint weights directly vs. through the converted AF3 params.

Usage: python3 compare_pair_features.py [checkpoint_path]
"""

import sys

import _paths

_paths.add_import_paths()

import numpy as np
import torch
from alphafold3.constants import mmcif_names
from alphafold3.constants import residue_names
from alphafold3.data import msa_features
from alphafold3.model import of3_weight_converter as conv

CKPT = sys.argv[1] if len(sys.argv) > 1 else _paths.checkpoint()

OF3_ALPHABET = (
    list(residue_names.PROTEIN_TYPES_WITH_UNKNOWN)
    + ['A', 'G', 'C', 'U', 'N']
    + ['DA', 'DG', 'DC', 'DT', 'DN']
    + ['-']
)
AF3_ALPHABET = list(residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP)
N_OF3, N_AF3 = len(OF3_ALPHABET), len(AF3_ALPHABET)
ATOM_COND = 384

# ── A small mixed complex: protein + RNA + DNA + ligand ──────────────────────
# (residue name, chain type) per token, in AF3 naming.
PROTEIN_SEQ = 'MGKWX'          # includes Gly and unknown
RNA_SEQ = 'AGCUN'              # every RNA base + unknown
DNA_SEQ = 'AGCT'               # every DNA base
TOKENS = (
    [(residue_names.PROTEIN_COMMON_ONE_TO_THREE.get(c, 'UNK'),
      mmcif_names.PROTEIN_CHAIN) for c in PROTEIN_SEQ]
    + [(c if c != 'N' else 'N', mmcif_names.RNA_CHAIN) for c in RNA_SEQ]
    + [('D' + c, mmcif_names.DNA_CHAIN) for c in DNA_SEQ]
    + [('UNK', mmcif_names.NON_POLYMER_CHAIN)]      # ligands are UNK / OF3 'X'
)
N_TOKENS = len(TOKENS)

# MSA: query row plus rows exercising gaps and substitutions, per chain.
MSA_ROWS = {
    mmcif_names.PROTEIN_CHAIN: ['MGKWX', 'M-KW-', 'AGKWC', '-----'],
    mmcif_names.RNA_CHAIN: ['AGCUN', 'A-CU-', 'GGCCN', '-----'],
    mmcif_names.DNA_CHAIN: ['AGCT', 'A-CT', 'GGCC', '----'],
}
N_MSA = 4

rng = np.random.default_rng(0)


def of3_restype_index(res_name: str, chain_type: str) -> int:
  if chain_type in mmcif_names.LIGAND_CHAIN_TYPES:
    return OF3_ALPHABET.index('UNK')      # OF3 'X' == AF3 UNK, index 20
  return OF3_ALPHABET.index(res_name)


def af3_restype_index(res_name: str, chain_type: str) -> int:
  # Mirrors features.py: ligands -> UNK, unknown DNA -> unknown nucleic.
  if chain_type in mmcif_names.LIGAND_CHAIN_TYPES:
    res_name = residue_names.UNK
  elif chain_type == mmcif_names.DNA_CHAIN and res_name == residue_names.UNK_DNA:
    res_name = residue_names.UNK_NUCLEIC_ONE_LETTER
  return residue_names.POLYMER_TYPES_ORDER_WITH_UNKNOWN_AND_GAP[res_name]


def build_msa():
  """Returns (af3_msa, of3_msa) integer arrays of shape (N_MSA, N_TOKENS)."""
  af3 = np.full((N_MSA, N_TOKENS), AF3_ALPHABET.index('-'), dtype=np.int64)
  of3 = np.full((N_MSA, N_TOKENS), OF3_ALPHABET.index('-'), dtype=np.int64)

  col = 0
  for chain_type, seqs in MSA_ROWS.items():
    width = len(seqs[0])
    # AF3 side: its real featurizer, which owns the char->class maps.
    af3_block, _ = msa_features.extract_msa_features(
        msa_sequences=seqs, chain_poly_type=chain_type
    )
    af3[:, col : col + width] = af3_block

    # OF3 side: same characters, resolved against OF3's alphabet.
    for row, seq in enumerate(seqs):
      for offset, char in enumerate(seq):
        if char == '-':
          name = '-'
        elif chain_type == mmcif_names.PROTEIN_CHAIN:
          name = residue_names.PROTEIN_COMMON_ONE_TO_THREE.get(char, 'UNK')
        elif chain_type == mmcif_names.RNA_CHAIN:
          name = char if char in ('A', 'G', 'C', 'U') else 'N'
        else:
          name = 'D' + char if char in ('A', 'G', 'C', 'T') else 'DN'
        of3[row, col + offset] = OF3_ALPHABET.index(name)
    col += width
  # The trailing ligand token stays a gap in both.
  return af3, of3


def one_hot(idx, num_classes):
  out = np.zeros(idx.shape + (num_classes,), dtype=np.float32)
  np.put_along_axis(out, idx[..., None], 1.0, axis=-1)
  return out


def build_features():
  af3_msa, of3_msa = build_msa()

  af3_restype = np.array(
      [af3_restype_index(n, t) for n, t in TOKENS], dtype=np.int64
  )
  of3_restype = np.array(
      [of3_restype_index(n, t) for n, t in TOKENS], dtype=np.int64
  )

  # Profiles: MSA class frequencies in each codebase's own alphabet.
  af3_profile = one_hot(af3_msa, N_AF3).mean(axis=0)
  of3_profile = one_hot(of3_msa, N_OF3).mean(axis=0)

  # Layout-independent features, shared verbatim by both sides.
  atom_cond = rng.standard_normal((N_TOKENS, ATOM_COND)).astype(np.float32)
  del_mean = rng.random((N_TOKENS, 1)).astype(np.float32)
  has_del = rng.integers(0, 2, (N_MSA, N_TOKENS, 1)).astype(np.float32)
  del_val = rng.random((N_MSA, N_TOKENS, 1)).astype(np.float32)

  af3 = {
      'target_feat': np.concatenate(
          [one_hot(af3_restype, N_AF3), af3_profile, del_mean, atom_cond],
          axis=-1,
      ),
      # AF3's one-hot has one spare trailing class (num_types + 1).
      'msa_feat': np.concatenate(
          [one_hot(af3_msa, N_AF3 + 1), has_del, del_val], axis=-1
      ),
      'restype': one_hot(af3_restype, N_AF3),
  }
  of3 = {
      's_input': np.concatenate(
          [atom_cond, one_hot(of3_restype, N_OF3), of3_profile, del_mean],
          axis=-1,
      ),
      'msa_feat': np.concatenate(
          [one_hot(of3_msa, N_OF3), has_del, del_val], axis=-1
      ),
      'restype': one_hot(of3_restype, N_OF3),
  }
  return af3, of3


def report(name, af3_out, of3_out, results):
  diff = np.max(np.abs(af3_out - of3_out))
  scale = np.max(np.abs(of3_out))
  results.append((name, diff, scale))
  status = 'MATCH' if diff <= 1e-5 * max(scale, 1.0) else 'MISMATCH'
  print(f'  {status:9s} {name:<34s} max|Δ| = {diff:.3e}   (|x|max = {scale:.3f})')


def main():
  print(f'Loading {CKPT}')
  sd = conv.load_of3_checkpoint(CKPT)
  print(f'  {len(sd)} tensors')
  print('Converting to AF3 params...')
  params = conv.map_of3_to_af3(sd)
  print(f'  {len(params)} param scopes\n')

  def of3_w(key):
    return sd[key].detach().float().numpy() if hasattr(sd[key], 'detach') else np.asarray(sd[key])

  af3, of3 = build_features()
  tf, s_in = af3['target_feat'], of3['s_input']
  results = []

  print('Pre-diffusion embeddings (OF3 reference vs converted AF3 params):')

  # 1. Trunk single init: evoformer/single_activations
  report(
      'single_activations (s_init)',
      tf @ params['diffuser/evoformer/single_activations']['weights'],
      s_in @ of3_w('input_embedder.linear_s.weight').T,
      results,
  )

  # 2. Trunk pair init z[i,j] — the pair features that seed the whole trunk.
  af3_z = (
      (tf @ params['diffuser/evoformer/left_single']['weights'])[:, None, :]
      + (tf @ params['diffuser/evoformer/right_single']['weights'])[None, :, :]
  )
  of3_z = (
      (s_in @ of3_w('input_embedder.linear_z_i.weight').T)[:, None, :]
      + (s_in @ of3_w('input_embedder.linear_z_j.weight').T)[None, :, :]
  )
  report('pair init z_ij (left+right_single)', af3_z, of3_z, results)

  # 3. MSA embedding — the tensor the alphabet bug corrupted.
  report(
      'msa_activations (m)',
      af3['msa_feat'] @ params['diffuser/evoformer/msa_activations']['weights'],
      of3['msa_feat'] @ of3_w('msa_module_embedder.linear_m.weight').T,
      results,
  )

  # 4. MSA module's single-input projection.
  report(
      'extra_msa_target_feat',
      tf @ params['diffuser/evoformer/extra_msa_target_feat']['weights'],
      s_in @ of3_w('msa_module_embedder.linear_s_input.weight').T,
      results,
  )

  # 5. Confidence head pair embedding — where PAE comes from.
  cscope = 'diffuser/confidence_head/~_embed_features'
  af3_cz = (
      (tf @ params[f'{cscope}/left_target_feat_project']['weights'])[None, :, :]
      + (tf @ params[f'{cscope}/right_target_feat_project']['weights'])[:, None, :]
  )
  of3_cz = (
      (s_in @ of3_w('aux_heads.pairformer_embedding.linear_i.weight').T)[:, None, :]
      + (s_in @ of3_w('aux_heads.pairformer_embedding.linear_j.weight').T)[None, :, :]
  )
  report('confidence pair embed (-> PAE)', af3_cz, of3_cz, results)

  # 6. Template aatype pair embedding.
  tscope = (
      'diffuser/evoformer/template_embedding/single_template_embedding/'
      'template_pair_embedding_'
  )
  af3_t = (
      (af3['restype'] @ params[f'{tscope}2']['weights'])[None, :, :]
      + (af3['restype'] @ params[f'{tscope}3']['weights'])[:, None, :]
  )
  tpe = 'template_embedder.template_pair_embedder'
  of3_t = (
      (of3['restype'] @ of3_w(f'{tpe}.aatype_linear_1.weight').T)[:, None, :]
      + (of3['restype'] @ of3_w(f'{tpe}.aatype_linear_2.weight').T)[None, :, :]
  )
  report('template aatype pair embed', af3_t, of3_t, results)

  print()
  bad = [r for r in results if r[1] > 1e-5 * max(r[2], 1.0)]
  if bad:
    print(f'RESULT: {len(bad)}/{len(results)} MISMATCH -> '
          + ', '.join(n for n, _, _ in bad))
    return 1
  print(f'RESULT: all {len(results)} pre-diffusion embeddings match OF3.')
  return 0


if __name__ == '__main__':
  sys.exit(main())
