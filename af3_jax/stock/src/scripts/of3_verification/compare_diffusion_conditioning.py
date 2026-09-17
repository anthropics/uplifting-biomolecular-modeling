"""Compare the diffusion conditioning module (Algorithm 21) in both codebases.

Covers, in one shot:
  * the relative-position encoding feature LAYOUT (rel_pos | rel_token |
    same_entity | rel_chain) — it is concatenated with the trunk pair rep and
    fed to one linear, so a layout mismatch would show up here
  * pair_cond / single_cond initial LayerNorm + projection
  * the two pair transitions and two single transitions (the _swiglu_flat path)
  * the Fourier noise embedding parameters converted from OF3 buffers
  * the target_feat features_1d block reordering (_reorder_features_1d)

Usage: python3 compare_diffusion_conditioning.py    SABOTAGE=1 to self-test
"""

import dataclasses
import os
import sys

import _paths

_paths.add_import_paths(require_openfold3=True)

import haiku as hk
import jax.numpy as jnp
import numpy as np
import torch

_paths.no_cuda_autocast()

from alphafold3.model import of3_weight_converter as conv        # noqa: E402
from alphafold3.model.network import diffusion_head              # noqa: E402
from openfold3.core.model.layers.diffusion_conditioning import (  # noqa: E402
    DiffusionConditioning,
)

CKPT = _paths.checkpoint()
N = 6
C_S_INPUT, C_S, C_Z = 449, 384, 128
SIGMA_DATA = 16.0

# Two chains so asym/entity/sym distinctions are exercised, and a repeated
# residue_index so rel_token's same_res condition fires.
ASYM = np.array([1, 1, 1, 2, 2, 2], dtype=np.int32)
ENTITY = np.array([1, 1, 1, 1, 1, 1], dtype=np.int32)
SYM = np.array([1, 1, 1, 2, 2, 2], dtype=np.int32)
RES_IDX = np.array([1, 2, 2, 1, 2, 3], dtype=np.int32)
TOK_IDX = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)


@dataclasses.dataclass
class _TokenFeatures:
  residue_index: np.ndarray
  token_index: np.ndarray
  asym_id: np.ndarray
  entity_id: np.ndarray
  sym_id: np.ndarray


@dataclasses.dataclass
class _Batch:
  token_features: _TokenFeatures


def main():
  print(f'Loading {CKPT}')
  sd = conv.load_of3_checkpoint(CKPT)
  params = conv.map_of3_to_af3(sd)

  rng = np.random.default_rng(11)
  s_trunk = rng.standard_normal((N, C_S)).astype(np.float32) * 0.5
  z_trunk = rng.standard_normal((N, N, C_Z)).astype(np.float32) * 0.5
  s_input_of3 = rng.standard_normal((N, C_S_INPUT)).astype(np.float32) * 0.5
  noise_level = np.array(2.0, dtype=np.float32)

  # AF3's target_feat is the same features in AF3 block order. Only the atom
  # conditioning block is populated so the residue permutation (already tested
  # elsewhere) stays out of this comparison.
  target_feat = np.zeros((N, 447), dtype=np.float32)
  target_feat[:, 2 * 31 + 1:] = s_input_of3[:, :384]
  of3_s_input = np.zeros_like(s_input_of3)
  of3_s_input[:, :384] = s_input_of3[:, :384]

  # ── OF3 reference ──
  of3 = DiffusionConditioning(
      c_s_input=C_S_INPUT, c_s=C_S, c_z=C_Z, sigma_data=SIGMA_DATA,
      c_fourier_emb=256, max_relative_idx=32, max_relative_chain=2,
      seed_fourier_emb=42,
  )
  prefix = 'diffusion_module.diffusion_conditioning.'
  sub = {k[len(prefix):]: v.detach().float()
         for k, v in sd.items() if k.startswith(prefix)}
  missing, unexpected = of3.load_state_dict(sub, strict=False)
  if missing or unexpected:
    print(f'  OF3 load: missing={missing[:5]} unexpected={unexpected[:5]}')
  of3.eval()
  batch_of3 = {
      'residue_index': torch.from_numpy(RES_IDX.astype(np.int64)),
      'token_index': torch.from_numpy(TOK_IDX.astype(np.int64)),
      'asym_id': torch.from_numpy(ASYM.astype(np.int64)),
      'entity_id': torch.from_numpy(ENTITY.astype(np.int64)),
      'sym_id': torch.from_numpy(SYM.astype(np.int64)),
      'token_mask': torch.ones(N),
  }
  with torch.no_grad():
    of3_s, of3_z = of3(
        batch=batch_of3,
        t=torch.tensor(float(noise_level)),
        si_input=torch.from_numpy(of3_s_input.copy()),
        si_trunk=torch.from_numpy(s_trunk.copy()),
        zij_trunk=torch.from_numpy(z_trunk.copy()),
        use_conditioning=True,
    )
  of3_s, of3_z = of3_s.numpy(), of3_z.numpy()

  # ── AF3 with converted params ──
  from alphafold3.model import model

  cfg_full = model.Model.Config()
  gc = cfg_full.global_config
  gc.of3_weights = True
  gc.bfloat16 = 'none'
  gc.flash_attention_implementation = 'xla'
  head_cfg = cfg_full.heads.diffusion

  batch = _Batch(_TokenFeatures(RES_IDX, TOK_IDX, ASYM, ENTITY, SYM))

  def fwd(single, pair, tfeat, nl):
    head = diffusion_head.DiffusionHead(head_cfg, gc)
    return head._conditioning(  # pylint: disable=protected-access
        batch=batch,
        embeddings={'single': single, 'pair': pair, 'target_feat': tfeat},
        noise_level=nl,
        use_conditioning=True,
    )

  transformed = hk.without_apply_rng(hk.transform(fwd))
  strip = 'diffuser/~/diffusion_head/'
  block_params = {
      k[len(strip):]: {n: jnp.asarray(v) for n, v in e.items()}
      for k, e in params.items() if k.startswith(strip)
  }
  # In the real model hk.get_parameter inside the transparent _conditioning
  # attaches to the enclosing DiffusionHead scope; standalone it lands on '~'.
  head_scope = params['diffuser/~/diffusion_head']
  block_params['~'] = {
      'fourier_embedding_weight': jnp.asarray(head_scope['fourier_embedding_weight']),
      'fourier_embedding_bias': jnp.asarray(head_scope['fourier_embedding_bias']),
  }

  if os.environ.get('SABOTAGE'):
    # Rotate the relpos columns to prove the feature layout is actually tested.
    tgt = 'pair_cond_initial_projection'
    w = np.asarray(block_params[tgt]['weights'])
    w = np.concatenate([w[:C_Z], np.roll(w[C_Z:], 1, axis=0)], axis=0)
    block_params[tgt] = {'weights': jnp.asarray(w)}
    print('  [sabotage] rolled relpos rows of', tgt)

  af3_s, af3_z = transformed.apply(
      block_params, jnp.asarray(s_trunk), jnp.asarray(z_trunk),
      jnp.asarray(target_feat), jnp.asarray(noise_level),
  )

  print('\nDiffusion conditioning outputs:')
  ok = True
  for label, a, b in (('single_cond', np.asarray(af3_s), of3_s),
                      ('pair_cond', np.asarray(af3_z), of3_z)):
    diff = np.max(np.abs(a - b))
    scale = np.max(np.abs(b))
    rel = diff / max(scale, 1e-9)
    corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    good = rel < 2e-3
    ok &= good
    print(f'  {"MATCH" if good else "MISMATCH":9s} {label:<12s} max|Δ| = {diff:.3e}'
          f'   |x|max = {scale:.3f}   rel = {rel:.2e}   r = {corr:.8f}')
  print('\nRESULT:', 'matches' if ok else 'MISMATCH')
  return 0 if ok else 1


if __name__ == '__main__':
  sys.exit(main())
