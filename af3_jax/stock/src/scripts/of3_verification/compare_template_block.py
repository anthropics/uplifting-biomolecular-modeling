"""Compare a template pair-stack block (pair-only PairFormerIteration, c_z=64).

The template stack is the one place _pairblock_params is called with a different
tri_mul_hidden argument and with_single=False, so it gets its own check.

Usage: python3 compare_template_block.py [block_idx]   SABOTAGE=1 to self-test
"""

import os
import sys

import _paths

_paths.add_import_paths(require_openfold3=True)

import haiku as hk
import jax.numpy as jnp
import numpy as np
import torch

_paths.no_cuda_autocast()

from alphafold3.model import of3_weight_converter as conv   # noqa: E402
from alphafold3.model.network import modules                # noqa: E402
from openfold3.core.model.latent.base_blocks import PairBlock  # noqa: E402

BLOCK = int(sys.argv[1]) if len(sys.argv) > 1 else 0
CKPT = _paths.checkpoint()
N_TOKENS, C_Z = 7, 64


def main():
  print(f'Loading {CKPT}  (template block {BLOCK})')
  sd = conv.load_of3_checkpoint(CKPT)
  params = conv.map_of3_to_af3(sd)

  rng = np.random.default_rng(3)
  z = rng.standard_normal((N_TOKENS, N_TOKENS, C_Z)).astype(np.float32) * 0.5
  pair_mask = np.ones((N_TOKENS, N_TOKENS), dtype=np.float32)

  # ── OF3 reference: the template pair stack block is a bare PairBlock ──
  of3 = PairBlock(
      c_z=C_Z,
      c_hidden_mul=64,
      c_hidden_pair_att=16,
      no_heads_pair=4,
      transition_type='swiglu',
      transition_n=2,
      pair_dropout=0.25,
      fuse_projection_weights=False,
      inf=1e8,
  )
  prefix = f'template_embedder.template_pair_stack.blocks.{BLOCK}.'
  sub = {k[len(prefix):]: v.detach().float()
         for k, v in sd.items() if k.startswith(prefix)}
  missing, unexpected = of3.load_state_dict(sub, strict=False)
  missing = [m for m in missing if 'dropout' not in m]
  if missing or unexpected:
    print(f'  OF3 load: missing={missing[:6]} unexpected={unexpected[:6]}')
  of3.eval()
  with torch.no_grad():
    of3_z = of3(
        z=torch.from_numpy(z.copy()),
        pair_mask=torch.from_numpy(pair_mask.copy()),
    ).numpy()

  # ── AF3 with converted params ──
  from alphafold3.model import model
  from alphafold3.model.network import template_modules

  gc = model.Model.Config().global_config
  gc.of3_weights = True
  gc.bfloat16 = 'none'
  gc.flash_attention_implementation = 'xla'
  cfg = template_modules.TemplateEmbedding.Config().template_stack

  def fwd(act, pmask):
    return modules.PairFormerIteration(
        cfg, gc, name='template_embedding_iteration'
    )(act, pmask)

  transformed = hk.without_apply_rng(hk.transform(fwd))
  name = 'template_embedding_iteration'
  prefix_scope = next(k.split(name)[0] for k in params if name in k)
  block_params = {
      k[len(prefix_scope):]: {n: jnp.asarray(v[BLOCK]) for n, v in e.items()}
      for k, e in params.items()
      if k.startswith(prefix_scope + name)
  }
  if os.environ.get('SABOTAGE'):
    tgt = f'{name}/pair_attention2/output_projection'
    block_params[tgt] = {'weights': block_params[tgt]['weights'].T}
    print('  [sabotage] transposed', tgt)

  af3_z = np.asarray(transformed.apply(
      block_params, jnp.asarray(z), jnp.asarray(pair_mask)
  ))

  diff = np.max(np.abs(af3_z - of3_z))
  scale = np.max(np.abs(of3_z))
  rel = diff / max(scale, 1e-9)
  corr = np.corrcoef(af3_z.ravel(), of3_z.ravel())[0, 1]
  good = rel < 2e-3
  print('\nTemplate pair-stack block output:')
  print(f'  {"MATCH" if good else "MISMATCH":9s} pair z  max|Δ| = {diff:.3e}'
        f'   |x|max = {scale:.3f}   rel = {rel:.2e}   r = {corr:.8f}')
  print('\nRESULT:', 'matches' if good else 'MISMATCH')
  return 0 if good else 1


if __name__ == '__main__':
  sys.exit(main())
