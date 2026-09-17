"""Same idea as compare_pairformer_block.py, for the MSA module block.

Covers OuterProductMean, MSA row attention with pair bias, the MSA transition,
and the pair stack inside the MSA block.

Usage: python3 compare_msa_block.py [block_idx]      SABOTAGE=1 to self-test
"""

import os
import sys

import _paths

_paths.add_import_paths(require_openfold3=True)

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import torch

_paths.no_cuda_autocast()

from alphafold3.model import of3_weight_converter as conv      # noqa: E402
from alphafold3.model.network import modules                   # noqa: E402
from openfold3.core.model.latent.msa_module import MSAModuleBlock  # noqa: E402

BLOCK = int(sys.argv[1]) if len(sys.argv) > 1 else 0
CKPT = _paths.checkpoint()
N_TOKENS, N_MSA = 7, 5
C_M, C_Z = 64, 128


def af3_config():
  from alphafold3.model import model

  cfg = model.Model.Config()
  gc = cfg.global_config
  gc.of3_weights = True
  gc.bfloat16 = 'none'
  gc.flash_attention_implementation = 'xla'
  return cfg.evoformer.msa_stack, gc


def build_of3_block(sd):
  block = MSAModuleBlock(
      c_m=C_M,
      c_z=C_Z,
      c_hidden_msa_att=8,
      c_hidden_opm=32,
      c_hidden_mul=128,
      c_hidden_pair_att=32,
      no_heads_msa=8,
      no_heads_pair=4,
      transition_type='swiglu',
      transition_n=4,
      msa_dropout=0.15,
      pair_dropout=0.25,
      opm_first=True,
      fuse_projection_weights=False,
      inf=1e8,
      eps=1e-8,
      last_block=(BLOCK == 3),  # OF3 drops msa_att_row/msa_transition in the last block
  )
  prefix = f'msa_module.blocks.{BLOCK}.'
  sub = {k[len(prefix):]: v.detach().float()
         for k, v in sd.items() if k.startswith(prefix)}
  missing, unexpected = block.load_state_dict(sub, strict=False)
  missing = [m for m in missing if 'dropout' not in m]
  if missing or unexpected:
    print(f'  OF3 load_state_dict: missing={missing[:6]} unexpected={unexpected[:6]}')
  block.eval()
  return block


def af3_block_params(params, marker, module_name):
  prefix = next(k.split(module_name)[0] for k in params if k.endswith(marker))
  return {
      scope[len(prefix):]: {n: jnp.asarray(a[BLOCK]) for n, a in entries.items()}
      for scope, entries in params.items()
      if scope.startswith(prefix + module_name)
  }


def main():
  print(f'Loading {CKPT}  (MSA block {BLOCK})')
  sd = conv.load_of3_checkpoint(CKPT)
  params = conv.map_of3_to_af3(sd)

  rng = np.random.default_rng(1)
  m = rng.standard_normal((N_MSA, N_TOKENS, C_M)).astype(np.float32) * 0.5
  z = rng.standard_normal((N_TOKENS, N_TOKENS, C_Z)).astype(np.float32) * 0.5
  msa_mask = np.ones((N_MSA, N_TOKENS), dtype=np.float32)
  pair_mask = np.ones((N_TOKENS, N_TOKENS), dtype=np.float32)

  of3 = build_of3_block(sd)
  with torch.no_grad():
    of3_m, of3_z = of3(
        m=torch.from_numpy(m.copy()),
        z=torch.from_numpy(z.copy()),
        msa_mask=torch.from_numpy(msa_mask.copy()),
        pair_mask=torch.from_numpy(pair_mask.copy()),
    )
  of3_m, of3_z = of3_m.numpy(), of3_z.numpy()

  cfg, gc = af3_config()

  def fwd(msa, pair, msa_m, pair_m):
    return modules.EvoformerIteration(cfg, gc, name='msa_stack')(
        activations={'msa': msa, 'pair': pair},
        masks={'msa': msa_m, 'pair': pair_m},
    )

  transformed = hk.without_apply_rng(hk.transform(fwd))
  block_params = af3_block_params(
      params, 'msa_stack/outer_product_mean/left_projection', 'msa_stack'
  )
  if os.environ.get('SABOTAGE'):
    tgt = 'msa_stack/outer_product_mean/left_projection'
    block_params[tgt] = {'weights': block_params[tgt]['weights'] * 1.05}
    print('  [sabotage] scaled', tgt)

  out = transformed.apply(
      block_params, jnp.asarray(m), jnp.asarray(z),
      jnp.asarray(msa_mask), jnp.asarray(pair_mask),
  )
  af3_m, af3_z = np.asarray(out['msa']), np.asarray(out['pair'])

  print('\nMSA block outputs (OF3 reference vs AF3 + converted params):')
  ok = True
  for name, a, b in (('msa m', af3_m, of3_m), ('pair z', af3_z, of3_z)):
    diff = np.max(np.abs(a - b))
    scale = np.max(np.abs(b))
    rel = diff / max(scale, 1e-9)
    corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    good = rel < 2e-3
    ok &= good
    print(f'  {"MATCH" if good else "MISMATCH":9s} {name:<8s} max|Δ| = {diff:.3e}'
          f'   |x|max = {scale:.3f}   rel = {rel:.2e}   r = {corr:.8f}')

  print('\nRESULT:', 'block matches' if ok else 'BLOCK MISMATCH')
  return 0 if ok else 1


if __name__ == '__main__':
  sys.exit(main())
