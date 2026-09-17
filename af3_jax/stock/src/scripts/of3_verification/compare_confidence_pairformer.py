"""Compare a confidence-head pairformer block (the PAE path) in both codebases.

Same classes as the trunk pairformer but a different checkpoint prefix and a
different converted scope, so it gets its own numerical check.

Usage: python3 compare_confidence_pairformer.py [block_idx]  SABOTAGE=1 self-test
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
from openfold3.core.model.latent.pairformer import PairFormerBlock  # noqa: E402

BLOCK = int(sys.argv[1]) if len(sys.argv) > 1 else 0
CKPT = _paths.checkpoint()
N_TOKENS, C_S, C_Z = 7, 384, 128


def main():
  print(f'Loading {CKPT}  (confidence pairformer block {BLOCK})')
  sd = conv.load_of3_checkpoint(CKPT)
  params = conv.map_of3_to_af3(sd)

  rng = np.random.default_rng(4)
  z = rng.standard_normal((N_TOKENS, N_TOKENS, C_Z)).astype(np.float32) * 0.5
  s = rng.standard_normal((N_TOKENS, C_S)).astype(np.float32) * 0.5
  pair_mask = np.ones((N_TOKENS, N_TOKENS), dtype=np.float32)
  seq_mask = np.ones((N_TOKENS,), dtype=np.float32)

  of3 = PairFormerBlock(
      c_s=C_S, c_z=C_Z, c_hidden_pair_bias=24, no_heads_pair_bias=16,
      c_hidden_mul=128, c_hidden_pair_att=32, no_heads_pair=4,
      transition_type='swiglu', transition_n=4, pair_dropout=0.25,
      fuse_projection_weights=False, inf=1e8,
  )
  prefix = f'aux_heads.pairformer_embedding.pairformer_stack.blocks.{BLOCK}.'
  sub = {k[len(prefix):]: v.detach().float()
         for k, v in sd.items() if k.startswith(prefix)}
  missing, unexpected = of3.load_state_dict(sub, strict=False)
  missing = [m for m in missing if 'dropout' not in m]
  if missing or unexpected:
    print(f'  OF3 load: missing={missing[:5]} unexpected={unexpected[:5]}')
  of3.eval()
  with torch.no_grad():
    of3_s, of3_z = of3(
        s=torch.from_numpy(s.copy()), z=torch.from_numpy(z.copy()),
        single_mask=torch.from_numpy(seq_mask.copy()),
        pair_mask=torch.from_numpy(pair_mask.copy()),
    )
  of3_s, of3_z = of3_s.numpy(), of3_z.numpy()

  from alphafold3.model import model
  from alphafold3.model.network import confidence_head

  gc = model.Model.Config().global_config
  gc.of3_weights = True
  gc.bfloat16 = 'none'
  gc.flash_attention_implementation = 'xla'
  cfg = confidence_head.ConfidenceHead.Config().pairformer

  def fwd(act, pmask, single, smask):
    return modules.PairFormerIteration(
        cfg, gc, with_single=True, name='confidence_pairformer'
    )(act, pmask, single, smask)

  transformed = hk.without_apply_rng(hk.transform(fwd))
  name = 'confidence_pairformer'
  scope_prefix = next(k.split(name)[0] for k in params if name in k)
  block_params = {
      k[len(scope_prefix):]: {n: jnp.asarray(v[BLOCK]) for n, v in e.items()}
      for k, e in params.items() if k.startswith(scope_prefix + name)
  }
  if os.environ.get('SABOTAGE'):
    tgt = f'{name}/single_attention_gating_query'
    block_params[tgt] = {'weights': block_params[tgt]['weights'].T}
    print('  [sabotage] transposed', tgt)

  af3_z, af3_s = transformed.apply(
      block_params, jnp.asarray(z), jnp.asarray(pair_mask),
      jnp.asarray(s), jnp.asarray(seq_mask),
  )

  ok = True
  print('\nConfidence pairformer block outputs:')
  for label, a, b in (('pair z', np.asarray(af3_z), of3_z),
                      ('single s', np.asarray(af3_s), of3_s)):
    diff = np.max(np.abs(a - b))
    scale = np.max(np.abs(b))
    rel = diff / max(scale, 1e-9)
    corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    good = rel < 2e-3
    ok &= good
    print(f'  {"MATCH" if good else "MISMATCH":9s} {label:<9s} max|Δ| = {diff:.3e}'
          f'   |x|max = {scale:.3f}   rel = {rel:.2e}   r = {corr:.8f}')
  print('\nRESULT:', 'matches' if ok else 'MISMATCH')
  return 0 if ok else 1


if __name__ == '__main__':
  sys.exit(main())
