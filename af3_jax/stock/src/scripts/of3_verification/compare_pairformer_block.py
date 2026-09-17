"""Run one real trunk PairFormer block in BOTH implementations and compare.

This is the check that covers the shape-invariant error classes the structural
audit cannot see:
  * square-matrix transposes (q/k/v/gating/output projections, tri-mul a/b/g/z)
  * SwiGLU gate-vs-value concatenation order
  * the of3-gated column-attention pair-bias transpose

OF3 side:  openfold3.core.model.latent.pairformer.PairFormerBlock loaded from
           `pairformer_stack.blocks.<i>.*` of the real checkpoint.
AF3 side:  alphafold3.model.network.modules.PairFormerIteration with converted
           params sliced out of the layer_stack at the same block index.

Usage: python3 compare_pairformer_block.py [block_idx]
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

jax.config.update('jax_enable_x64', False)

BLOCK = int(sys.argv[1]) if len(sys.argv) > 1 else 0
CKPT = _paths.checkpoint()
N_TOKENS = 7
C_S, C_Z = 384, 128

_paths.no_cuda_autocast()

from alphafold3.model import model_config                      # noqa: E402
from alphafold3.model import of3_weight_converter as conv      # noqa: E402
from alphafold3.model.network import modules                   # noqa: E402
from openfold3.core.model.latent.pairformer import PairFormerBlock  # noqa: E402


def af3_config():
  """Trunk pairformer config + global config with of3_weights enabled."""
  from alphafold3.model import model

  cfg = model.Model.Config()
  gc = cfg.global_config
  gc.of3_weights = True
  gc.bfloat16 = 'none'
  gc.flash_attention_implementation = 'xla'  # no flash attention on CPU
  return cfg.evoformer.pairformer, gc


def build_of3_block(sd):
  block = PairFormerBlock(
      c_s=C_S,
      c_z=C_Z,
      c_hidden_pair_bias=24,
      no_heads_pair_bias=16,
      c_hidden_mul=128,
      c_hidden_pair_att=32,
      no_heads_pair=4,
      transition_type='swiglu',
      transition_n=4,
      pair_dropout=0.25,
      fuse_projection_weights=False,
      inf=1e8,
  )
  prefix = f'pairformer_stack.blocks.{BLOCK}.'
  sub = {
      k[len(prefix):]: v.detach().float()
      for k, v in sd.items()
      if k.startswith(prefix)
  }
  missing, unexpected = block.load_state_dict(sub, strict=False)
  # dropout/LayerNorm buffers may legitimately be absent; report anything else.
  missing = [m for m in missing if 'dropout' not in m]
  if missing or unexpected:
    print(f'  OF3 load_state_dict: missing={missing[:5]} unexpected={unexpected[:5]}')
  block.eval()
  return block


def af3_block_params(params):
  """Slice block BLOCK out of the trunk pairformer layer_stack."""
  prefix = next(
      k.split('trunk_pairformer')[0]
      for k in params
      if k.endswith('trunk_pairformer/pair_attention1/act_norm')
  )
  out = {}
  for scope, entries in params.items():
    if not scope.startswith(prefix + 'trunk_pairformer'):
      continue
    new_scope = scope[len(prefix):]
    out[new_scope] = {
        name: jnp.asarray(arr[BLOCK]) for name, arr in entries.items()
    }
  return out


def main():
  print(f'Loading {CKPT}  (block {BLOCK})')
  sd = conv.load_of3_checkpoint(CKPT)
  params = conv.map_of3_to_af3(sd)

  rng = np.random.default_rng(0)
  z = rng.standard_normal((N_TOKENS, N_TOKENS, C_Z)).astype(np.float32) * 0.5
  s = rng.standard_normal((N_TOKENS, C_S)).astype(np.float32) * 0.5
  pair_mask = np.ones((N_TOKENS, N_TOKENS), dtype=np.float32)
  seq_mask = np.ones((N_TOKENS,), dtype=np.float32)

  # ── OF3 reference ──
  of3 = build_of3_block(sd)
  with torch.no_grad():
    of3_s, of3_z = of3(
        s=torch.from_numpy(s.copy()),
        z=torch.from_numpy(z.copy()),
        single_mask=torch.from_numpy(seq_mask.copy()),
        pair_mask=torch.from_numpy(pair_mask.copy()),
    )
  of3_s = of3_s.numpy()
  of3_z = of3_z.numpy()

  # ── AF3 with converted params ──
  cfg, gc = af3_config()

  def fwd(act, pmask, single, smask):
    return modules.PairFormerIteration(
        cfg, gc, with_single=True, name='trunk_pairformer'
    )(act, pmask, single, smask)

  transformed = hk.without_apply_rng(hk.transform(fwd))
  block_params = af3_block_params(params)
  if os.environ.get('SABOTAGE'):
    tgt = 'trunk_pairformer/pair_attention1/gating_query'
    block_params[tgt] = {'weights': block_params[tgt]['weights'].T}
    print('  [sabotage] transposed', tgt)
  af3_z, af3_s = transformed.apply(
      block_params, jnp.asarray(z), jnp.asarray(pair_mask),
      jnp.asarray(s), jnp.asarray(seq_mask),
  )
  af3_z, af3_s = np.asarray(af3_z), np.asarray(af3_s)

  print('\nPairFormer block outputs (OF3 reference vs AF3 + converted params):')
  ok = True
  for name, a, b in (('pair z', af3_z, of3_z), ('single s', af3_s, of3_s)):
    diff = np.max(np.abs(a - b))
    scale = np.max(np.abs(b))
    rel = diff / max(scale, 1e-9)
    good = rel < 2e-3
    ok &= good
    print(f'  {"MATCH" if good else "MISMATCH":9s} {name:<10s} '
          f'max|Δ| = {diff:.3e}   |x|max = {scale:.3f}   rel = {rel:.2e}')

  # A correlation check catches "structured but scaled" disagreement that a
  # tolerance on max|Δ| alone might let through.
  for name, a, b in (('pair z', af3_z, of3_z), ('single s', af3_s, of3_s)):
    corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    print(f'            {name:<10s} pearson r = {corr:.8f}')

  print('\nRESULT:', 'block matches' if ok else 'BLOCK MISMATCH')
  return 0 if ok else 1


if __name__ == '__main__':
  sys.exit(main())
