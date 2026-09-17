"""Compare the full 24-block diffusion transformer in both implementations.

This is the most heavily OF3-branched module in the AF3 code: per-block pair
LayerNorm + projection instead of AF3's single shared pre-stack LN, plus the
adaLN single conditioning and the flat SwiGLU conversion (_swiglu_flat) that is
a separate code path from the pairformer's SwiGLU helper.

Usage: python3 compare_diffusion_transformer.py     SABOTAGE=1 to self-test
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

from alphafold3.model import of3_weight_converter as conv        # noqa: E402
from alphafold3.model.network import diffusion_transformer       # noqa: E402
from openfold3.core.model.layers.diffusion_transformer import (  # noqa: E402
    DiffusionTransformer,
)

CKPT = _paths.checkpoint()
N_TOKENS = 6
C_A, C_S, C_Z = 768, 384, 128


def af3_config():
  from alphafold3.model import model

  cfg = model.Model.Config()
  gc = cfg.global_config
  gc.of3_weights = True
  gc.bfloat16 = 'none'
  gc.flash_attention_implementation = 'xla'
  return cfg.heads.diffusion.transformer, gc


def main():
  print(f'Loading {CKPT}')
  sd = conv.load_of3_checkpoint(CKPT)
  params = conv.map_of3_to_af3(sd)

  rng = np.random.default_rng(2)
  a = rng.standard_normal((N_TOKENS, C_A)).astype(np.float32) * 0.5
  s = rng.standard_normal((N_TOKENS, C_S)).astype(np.float32) * 0.5
  z = rng.standard_normal((N_TOKENS, N_TOKENS, C_Z)).astype(np.float32) * 0.5
  mask = np.ones((N_TOKENS,), dtype=np.float32)

  # ── OF3 reference ──
  of3 = DiffusionTransformer(
      c_a=C_A, c_s=C_S, c_z=C_Z, c_hidden=48, no_heads=16, no_blocks=24,
      n_transition=2, use_ada_layer_norm=True, n_query=None, n_key=None,
      inf=1e8,
  )
  prefix = 'diffusion_module.diffusion_transformer.'
  sub = {k[len(prefix):]: v.detach().float()
         for k, v in sd.items() if k.startswith(prefix)}
  missing, unexpected = of3.load_state_dict(sub, strict=False)
  if missing or unexpected:
    print(f'  OF3 load: missing={missing[:5]} unexpected={unexpected[:5]}')
  of3.eval()
  with torch.no_grad():
    of3_out = of3(
        a=torch.from_numpy(a.copy()),
        s=torch.from_numpy(s.copy()),
        z=torch.from_numpy(z.copy()),
        mask=torch.from_numpy(mask.copy()),
    ).numpy()

  # ── AF3 with converted params ──
  cfg, gc = af3_config()
  print(f'  AF3 config: num_blocks={cfg.num_blocks} '
        f'super_block_size={cfg.super_block_size}')

  def fwd(act, m, single_cond, pair_cond):
    return diffusion_transformer.Transformer(cfg, gc, name='transformer')(
        act, m, single_cond, pair_cond
    )

  transformed = hk.without_apply_rng(hk.transform(fwd))
  scope_prefix = 'diffuser/~/diffusion_head/'
  block_params = {
      k[len(scope_prefix):]: {n: jnp.asarray(v) for n, v in e.items()}
      for k, e in params.items()
      if k.startswith(scope_prefix + 'transformer')
  }
  if os.environ.get('SABOTAGE'):
    tgt = next(k for k in block_params if k.endswith('transformerq_projection'))
    w = block_params[tgt]['weights']
    block_params[tgt] = {**block_params[tgt], 'weights': w * 1.02}
    print('  [sabotage] scaled', tgt)

  af3_out = np.asarray(transformed.apply(
      block_params, jnp.asarray(a), jnp.asarray(mask),
      jnp.asarray(s), jnp.asarray(z),
  ))

  diff = np.max(np.abs(af3_out - of3_out))
  scale = np.max(np.abs(of3_out))
  rel = diff / max(scale, 1e-9)
  corr = np.corrcoef(af3_out.ravel(), of3_out.ravel())[0, 1]
  good = rel < 2e-3
  print('\nDiffusion transformer output (24 blocks):')
  print(f'  {"MATCH" if good else "MISMATCH":9s} a   max|Δ| = {diff:.3e}'
        f'   |x|max = {scale:.3f}   rel = {rel:.2e}   r = {corr:.8f}')
  print('\nRESULT:', 'matches' if good else 'MISMATCH')
  return 0 if good else 1


if __name__ == '__main__':
  sys.exit(main())
