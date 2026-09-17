"""Two questions about unread checkpoint tensors:

1. Are any of them NOT part of the duplicated `sample_diffusion.*` subtree?
   Those would be genuinely dropped weights.
2. Is `sample_diffusion.diffusion_module.X` numerically identical to
   `diffusion_module.X`? OF3 uses the sample_diffusion copy at inference, so if
   they differ the converter is reading the wrong one.
"""

import collections
import sys

import _paths

_paths.add_import_paths()

import numpy as np
from alphafold3.model import of3_weight_converter as conv

accessed = set()
_orig_get, _orig_has = conv._get, conv._has
conv._get = lambda sd, k: (accessed.add(k), _orig_get(sd, k))[1]
conv._has = lambda sd, k: (accessed.add(k), _orig_has(sd, k))[1]

sd = conv.load_of3_checkpoint(_paths.checkpoint())
conv.map_of3_to_af3(sd)
unconsumed = sorted(set(sd) - accessed)

print(f'total tensors: {len(sd)}   unread: {len(unconsumed)}')

other = [k for k in unconsumed if not k.startswith('sample_diffusion.')]
print(f'\n=== unread and NOT under sample_diffusion.: {len(other)} ===')
groups = collections.Counter(
    '.'.join('N' if p.isdigit() else p for p in k.split('.')) for k in other
)
for pattern, count in sorted(groups.items()):
  example = next(
      k for k in other
      if '.'.join('N' if p.isdigit() else p for p in k.split('.')) == pattern
  )
  print(f'  x{count:<4d} {pattern}   {tuple(sd[example].shape)}')

print('\n=== sample_diffusion.X vs X: numerically identical? ===')
pairs = 0
differing = []
for key in sd:
  if not key.startswith('sample_diffusion.'):
    continue
  base = key[len('sample_diffusion.'):]
  if base not in sd:
    differing.append((key, 'NO COUNTERPART'))
    continue
  pairs += 1
  a = sd[key].detach().float().numpy()
  b = sd[base].detach().float().numpy()
  if a.shape != b.shape:
    differing.append((key, f'shape {a.shape} vs {b.shape}'))
  else:
    d = float(np.max(np.abs(a - b)))
    if d > 0:
      differing.append((key, f'max|Δ|={d:.3e}'))

print(f'  compared {pairs} pairs')
if differing:
  print(f'  DIFFERING: {len(differing)}')
  for k, why in differing[:20]:
    print(f'    {k}: {why}')
else:
  print('  all identical -> the sample_diffusion subtree is a redundant copy')
