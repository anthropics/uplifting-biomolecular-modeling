"""Close the coverage hole: keys probed with _has() but whose values are never
read via _get() would have counted as 'accessed'. Track them separately.

Also enumerate the shape-invariant risk surface: square weight matrices, where
a missing or spurious transpose cannot be caught by any shape check.
"""

import collections
import sys

import _paths

_paths.add_import_paths()

from alphafold3.model import of3_weight_converter as conv

got, probed = set(), set()
_orig_get, _orig_has = conv._get, conv._has
conv._get = lambda sd, k: (got.add(k), _orig_get(sd, k))[1]
conv._has = lambda sd, k: (probed.add(k), _orig_has(sd, k))[1]

sd = conv.load_of3_checkpoint(_paths.checkpoint())
conv.map_of3_to_af3(sd)

real = set(sd)
never_read = sorted(real - got)
probed_not_read = sorted((real & probed) - got)

print(f'tensors: {len(real)}   read via _get: {len(real & got)}')
print(f'never read: {len(never_read)}')
print(f'  of those, probed by _has but value never used: {len(probed_not_read)}')

non_dup = [k for k in never_read if not k.startswith('sample_diffusion.')]
print(f'\nnever-read tensors outside the duplicated sample_diffusion subtree: '
      f'{len(non_dup)}')
for k in non_dup:
  print(f'  {k}  {tuple(sd[k].shape)}')

probed_non_dup = [k for k in probed_not_read if not k.startswith('sample_diffusion.')]
print(f'\nprobed-but-unused outside sample_diffusion: {len(probed_non_dup)}')
for k in probed_non_dup:
  print(f'  {k}  {tuple(sd[k].shape)}')

print('\n=== square weight matrices (transpose errors are shape-invisible) ===')
squares = collections.Counter()
for k, v in sd.items():
  if k.startswith('sample_diffusion.'):
    continue
  shape = tuple(v.shape)
  if len(shape) == 2 and shape[0] == shape[1]:
    squares['.'.join('N' if p.isdigit() else p for p in k.split('.'))] = shape
print(f'{len(squares)} distinct square-matrix patterns:')
for pattern, shape in sorted(squares.items()):
  print(f'  {pattern}   {shape}')
