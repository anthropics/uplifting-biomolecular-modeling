"""Structural audit of the OF3 -> AF3 conversion.

1. Which OF3 checkpoint tensors does the converter never read? (unconsumed
   weights = something the AF3 side is silently getting from its own init)
2. How does the converted param tree compare to REAL AF3 native params —
   missing scopes, extra scopes, shape mismatches?
"""

import collections
import sys

import _paths

_paths.add_import_paths()

import numpy as np
from alphafold3.model import of3_weight_converter as conv

OF3_CKPT = _paths.checkpoint()
AF3_NATIVE = _paths.af3_native_dir()

# ── 1. instrument _get/_has to record every checkpoint key touched ────────────
accessed = set()
_orig_get, _orig_has = conv._get, conv._has


def _get_logged(sd, key):
  accessed.add(key)
  return _orig_get(sd, key)


def _has_logged(sd, key):
  accessed.add(key)
  return _orig_has(sd, key)


conv._get, conv._has = _get_logged, _has_logged

print(f'Loading OF3 checkpoint {OF3_CKPT}')
sd = conv.load_of3_checkpoint(OF3_CKPT)
params = conv.map_of3_to_af3(sd)
print(f'  {len(sd)} checkpoint tensors, {len(params)} converted scopes')

unconsumed = sorted(set(sd) - accessed)
print(f'\n=== 1. Checkpoint tensors never read: {len(unconsumed)} ===')
groups = collections.Counter()
for key in unconsumed:
  # Collapse block indices so the report is readable.
  parts = [p if not p.isdigit() else 'N' for p in key.split('.')]
  groups['.'.join(parts)] += 1
for pattern, count in sorted(groups.items()):
  shape = tuple(sd[next(k for k in unconsumed
                        if [p if not p.isdigit() else 'N'
                            for p in k.split('.')] == pattern.split('.'))].shape)
  print(f'  x{count:<4d} {pattern}   {shape}')

# ── 2. compare against real AF3 native params ────────────────────────────────
print(f'\n=== 2. Structural diff vs AF3 native params ===')
try:
  from alphafold3.model import params as af3_params

  native = af3_params.get_model_haiku_params(AF3_NATIVE)
except Exception as exc:  # noqa: BLE001
  print(f'  could not load AF3 native params: {exc!r}')
  sys.exit(0)

native_flat = {
    f'{scope}/{name}': np.asarray(arr).shape
    for scope, entries in native.items()
    for name, arr in entries.items()
}
conv_flat = {
    f'{scope}/{name}': np.asarray(arr).shape
    for scope, entries in params.items()
    for name, arr in entries.items()
}
print(f'  native: {len(native_flat)} arrays, converted: {len(conv_flat)} arrays')

missing = sorted(set(native_flat) - set(conv_flat))
extra = sorted(set(conv_flat) - set(native_flat))
mismatched = sorted(
    k for k in set(native_flat) & set(conv_flat)
    if native_flat[k] != conv_flat[k]
)

print(f'\n  -- in AF3 but NOT produced by converter: {len(missing)}')
for k in missing:
  print(f'     {k}  {native_flat[k]}')
print(f'\n  -- produced by converter but NOT in AF3: {len(extra)}')
for k in extra:
  print(f'     {k}  {conv_flat[k]}')
print(f'\n  -- SHAPE MISMATCH: {len(mismatched)}')
for k in mismatched:
  print(f'     {k}  af3={native_flat[k]}  converted={conv_flat[k]}')
