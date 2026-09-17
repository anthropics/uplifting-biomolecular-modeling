"""Dump {scope/name: [shape, md5]} for every converted AF3 param.

Used to diff the converted parameter tree before vs. after the fixes: shapes
must be identical (so nothing downstream can fail to load) and only the
deliberately-changed arrays may differ in content.
"""

import hashlib
import json
import sys

import _paths

_paths.add_import_paths()

import numpy as np
from alphafold3.model import of3_weight_converter as conv

CKPT = _paths.checkpoint()
out_path = sys.argv[1]

sd = conv.load_of3_checkpoint(CKPT)
params = conv.map_of3_to_af3(sd)

digest = {}
for scope, entries in params.items():
  for name, arr in entries.items():
    arr = np.ascontiguousarray(np.asarray(arr))
    digest[f'{scope}/{name}'] = [
        list(arr.shape),
        hashlib.md5(arr.tobytes()).hexdigest(),
    ]

with open(out_path, 'w') as f:
  json.dump(digest, f, indent=0, sort_keys=True)
print(f'{len(digest)} arrays -> {out_path}')
