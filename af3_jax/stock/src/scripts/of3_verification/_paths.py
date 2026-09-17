"""Path resolution shared by the OF3 verification scripts.

Everything is overridable by environment variable so the scripts are not tied to
one machine:

  OF3_CHECKPOINT   path to the OpenFold3 .pt checkpoint
  AF3_NATIVE_DIR   directory holding native AF3 params (af3.bin.zst), optional
  OPENFOLD3_PATH   checkout of aqlaboratory/openfold3, for the module comparisons

Each falls back to a sibling-of-the-repo layout:

  <parent>/alphafold3          <- this repository
  <parent>/openfold3           <- OF3 source
  <parent>/af3_of3/weights/    <- of3-p2-155k.pt, af3_native/
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
_PARENT = os.path.dirname(REPO_ROOT)

_CHECKPOINT_FALLBACKS = (
    os.path.join(_PARENT, 'af3_of3', 'weights', 'of3-p2-155k.pt'),
)
_NATIVE_FALLBACKS = (os.path.join(_PARENT, 'af3_of3', 'weights', 'af3_native'),)
_OPENFOLD3_FALLBACKS = (os.path.join(_PARENT, 'openfold3'),)


def _first_existing(*candidates: str | None) -> str | None:
  return next((c for c in candidates if c and os.path.exists(c)), None)


def add_import_paths(*, require_openfold3: bool = False) -> None:
  """Put this repo's src/ (and optionally openfold3) on sys.path."""
  sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))
  of3_src = _first_existing(
      os.environ.get('OPENFOLD3_PATH'), *_OPENFOLD3_FALLBACKS
  )
  if of3_src:
    sys.path.insert(0, of3_src)
  elif require_openfold3:
    raise SystemExit(
        'This script needs the openfold3 source. Set $OPENFOLD3_PATH to a '
        'checkout of aqlaboratory/openfold3.'
    )


def checkpoint() -> str:
  """Path to the OF3 checkpoint, or exit with an explanatory message."""
  path = _first_existing(
      os.environ.get('OF3_CHECKPOINT'), *_CHECKPOINT_FALLBACKS
  )
  if not path:
    raise SystemExit(
        'No OF3 checkpoint found. Set $OF3_CHECKPOINT to an OpenFold3 .pt file '
        '(e.g. s3://openfold/staging/of3-p2-155k.pt).'
    )
  return path


def af3_native_dir() -> str | None:
  """Directory of native AF3 params, or None if unavailable."""
  return _first_existing(
      os.environ.get('AF3_NATIVE_DIR'), *_NATIVE_FALLBACKS
  )


def no_cuda_autocast() -> None:
  """Neutralise OF3's `torch.amp.autocast(device_type='cuda')` blocks on CPU."""
  import contextlib

  import torch

  torch.amp.autocast = lambda *a, **k: contextlib.nullcontext()
