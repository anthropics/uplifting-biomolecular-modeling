"""The ONE loader of prediction items (`pred --input <items>`), for the stock caller and the kit driver alike.

Loader rules:
* `<items>` is a DIRECTORY: every `*.npy` file in it, sorted by name; the item id is the file stem.
* every item file is a numpy array of shape (4, 524288): the one-hot of the window with rows A, C, G, T (the upstream one-hot rule; the
  row order is the caller's promise — it cannot be checked from the bytes). Accepted dtypes: uint8 / bool / integer / float with every
  value 0 or 1 and every column sum <= 1 (an N position is an all-zero column); the array is returned as uint8 (4, 524288),
  C-contiguous. `.npz` files are accepted when they carry the array under the key `x` (the kit's canary window form).
* anything else is refused by name (InputError): a wrong shape, a value outside {0, 1}, a column with two ones, a duplicate id.

The driver turns the uint8 array into the documented call's device tensor with `to_device_tensor` (float32 on the device).
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np

SEQ_LEN = 524288
N_ROWS = 4
NPZ_KEY = "x"


class InputError(ValueError):
    """An item the loader refuses, by name."""


def list_items(items_path: str) -> List[Tuple[str, str]]:
    """[(item_id, file_path), ...] per the loader rules; refuses an empty set."""
    if os.path.isdir(items_path):
        names = sorted(f for f in os.listdir(items_path) if f.endswith(".npy"))
        out = [(os.path.splitext(f)[0], os.path.join(items_path, f)) for f in names]
    else:
        raise InputError(f"{items_path}: not a directory of <item>.npy files")
    ids = [i for i, _ in out]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise InputError(f"duplicate item ids {dup}")
    if not out:
        raise InputError(f"{items_path}: no items")
    for i, p in out:
        if not os.path.isfile(p):
            raise InputError(f"item {i}: file not found: {p}")
    return out


def load_onehot(path: str) -> np.ndarray:
    """One item's (4, 524288) uint8 one-hot per the loader rules."""
    if path.endswith(".npz"):
        with np.load(path) as z:
            if NPZ_KEY not in z.files:
                raise InputError(f"{path}: no key {NPZ_KEY!r} in the npz (keys {z.files})")
            x = z[NPZ_KEY]
    else:
        x = np.load(path, allow_pickle=False)
    if not isinstance(x, np.ndarray) or x.shape != (N_ROWS, SEQ_LEN):
        raise InputError(f"{path}: shape {getattr(x, 'shape', None)} != ({N_ROWS}, {SEQ_LEN})")
    if x.dtype == np.bool_:
        u = x.astype(np.uint8)
    elif np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.floating):
        if not np.isin(x, (0, 1)).all():
            raise InputError(f"{path}: values outside {{0, 1}}")
        u = x.astype(np.uint8)
    else:
        raise InputError(f"{path}: dtype {x.dtype} is not a one-hot dtype")
    if (u.sum(axis=0) > 1).any():
        raise InputError(f"{path}: a column with more than one 1 (not a one-hot)")
    return np.ascontiguousarray(u)


def to_device_tensor(x: np.ndarray, device: str = "cuda"):
    """The documented call's input: the (4, 524288) one-hot as a float32 tensor on the device."""
    import torch
    return torch.from_numpy(np.ascontiguousarray(x)).to(device=device).float()
