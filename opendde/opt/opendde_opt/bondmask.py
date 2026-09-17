"""drop_bond_mask — the tree's own lever on the stock featurizer (exact class, bitwise by construction).

The stock `Featurizer.get_mask_features` (`opendde/data/core/featurizer.py:857-865`) builds `mask_features["bond_mask"]`, an
[N_atom, N_atom] int64 tensor that no consumer reads at inference: the model reads `token_bonds`; the key's only other mentions are the
training data pipeline's shape table (`opendde/data/utils.py:970`) and test fixtures. The feature dict moves to the GPU whole, so the
tensor costs 8 B × N_atom² of device memory for the whole prediction (0.5 GB at 8,000 atoms, 3.3 GB at 20,000). This lever drops the
key from the featurizer's return before the dict leaves the host; the stock's host-side transient inside the call (the fp64, fp32 and
int64 copies) remains. STATS counts every drop and its bytes; a call whose return has no such key is counted (`missing`) and named by
the kit's gate as a fallback (the stock changed under the lever).

Installed from `stack._apply` for the lines that carry the lever through the core's per-site patch (`opt_core.autoload.patch_attr_at_import`):
the class is patched at once when the featurizer module is already imported, else right after the module's own body executes. Idempotent.
"""
import sys

from opt_core import autoload

TARGET = "opendde.data.core.featurizer"
CLASS, METHOD, KEY = "Featurizer", "get_mask_features", "bond_mask"
TAG = "opendde-opt"
STATS = {"installed": False, "armed": False, "dropped": 0, "bytes_dropped": 0, "missing": 0}


class ActivationError(RuntimeError):
    """The featurizer has no `Featurizer.get_mask_features` to wrap: the kit's activation fails by name."""


def make_wrapper(orig):
    """The method installed on the class: the stock's mask features without the unread ``bond_mask`` key (counted)."""
    def get_mask_features(self):
        out = orig(self)
        t = out.pop(KEY, None)
        if t is None:
            STATS["missing"] += 1
        else:
            STATS["dropped"] += 1
            STATS["bytes_dropped"] += int(t.numel()) * int(t.element_size())
        return out
    get_mask_features._drop_bond_mask = True
    get_mask_features._orig = orig
    return get_mask_features


def _sync() -> None:
    p = STATS.get("patch")
    if p is not None:
        STATS["installed"], STATS["armed"] = p.state == "installed", p.state == "armed"


def install() -> None:
    """Patch now when the featurizer is imported, else at its import (the core's per-site patch). Idempotent."""
    _sync()
    if STATS["installed"] or STATS["armed"]:
        return
    try:
        STATS["patch"] = autoload.patch_attr_at_import(TARGET, f"{CLASS}.{METHOD}", make_wrapper, tag=TAG, name="drop_bond_mask")
    except autoload.PatchError as e:
        raise ActivationError(str(e)) from None
    _sync()


def fallbacks(planned) -> list:
    """The lever's named events at exit: the featurizer imported but not wrapped, or the key absent from the stock's return. A process
    that never imported the featurizer has no event."""
    if "drop_bond_mask" not in planned:
        return []
    out = []
    _sync()
    if TARGET in sys.modules and not STATS["installed"]:
        out.append("drop_bond_mask: the featurizer is imported but not wrapped (installed after its import without the patch)")
    if STATS["missing"]:
        out.append(f"drop_bond_mask: {STATS['missing']} featurizer calls returned no bond_mask (the stock changed under the lever)")
    return out
