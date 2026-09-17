"""ptx_drop_bond_mask -- never build Protenix's [N_atom, N_atom] int64 'bond_mask' feature at inference.

WHAT IT PATCHES : protenix.data.core.featurizer.Featurizer.get_mask_features (protenix 2.0.0, featurizer.py 645-679). The stock method ends with
                  bond_mask_mat = np.zeros((num_atoms, num_atoms)) [float64] -> torch.Tensor(...) [fp32 copy] -> .long() [int64]  = 20 B per ATOM PAIR
                  transiently, 8 B resident, and the runner then ships it to the GPU with every other feature. The patch runs the stock statements
                  with the two N_atom^2 allocations neutralised and removes the key; every other mask feature is produced by the untouched stock code.
WHY IT IS SAFE  : the ONLY consumer of input_feature_dict["bond_mask"] in protenix 2.0.0 is the TRAINING bond loss (protenix/model/loss.py 317-410);
                  the inference forward, the dumper and the confidence code never read it => outputs cannot change. No python `random` is consumed here => the MC-dropout draw is unchanged.
SIZES           : 8 B per atom pair resident (20 B transiently while building): ~10 GB at 37k atoms, ~120 GB at 125k, ~470 GB at 250k.
                  On 1 card it sits on the GPU too.
USAGE           : import ptx_drop_bond_mask; ptx_drop_bond_mask.apply()          # before featurization (i.e. before the DataLoader iterates)
                  or env-gated: PTX_DROP_BOND_MASK=1 (alias PTX_TP_DROP_BOND_MASK=1) + `import ptx_drop_bond_mask; ptx_drop_bond_mask.apply_from_env()`
                  e.g. from a usercustomize.py / sitecustomize.py on PYTHONPATH. Default OFF outside the PTX-TP launcher (which turns it on for TP runs).
                  remove() restores the stock method; report() -> {"applied": bool, "calls": n, "atoms_last": N_atom}."""
from __future__ import annotations

import os
import sys

__version__ = "1.0"
_STATE = {"applied": False, "calls": 0, "atoms_last": None, "orig": None}


def _log(msg: str) -> None:
    print(f"[ptx_drop_bond_mask] {msg}", file=sys.stderr, flush=True)


def apply(verbose: bool = True) -> bool:
    """Install the patch (idempotent). Returns True if installed now or earlier."""
    import numpy as np
    import protenix.data.core.featurizer as FZ
    cls = FZ.Featurizer
    if _STATE["applied"]:
        return True
    orig = cls.get_mask_features
    glpbm = FZ.get_ligand_polymer_bond_mask

    def get_mask_features(self):
        n = len(self.cropped_atom_array)
        np_zeros = np.zeros

        def zeros(shape, *a, **k):                      # only the exact (N_atom, N_atom) request is redirected to a 1x1 dummy
            if isinstance(shape, tuple) and shape == (n, n) and not a and not k:
                return np_zeros((1, 1))
            return np_zeros(shape, *a, **k)

        FZ.get_ligand_polymer_bond_mask = lambda atom_array: np_zeros((0, 3), dtype=np.int64)   # no bonds to scatter into the dummy
        FZ.np.zeros = zeros
        try:
            feats = orig(self)
        finally:
            FZ.np.zeros = np_zeros
            FZ.get_ligand_polymer_bond_mask = glpbm
        feats.pop("bond_mask", None)
        _STATE["calls"] += 1
        _STATE["atoms_last"] = n
        return feats

    get_mask_features._ptx_drop_bond_mask = True
    cls.get_mask_features = get_mask_features
    _STATE.update(applied=True, orig=orig)
    if verbose:
        _log("APPLIED: Featurizer.get_mask_features no longer builds the [N_atom,N_atom] int64 'bond_mask' (training-loss-only feature)")
    return True


def remove() -> bool:
    import protenix.data.core.featurizer as FZ
    if not _STATE["applied"]:
        return False
    FZ.Featurizer.get_mask_features = _STATE["orig"]
    _STATE.update(applied=False, orig=None)
    _log("removed (stock get_mask_features restored)")
    return True


def enabled_from_env() -> bool:
    return os.environ.get("PTX_DROP_BOND_MASK", os.environ.get("PTX_TP_DROP_BOND_MASK", "0")) == "1"


def apply_from_env(verbose: bool = True) -> bool:
    """Apply iff PTX_DROP_BOND_MASK=1 or PTX_TP_DROP_BOND_MASK=1 (default OFF)."""
    if enabled_from_env():
        return apply(verbose=verbose)
    if verbose:
        _log("not applied (set PTX_DROP_BOND_MASK=1)")
    return False


def report() -> dict:
    return {"applied": _STATE["applied"], "calls": _STATE["calls"], "atoms_last": _STATE["atoms_last"], "version": __version__}


if __name__ == "__main__":      # python ptx_drop_bond_mask.py -> prints report() (nothing is applied)
    print(report())
