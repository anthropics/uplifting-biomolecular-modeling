"""Atom37 layout and scatter helpers for `StructureModelOutput`.

Atom37 is the standard heavy-atom layout used by AlphaFold, OpenFold, ESMFold,
proteina, etc. — 37 named heavy-atom slots in a fixed order. Each
`StructureModelOutput` carries a per-residue atom37 view of the predicted
all-atom coordinates, populated by the model wrapper that produced it. This
module exposes the canonical name list plus a scatter helper that backbones
the protenix / boltz / boltz2 implementations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

# Canonical atom37 ordering (matches AlphaFold residue_constants.atom_types).
ATOM37_NAMES: tuple[str, ...] = (
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1",
    "SG", "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD",
    "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2",
    "CH2", "NH1", "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
)
assert len(ATOM37_NAMES) == 37

ATOM37_INDEX: dict[str, int] = {name: i for i, name in enumerate(ATOM37_NAMES)}


def scatter_atom37(
    atom_coords: Float[Array, "N_atom 3"],
    atom_to_token: Int[Array, " N_atom"],
    atom37_idx: Int[Array, " N_atom"],
    n_token: int,
) -> tuple[Float[Array, "N_token 37 3"], Float[Array, "N_token 37"]]:
    """Scatter all-atom coords into a `[n_token, 37, 3]` grid + mask.

    `atom37_idx` is `-1` for atoms that have no atom37 slot (e.g. padding atoms,
    or real atoms whose name isn't in the 37-list). Such atoms must be dropped:
    `mode='drop'` only discards indices `>= 37`, NOT negative ones — a raw `-1`
    wraps to slot 36 (OXT) and pollutes token 0 — so remap `< 0` to an explicitly
    out-of-range slot (37) first.
    """
    slot = jnp.where(atom37_idx < 0, 37, atom37_idx)
    coords = (
        jnp.zeros((n_token, 37, 3), dtype=jnp.float32)
        .at[atom_to_token, slot]
        .set(atom_coords, mode="drop")
    )
    mask = (
        jnp.zeros((n_token, 37), dtype=jnp.float32)
        .at[atom_to_token, slot]
        .set(jnp.ones(atom_coords.shape[:-1], dtype=jnp.float32), mode="drop")
    )
    return coords, mask
