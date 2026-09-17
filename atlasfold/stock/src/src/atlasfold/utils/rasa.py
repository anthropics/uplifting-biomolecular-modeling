# Copyright 2026 Seonghwan Seo
# SPDX-License-Identifier: MIT
#
# This is an original AtlasFold implementation of RASA-based disorder scoring.
# It uses a Numba-parallel Shrake-Rupley kernel and SciPy cKDTree neighbor lists;
# no Biotite or AlphaFold 3 source code is included. Biotite's SASA result was
# used as the numerical reference when validating this implementation.
#
# Method and parameter references:
# - Shrake & Rupley, 1973: https://doi.org/10.1016/0022-2836(73)90011-9
# - Sander & Rost, 1994: https://doi.org/10.1002/prot.340200303
# - Tsai et al., 1999: https://doi.org/10.1006/jmbi.1999.2829
# - AlphaFold 3 fraction-disordered metric definition:
#   https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/model/confidences.py
# - Biotite SASA numerical reference (BSD-3-Clause):
#   https://github.com/biotite-dev/biotite

"""Relative solvent accessibility metrics for protein structures."""

import os
from collections.abc import Sequence

import numpy as np
from numba import njit, prange, set_num_threads
from scipy.spatial import cKDTree

from atlasfold.common import residue_constants

# Sander & Rost, 1994: https://doi.org/10.1002/prot.340200303
_MAX_ACCESSIBLE_SURFACE_AREA = np.array(
    [
        106.0,  # ALA
        248.0,  # ARG
        157.0,  # ASN
        163.0,  # ASP
        135.0,  # CYS
        198.0,  # GLN
        194.0,  # GLU
        84.0,  # GLY
        184.0,  # HIS
        169.0,  # ILE
        164.0,  # LEU
        205.0,  # LYS
        188.0,  # MET
        197.0,  # PHE
        136.0,  # PRO
        130.0,  # SER
        142.0,  # THR
        227.0,  # TRP
        222.0,  # TYR
        142.0,  # VAL
        106.0,  # UNK, treated as ALA
    ],
    dtype=np.float32,
)

# ProtOr radii for AtlasFold's atom14 ordering. Values follow Tsai et al.,
# 1999, as used by Biotite/OpenFold3. Padded atom slots have radius zero.
# fmt: off
_RESTYPE_ATOM14_PROTOR_RADII = np.array(
    [
        [1.64, 1.88, 1.61, 1.42, 1.88, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1.64, 1.88, 1.61, 1.42, 1.88, 1.88, 1.88, 1.64, 1.61, 1.64, 1.64, 0, 0, 0],
        [1.64, 1.88, 1.61, 1.42, 1.88, 1.61, 1.42, 1.64, 0, 0, 0, 0, 0, 0],
        [1.64, 1.88, 1.61, 1.42, 1.88, 1.61, 1.42, 1.46, 0, 0, 0, 0, 0, 0],
        [1.64, 1.88, 1.61, 1.42, 1.88, 1.77, 0, 0, 0, 0, 0, 0, 0, 0],
        [1.64, 1.88, 1.61, 1.42, 1.88, 1.88, 1.61, 1.42, 1.64, 0, 0, 0, 0, 0],
        [1.64, 1.88, 1.61, 1.42, 1.88, 1.88, 1.61, 1.42, 1.46, 0, 0, 0, 0, 0],
        [1.64, 1.88, 1.61, 1.42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1.64, 1.88, 1.61, 1.42, 1.88, 1.61, 1.64, 1.76, 1.76, 1.64, 0, 0, 0, 0],
        [1.64, 1.88, 1.61, 1.42, 1.88, 1.88, 1.88, 1.88, 0, 0, 0, 0, 0, 0],
        [1.64, 1.88, 1.61, 1.42, 1.88, 1.88, 1.88, 1.88, 0, 0, 0, 0, 0, 0],
        [1.64, 1.88, 1.61, 1.42, 1.88, 1.88, 1.88, 1.88, 1.64, 0, 0, 0, 0, 0],
        [1.64, 1.88, 1.61, 1.42, 1.88, 1.88, 1.77, 1.88, 0, 0, 0, 0, 0, 0],
        [1.64, 1.88, 1.61, 1.42, 1.88, 1.61, 1.76, 1.76, 1.76, 1.76, 1.76, 0, 0, 0],
        [1.64, 1.88, 1.61, 1.42, 1.88, 1.88, 1.88, 0, 0, 0, 0, 0, 0, 0],
        [1.64, 1.88, 1.61, 1.42, 1.88, 1.46, 0, 0, 0, 0, 0, 0, 0, 0],
        [1.64, 1.88, 1.61, 1.42, 1.88, 1.46, 1.88, 0, 0, 0, 0, 0, 0, 0],
        [1.64, 1.88, 1.61, 1.42, 1.88, 1.61, 1.76, 1.61, 1.64, 1.61, 1.76, 1.76, 1.76, 1.76], # noqa
        [1.64, 1.88, 1.61, 1.42, 1.88, 1.61, 1.76, 1.76, 1.76, 1.76, 1.61, 1.46, 0, 0],
        [1.64, 1.88, 1.61, 1.42, 1.88, 1.88, 1.88, 0, 0, 0, 0, 0, 0, 0],
        [1.64, 1.88, 1.61, 1.42, 1.88, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.float32,
)
# fmt: on


def _fibonacci_sphere(num_points: int) -> np.ndarray:
    phi = (3.0 - np.sqrt(5.0)) * np.pi * np.arange(num_points)
    z = np.linspace(1.0 - 1.0 / num_points, 1.0 / num_points - 1.0, num_points)
    radius = np.sqrt(1.0 - z * z)
    return np.stack((radius * np.cos(phi), radius * np.sin(phi), z), axis=-1).astype(
        np.float32
    )


@njit(parallel=True, fastmath=True, cache=True)
def _compute_atom_sasa(
    coordinates: np.ndarray,
    radii: np.ndarray,
    sphere_points: np.ndarray,
    neighbor_offsets: np.ndarray,
    neighbor_indices: np.ndarray,
) -> np.ndarray:
    """Compute atom SASA from a CSR neighbor list using Shrake-Rupley."""
    num_points = sphere_points.shape[0]
    sasa = np.empty(coordinates.shape[0], dtype=np.float32)
    area_scale = np.float32(4.0 * np.pi / num_points)

    for atom_idx in prange(coordinates.shape[0]):
        radius = radii[atom_idx]
        accessible = 0
        for point_idx in range(num_points):
            point_x = coordinates[atom_idx, 0] + radius * sphere_points[point_idx, 0]
            point_y = coordinates[atom_idx, 1] + radius * sphere_points[point_idx, 1]
            point_z = coordinates[atom_idx, 2] + radius * sphere_points[point_idx, 2]
            is_accessible = True
            for neighbor_offset in range(
                neighbor_offsets[atom_idx], neighbor_offsets[atom_idx + 1]
            ):
                neighbor_idx = neighbor_indices[neighbor_offset]
                dx = point_x - coordinates[neighbor_idx, 0]
                dy = point_y - coordinates[neighbor_idx, 1]
                dz = point_z - coordinates[neighbor_idx, 2]
                if dx * dx + dy * dy + dz * dz < radii[neighbor_idx] ** 2:
                    is_accessible = False
                    break
            if is_accessible:
                accessible += 1
        sasa[atom_idx] = accessible * area_scale * radius * radius
    return sasa


def compute_rasa(
    coordinates: np.ndarray,
    chain_sequences: Sequence[str],
    *,
    probe_radius: float = 1.4,
    num_sphere_points: int = 1000,
) -> np.ndarray:
    """Compute per-residue relative solvent-accessible surface area (RASA).

    Parameters
    ----------
    coordinates
        Atom14 coordinates with shape ``[num_samples, num_residues, 14, 3]``.
    chain_sequences
        Protein sequences in coordinate order. Solvent accessibility is computed
        independently for each chain.

    Returns
    -------
    np.ndarray
        Per-residue RASA for each sample, with shape
        ``[num_samples, num_residues]``.
    """
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.ndim != 4 or coordinates.shape[-2:] != (14, 3):
        raise ValueError(
            "coordinates must have shape [num_samples, num_residues, 14, 3], "
            f"got {coordinates.shape}."
        )
    chain_sequences = list(chain_sequences)
    assert chain_sequences and all(chain_sequences)
    num_samples, num_residues = coordinates.shape[:2]
    assert sum(map(len, chain_sequences)) == num_residues

    sequences = "".join(chain_sequences)
    aatype = np.array(
        [residue_constants.restype_orders[residue] for residue in sequences],
        dtype=np.int64,
    )
    atom_mask = residue_constants.restype_atom14_mask[aatype]
    atom_radii = _RESTYPE_ATOM14_PROTOR_RADII[aatype] + np.float32(probe_radius)
    max_accessible_area = _MAX_ACCESSIBLE_SURFACE_AREA[aatype]
    sphere_points = _fibonacci_sphere(num_sphere_points)

    all_coordinates: list[np.ndarray] = []
    all_radii: list[np.ndarray] = []
    all_residue_indices: list[np.ndarray] = []
    neighbor_offsets = [0]
    neighbor_indices: list[int] = []
    atom_offset = 0

    chain_ranges = []
    chain_start = 0
    for sequence in chain_sequences:
        chain_end = chain_start + len(sequence)
        chain_ranges.append((chain_start, chain_end))
        chain_start = chain_end

    for sample_idx in range(num_samples):
        for chain_start, chain_end in chain_ranges:
            chain_mask = atom_mask[chain_start:chain_end]
            chain_coordinates = coordinates[sample_idx, chain_start:chain_end][chain_mask]
            chain_radii = atom_radii[chain_start:chain_end][chain_mask]
            residue_indices = np.broadcast_to(
                np.arange(chain_start, chain_end, dtype=np.int64)[:, None],
                chain_mask.shape,
            )[chain_mask]
            residue_indices = residue_indices + sample_idx * num_residues

            tree = cKDTree(chain_coordinates)
            candidate_neighbors = tree.query_ball_point(
                chain_coordinates,
                chain_radii + chain_radii.max(),
            )
            for local_atom_idx, candidates in enumerate(candidate_neighbors):
                for candidate_idx in candidates:
                    if candidate_idx != local_atom_idx:
                        neighbor_indices.append(atom_offset + candidate_idx)
                neighbor_offsets.append(len(neighbor_indices))

            all_coordinates.append(chain_coordinates)
            all_radii.append(chain_radii)
            all_residue_indices.append(residue_indices)
            atom_offset += len(chain_coordinates)

    flat_coordinates = np.concatenate(all_coordinates).astype(np.float32, copy=False)
    flat_radii = np.concatenate(all_radii).astype(np.float32, copy=False)
    flat_residue_indices = np.concatenate(all_residue_indices)
    num_threads = int(os.environ.get("NUMBA_NUM_THREADS", 4))
    set_num_threads(num_threads)
    atom_sasa = _compute_atom_sasa(
        flat_coordinates,
        flat_radii,
        sphere_points,
        np.asarray(neighbor_offsets, dtype=np.int64),
        np.asarray(neighbor_indices, dtype=np.int64),
    )

    residue_sasa = np.zeros(num_samples * num_residues, dtype=np.float32)
    np.add.at(residue_sasa, flat_residue_indices, atom_sasa)
    residue_rasa = residue_sasa.reshape(num_samples, num_residues)
    return np.clip(residue_rasa / max_accessible_area[None], 0.0, 1.0)
