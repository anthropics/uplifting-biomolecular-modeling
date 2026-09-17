"""Pipeline to prepare reference structures
# TODO: add PDB parsing later
"""

import logging
from typing import Any

import gemmi

from atlasfold.common import ccd, metadata, protein, residue_constants

logger = logging.getLogger(__name__)


# ==================================================
# Helper functions for Metadata parsing
# ==================================================
def read_experiment_record(block: gemmi.cif.Block) -> metadata.ExperimentRecord:
    """Parse RCSB PDB metadata from CIF block."""

    def get_first_value(block: gemmi.cif.Block, tag: str, cast: type = str) -> Any | None:
        values = block.find_values(tag)
        if len(values) > 0:
            try:
                return cast(values[0])
            except Exception:
                return None
        return None

    # PDB ID
    pdb_id = get_first_value(block, "_entry.id")
    assert pdb_id is not None, "PDB ID is missing in metadata."

    # Release Date
    rev_dates = block.find_values("_pdbx_audit_revision_history.revision_date")
    release_date = min(rev_dates) if rev_dates else None
    assert release_date is not None, "Release date is missing in metadata."

    # Method (e.g., X-RAY DIFFRACTION)
    method = get_first_value(block, "_exptl.method")
    assert method is not None, "Experimental method is missing in metadata."
    method = method.replace("'", "").replace('"', "").upper()  # clean quotes
    if method not in metadata.ALL_EXPERIMENT_METHODS:
        method = "OTHER"

    # Resolution (Handle X-ray vs EM vs NMR)
    # NMR: no resolution (None)
    # X-ray standard
    resolution = get_first_value(block, "_refine.ls_d_res_high", float)
    if not resolution:
        # EM
        resolution = get_first_value(block, "_em_3d_reconstruction.resolution", float)
    if not resolution:
        # Fallback
        resolution = get_first_value(block, "_reflns.d_resolution_high", float)

    return metadata.ExperimentRecord(
        pdb_id=pdb_id,
        release_date=release_date,
        method=method,
        resolution=resolution,
    )


# ==================================================
# Helper functions for gemmi structure validation
# ==================================================
def clean_up_gemmi_structure(
    raw_struct: gemmi.Structure,
    map_mse_to_met: bool = True,
    canonicalize_arginines: bool = True,
) -> None:
    """Clean up gemmi Structure object in-place.

    See AlphaFold3 Section 2.1 Parsing.
    """
    raw_struct.merge_chain_parts()
    raw_struct.remove_alternative_conformations()
    raw_struct.remove_hydrogens()
    raw_struct.remove_waters()
    raw_struct.remove_empty_chains()

    protein_entity_ids: set[int] = set()
    protein_chains: set[str] = set()
    for entity in raw_struct.entities:
        # In the case of microheterogeneity, take the first monomer
        entity.full_sequence = [
            gemmi.Entity.first_mon(res) for res in entity.full_sequence
        ]
        if not (
            entity.entity_type == gemmi.EntityType.Polymer
            and entity.polymer_type == gemmi.PolymerType.PeptideL
        ):
            # Skip non-protein entities
            continue
        # Collect protein asym_ids
        protein_entity_ids.add(int(entity.name))
        protein_chains.update(entity.subchains)

        if map_mse_to_met:
            # Map MSE to MET
            if entity.name in protein_entity_ids:
                entity.full_sequence = [
                    "MET" if res == "MSE" else res for res in entity.full_sequence
                ]

    model: gemmi.Model = raw_struct[0]
    for res_span in model.subchains():
        label_id = res_span.subchain_id()
        if label_id in protein_chains:
            for residue in res_span:
                if map_mse_to_met and residue.name == "MSE":
                    # Map MSE to MET
                    residue.name = "MET"
                    for atom in residue:
                        if atom.name == "SE":
                            atom.name = "SD"
                            atom.element = gemmi.Element("S")
                if canonicalize_arginines and residue.name == "ARG":
                    # Ensure arginine NH1/NH2 naming is canonical
                    try:
                        cd: gemmi.Atom = residue["CD"][0]
                        nh1: gemmi.Atom = residue["NH1"][0]
                        nh2: gemmi.Atom = residue["NH2"][0]
                    except Exception:
                        continue
                    # Calculate distances
                    dist_cd_nh1 = cd.pos.dist(nh1.pos)
                    dist_cd_nh2 = cd.pos.dist(nh2.pos)
                    # Swap if NH2 is closer to CD
                    if dist_cd_nh2 < dist_cd_nh1:
                        # Swap names
                        nh1.name, nh2.name = "NH2", "NH1"


# ==================================================
# Main functions for reference structure preparation
# ==================================================
def get_protein_chains(
    raw_struct: gemmi.Structure,
) -> list[tuple[protein.Protein, dict[str, Any]]]:
    """Parse protein chains from gemmi Structure and return a list of Structures."""

    label_id_to_auth_id: dict[str, str] = {}
    for chain in raw_struct[0]:
        auth_id: str = chain.name
        for subchain in chain.subchains():
            label_id: str = subchain.subchain_id()
            label_id_to_auth_id[label_id] = auth_id

    chains: list[tuple[protein.Protein, dict[str, Any]]] = []
    for entity in raw_struct.entities:
        if len(entity.subchains) == 0:
            # Skip entities without subchains
            continue
        if entity.entity_type != gemmi.EntityType.Polymer:
            # Skip non-polymer entities
            continue
        if entity.polymer_type != gemmi.PolymerType.PeptideL:
            # Skip non-protein polymers
            continue

        # Get 1-letter sequence
        ccd_sequences: list[str] = entity.full_sequence
        aas = [ccd.CCD_NAME_TO_ONE_LETTER.get(restype, "X") for restype in ccd_sequences]
        aas = [aa if aa in residue_constants.restype_1to3 else "X" for aa in aas]
        sequence = "".join(aas)

        # Mark all label_ids as valid initially
        for label_id in entity.subchains:
            if label_id not in label_id_to_auth_id:
                logger.warning(
                    f"Label ID {label_id} not found in gemmi structure; skipping."
                )
                continue
            auth_id = label_id_to_auth_id[label_id]
            c = protein.Protein.get_empty(auth_id, sequence)
            m = {
                "label_id": label_id,
                "auth_id": auth_id,
                "entity_id": int(entity.name),
            }
            chains.append((c, m))

    label_id_to_chain: dict[str, protein.Protein] = {m["label_id"]: c for c, m in chains}
    for raw_chain in raw_struct[0].subchains():
        raw_chain: gemmi.ResidueSpan
        label_id = raw_chain.subchain_id()
        if label_id not in label_id_to_chain:
            continue
        # Insert coordinates
        insert_chain_coordinates(label_id_to_chain[label_id], raw_chain)

    return chains


def insert_chain_coordinates(
    chain: protein.Protein,
    raw_chain: gemmi.ResidueSpan,
) -> None:
    """Insert Coordinates from raw gemmi ResidueSpan into reference chain."""
    sequence: str = chain.sequence
    for res in raw_chain:
        res: gemmi.Residue

        # Get residue index
        res_idx = res.label_seq
        if res_idx is None:
            logger.warning(
                f"Residue {res.name} in chain {raw_chain.subchain_id()} "
                f"missing label_seq; skipping."
            )
            continue

        if res_idx < 1 or res_idx > len(sequence):
            # Skip invalid residue indices
            logger.warning(
                f"Residue index {res_idx} out of bounds for chain with length "
                f"{len(sequence)}."
            )
            continue

        aa = sequence[res_idx - 1]
        restype = residue_constants.restype_1to3.get(aa, "UNK")
        res_atom14_order = residue_constants.restype_atom14_order[restype]

        # Get atoms.
        for a in res:
            a: gemmi.Atom
            name: str = a.name.upper().strip()
            if name not in res_atom14_order:
                # Skip non-standard atoms
                if name in ("OXT", "H", "D"):
                    # Ignore loging for missing OXT and hydrogens
                    continue
                logger.debug(
                    f"Atom {name} in residue {res.name} {res_idx} not in "
                    f"standard atom list for residue type {restype}; skipping."
                )
                continue
            atom_i: int = res_atom14_order[name]
            coords: gemmi.Position = a.pos
            chain.coordinates[res_idx - 1, atom_i, :] = (coords.x, coords.y, coords.z)
            chain.b_factors[res_idx - 1, atom_i] = a.b_iso
