import copy
import dataclasses
import functools

import gemmi
import numpy as np

from atlasfold.common import residue_constants


@functools.lru_cache(21)
def get_residue_template(restype: str) -> gemmi.Residue:
    residue = gemmi.Residue()
    residue.name = restype
    residue.entity_type = gemmi.EntityType.Polymer
    atom_names: tuple[str, ...] = residue_constants.residue_atoms[restype]
    for an in atom_names:
        atom = gemmi.Atom()
        atom.name = an
        atom.element = gemmi.Element(an[0])
        residue.add_atom(atom)
    return residue


@dataclasses.dataclass
class ChainInfo:
    label_asym_id: str  # stable mmCIF label_asym_id, e.g. A B C
    auth_asym_id: str  # author/PDB chain ID
    entity_id: int  # 1 2 3
    sequence: str
    coordinates: np.ndarray  # [L, 14, 3]
    b_factors: np.ndarray | None  # [L, 14] or [L,] or None


def to_gemmi_chain(cinfo: ChainInfo) -> gemmi.Chain:
    label_asym_id = cinfo.label_asym_id
    auth_asym_id = cinfo.auth_asym_id
    entity_id = cinfo.entity_id
    sequence = cinfo.sequence
    coordinates = cinfo.coordinates
    b_factors = cinfo.b_factors
    length = len(cinfo.sequence)
    if coordinates.shape != (length, 14, 3):
        raise ValueError(
            f"Invalid coordinates shape: {coordinates.shape}. "
            f"Expected (L, 14, 3) where L is the sequence length."
        )
    if b_factors is not None and b_factors.shape not in [(length, 14), (length,)]:
        raise ValueError(
            f"Invalid b_factors shape: {b_factors.shape}. "
            f"Expected (L, 14) or (L,) where L is the sequence length."
        )

    if b_factors is not None and b_factors.shape == (length,):
        # Broadcast the per-residue B-factors to all atoms in the residue
        b_factors = np.broadcast_to(b_factors[:, np.newaxis], (length, 14))

    chain = gemmi.Chain(auth_asym_id)

    full_sequence = [residue_constants.restype_1to3[aa] for aa in sequence]
    residues: list[gemmi.Residue] = [
        copy.deepcopy(get_residue_template(restype)) for restype in full_sequence
    ]
    for res_i, residue in enumerate(residues):
        # Set residue index
        res_idx = res_i + 1
        residue.entity_id = str(entity_id)
        residue.subchain = label_asym_id
        residue.seqid.num = res_idx
        residue.label_seq = res_idx

        # Set atom coordinates and B-factors
        if b_factors is not None:
            biso = b_factors[res_i]  # Shape: (14,)
        for atom_i, atom in enumerate(residue):
            x, y, z = coordinates[res_i, atom_i]
            atom.pos = gemmi.Position(x, y, z)
            atom.b_iso = biso[atom_i] if b_factors is not None else 100.0
    chain.append_residues(residues)
    return chain


def to_gemmi_structure(name: str, chains: list[ChainInfo]) -> gemmi.Structure:
    """Convert a list of Chains into a gemmi.Structure."""
    struct = gemmi.Structure()
    struct.name = name

    # Set up entities
    entities: dict[int, gemmi.Entity] = {}
    for cinfo in chains:
        if cinfo.entity_id in entities:
            continue
        full_sequence = [residue_constants.restype_1to3[aa] for aa in cinfo.sequence]
        entity = gemmi.Entity(str(cinfo.entity_id))
        entity.entity_type = gemmi.EntityType.Polymer
        entity.polymer_type = gemmi.PolymerType.PeptideL
        entity.full_sequence = full_sequence
        entities[cinfo.entity_id] = entity

    for cinfo in chains:
        entity = entities[cinfo.entity_id]
        entity.subchains.append(cinfo.label_asym_id)

    struct.entities = gemmi.EntityList([entities[eid] for eid in sorted(entities.keys())])

    # Create a model
    try:
        model = gemmi.Model(1)
    except Exception:
        model = gemmi.Model("1")  # Fallback for older Gemmi versions

    for cinfo in chains:
        chain = to_gemmi_chain(cinfo)
        model.add_chain(chain)

    # Add model to structure
    struct.add_model(model)
    struct.setup_entities()
    return struct


# ============================================================
# PDB/mmCIF output functions
# ============================================================


RAW_PDB_HEADER = """
HEADER
TITLE     %s
REMARK   1 REFERENCE 1
REMARK   1  AUTH   SEONGHWAN SEO, HYEONGWOO KIM, SEOKHYUN MOON, WOO YOUN KIM
REMARK   1  TITL   ATLASFOLD: PROTEIN STRUCTURE PREDICTION WITH METAGENOMIC-
REMARK   1  TITL 2 SCALE LANGUAGE MODELS
REMARK   1  REF    TO BE PUBLISHED
"""


def to_pdb(struct: gemmi.Structure, model: str | None = None) -> str:
    if model == "monomer":
        title = "ATLASFOLD MONOMER PREDICTION of " + struct.name
    elif model == "multimer":
        title = "ATLASFOLD MULTIMER PREDICTION of " + struct.name
    elif model is None:
        title = "ATLASFOLD PREDICTION of " + struct.name
    else:
        raise ValueError(f"Unknown model type: {model}")
    header = RAW_PDB_HEADER % title
    struct.raw_remarks = header.strip().splitlines()
    return struct.make_pdb_string()


def to_mmcif(struct: gemmi.Structure, model: str | None = None) -> str:
    cif_block = gemmi.cif.Block(struct.name)

    # Add metadata
    cif_block.set_pair("_entry.id", struct.name)

    author_loop: gemmi.cif.Loop = cif_block.init_loop(
        "_citation_author.", ["citation_id", "ordinal", "name"]
    )
    author_loop.add_row(["primary", "1", '"Seo, Seonghwan"'])
    author_loop.add_row(["primary", "2", '"Kim, Hyeongwoo"'])
    author_loop.add_row(["primary", "3", '"Moon, Seokhyun"'])
    author_loop.add_row(["primary", "4", '"Kim, Woo Youn"'])

    software_loop = cif_block.init_loop(
        "_software.",
        ["pdbx_ordinal", "name", "type", "description", "version"],
    )

    if model == "monomer":
        description = "AtlasFold Monomer Prediction Pipeline"
    elif model == "multimer":
        description = "AtlasFold Multimer Prediction Pipeline"
    elif model is None:
        description = "AtlasFold Prediction Pipeline"
    else:
        raise ValueError(f"Unknown model type: {model}")
    software_loop.add_row(
        ["1", "AtlasFold", "model", gemmi.cif.quote(description), "1.0.0"]
    )

    # Add structure data
    struct.update_mmcif_block(cif_block)

    # Round coordinates and B-factors
    atom_site_table = cif_block.find("_atom_site.", ["Cartn_x", "Cartn_y", "Cartn_z"])
    for row in atom_site_table:
        row[0] = f"{float(row[0]):.3f}"
        row[1] = f"{float(row[1]):.3f}"
        row[2] = f"{float(row[2]):.3f}"
    b_factor_table = cif_block.find("_atom_site.", ["B_iso_or_equiv"])
    if b_factor_table:
        for row in b_factor_table:
            row[0] = f"{float(row[0]):.2f}"

    # Add custom categories for OST compatibility
    _add_pdbx_poly_seq_scheme(cif_block, struct)
    _update_entity_poly(cif_block, struct)
    _update_entity_poly_seq(cif_block, struct)
    _update_chem_comp(cif_block)

    return cif_block.as_string()


def _add_pdbx_poly_seq_scheme(block: gemmi.cif.Block, structure: gemmi.Structure):
    """
    Manually add the _pdbx_poly_seq_scheme category to the CIF block.
    This is required for OST compatibility and proper polymer parsing.
    """
    # Columns required for _pdbx_poly_seq_scheme
    columns = [
        "asym_id",  # label_asym_id (residue.subchain)
        "entity_id",  # entity_id
        "mon_id",  # residue name
        "seq_id",  # residue sequence number
        "pdb_strand_id",  # auth_asym_id (chain.name)
        "pdb_seq_num",  # auth_seq_id
        "pdb_ins_code",  # PDB insertion code
    ]
    loop = block.init_loop("_pdbx_poly_seq_scheme.", columns)
    # Iterate strictly over the first model (assuming single model structure for AF3)
    model = structure[0]
    for chain in model:
        for res in chain:
            # Check if residue is part of a polymer ('A' het_flag)
            if res.het_flag == "A":
                # Map values
                asym_id = res.subchain if res.subchain else chain.name
                entity_id = res.entity_id
                mon_id = res.name
                seq_num = str(res.seqid.num)
                strand_id = chain.name  # auth_asym_id
                ins_code = "." if res.seqid.icode == " " else res.seqid.icode
                loop.add_row(
                    [
                        asym_id,  # asym_id
                        entity_id,  # entity_id
                        mon_id,  # mon_id
                        seq_num,  # seq_id
                        strand_id,  # pdb_strand_id
                        seq_num,  # pdb_seq_num
                        ins_code,  # pdb_ins_code
                    ]
                )


def _update_entity_poly(block: gemmi.cif.Block, structure: gemmi.Structure):
    """Update the _entity_poly_seq category in the CIF block to reflect sequences."""
    table: gemmi.cif.Table = block.find_mmcif_category("_entity_poly.")

    rows = []
    for row in table:
        rows.append(
            [
                row["entity_id"],
                row["type"],
                row["pdbx_strand_id"],
                row["pdbx_seq_one_letter_code"],
                row["pdbx_seq_one_letter_code"],
            ],
        )
    loop: gemmi.cif.Loop = block.init_mmcif_loop(
        "_entity_poly.",
        [
            "entity_id",
            "type",
            "pdbx_strand_id",
            "pdbx_seq_one_letter_code",
            "pdbx_seq_one_letter_code_can",
        ],
    )
    for row in rows:
        loop.add_row(row)


def _update_entity_poly_seq(block: gemmi.cif.Block, structure: gemmi.Structure):
    """Update the _entity_poly_seq category in the CIF block to reflect sequences."""
    table: gemmi.cif.Table = block.find_mmcif_category("_entity_poly_seq.")

    rows = []
    for row in table:
        rows.append([row["entity_id"], row["num"], row["mon_id"], "n"])
    loop: gemmi.cif.Loop = block.init_mmcif_loop(
        "_entity_poly_seq.",
        [
            "entity_id",
            "num",
            "mon_id",
            "hetero",
        ],
    )
    for row in rows:
        loop.add_row(row)


def _update_chem_comp(block: gemmi.cif.Block):
    """Add or modify the _chem_comp category in the CIF block to include residue types."""
    table: gemmi.cif.Table = block.find_mmcif_category("_chem_comp.")

    rows = []
    for row in table:
        res_id = row["id"]
        res: gemmi.ResidueInfo = gemmi.find_tabulated_residue(res_id)
        rows.append(
            [
                res_id,
                "'L-peptide linking'",
                ".",
                ".",
                f"{res.weight:.3f}",
                "y",
            ]
        )
    loop: gemmi.cif.Loop = block.init_mmcif_loop(
        "_chem_comp.",
        [
            "id",
            "type",
            "name",
            "formula",
            "formula_weight",
            "mon_nstd_flag",
        ],
    )
    for row in rows:
        loop.add_row(row)
