import numpy as np

from atlasfold.common import ccd

# one-letter codes for standard amino acids and nucleic acid bases
amino_acids: tuple[str, ...] = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "UNK"
)  # fmt: skip
restypes = [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
    "X",
]  # fmt: skip
restype_orders = {r: i for i, r in enumerate(restypes)}

restype_3to1: dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "UNK": "X",
}
restype_1to3: dict[str, str] = {v: k for k, v in restype_3to1.items()}

ambiguous_restype_mapping: dict[str, str] = {
    "B": "D",  # Aspartic acid or Asparagine
    "U": "C",  # Selenocysteine (treated as Cysteine)
    "Z": "E",  # Glutamic acid or Glutamine
}


def convert_full_sequence_to_sequence(
    full_sequence: list[str],
    standardize_ambiguous: bool = False,
) -> str:
    """Convert a full sequence (with ambiguous residues) to a standard sequence."""

    def three_to_one(aa3: str) -> str:
        aa3 = str(aa3).upper().strip()
        aa = ccd.CCD_NAME_TO_ONE_LETTER.get(aa3, "X")
        if standardize_ambiguous:
            aa = ambiguous_restype_mapping.get(aa, aa)
        if aa not in restype_orders:
            aa = "X"
        return aa

    return "".join(map(three_to_one, full_sequence))


atom_names: tuple[str, ...] = (
    "N",   "CA",  "C",   "CB",  "O",   "CG",  "CG1", "CG2", "OG",  "OG1",
    "SG",  "CD",  "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD",  "CE",
    "CE1", "CE2", "CE3", "NE",  "NE1", "NE2", "OE1", "OE2", "CH2", "NH1",
    "NH2", "OH",  "CZ",  "CZ2", "CZ3", "NZ",  "OXT"
)  # fmt: skip

residue_atoms: dict[str, tuple[str, ...]] = {
    "ALA": ("N", "CA", "C", "O", "CB"),
    "ARG": ("N", "CA", "C", "O", "CB", "CG",  "CD",  "NE",  "CZ",  "NH1", "NH2"),
    "ASN": ("N", "CA", "C", "O", "CB", "CG",  "OD1", "ND2"),
    "ASP": ("N", "CA", "C", "O", "CB", "CG",  "OD1", "OD2"),
    "CYS": ("N", "CA", "C", "O", "CB", "SG"),
    "GLN": ("N", "CA", "C", "O", "CB", "CG",  "CD",  "OE1", "NE2"),
    "GLU": ("N", "CA", "C", "O", "CB", "CG",  "CD",  "OE1", "OE2"),
    "GLY": ("N", "CA", "C", "O"),
    "HIS": ("N", "CA", "C", "O", "CB", "CG",  "ND1", "CD2", "CE1", "NE2"),
    "ILE": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"),
    "LEU": ("N", "CA", "C", "O", "CB", "CG",  "CD1", "CD2"),
    "LYS": ("N", "CA", "C", "O", "CB", "CG",  "CD",  "CE",  "NZ"),
    "MET": ("N", "CA", "C", "O", "CB", "CG",  "SD",  "CE"),
    "PHE": ("N", "CA", "C", "O", "CB", "CG",  "CD1", "CD2", "CE1", "CE2", "CZ"),
    "PRO": ("N", "CA", "C", "O", "CB", "CG",  "CD"),
    "SER": ("N", "CA", "C", "O", "CB", "OG"),
    "THR": ("N", "CA", "C", "O", "CB", "OG1", "CG2"),
    "TRP": ("N", "CA", "C", "O", "CB", "CG",  "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),  # noqa
    "TYR": ("N", "CA", "C", "O", "CB", "CG",  "CD1", "CD2", "CE1", "CE2", "CZ",  "OH"),
    "VAL": ("N", "CA", "C", "O", "CB", "CG1", "CG2"),
    "UNK": ("N", "CA", "C", "O", "CB"),
}  # fmt: skip
restype_atom14_order: dict[str, dict[str, int]] = {
    res: {atom: idx for idx, atom in enumerate(atoms)}
    for res, atoms in residue_atoms.items()
}
num_residue_atoms: dict[str, int] = {
    res: len(atoms) for res, atoms in residue_atoms.items()
}

atom37_order: dict[str, int] = {name: idx for idx, name in enumerate(atom_names)}
restype_atom37_mask: np.ndarray = np.zeros((21, 37), dtype=bool)
restype_atom14_mask: np.ndarray = np.zeros((21, 14), dtype=bool)
for res_name, atoms in residue_atoms.items():
    res_idx = restype_orders[restype_3to1[res_name]]
    restype_atom14_mask[res_idx, : len(atoms)] = True
    for atom_name in atoms:
        atom_idx = atom37_order[atom_name]
        restype_atom37_mask[res_idx, atom_idx] = True


def get_atom14_mask_from_sequence(sequence: str) -> np.ndarray:
    """Get a boolean mask indicating which atoms are present in the structure"""
    res_indices = [restype_orders[aa] for aa in sequence]
    res_indices = np.array(res_indices)
    return restype_atom14_mask[res_indices]


def get_atom37_mask_from_sequence(sequence: str) -> np.ndarray:
    """Get a boolean mask indicating which atoms are present in the structure"""
    res_indices = [restype_orders[aa] for aa in sequence]
    res_indices = np.array(res_indices)
    return restype_atom37_mask[res_indices]


def _get_atom14_to_atom37_mapping() -> tuple[np.ndarray, np.ndarray]:
    gather_indices = np.zeros((21, 37), dtype=np.int64)
    gather_mask = np.zeros((21, 37), dtype=bool)
    for res_name, atoms in residue_atoms.items():
        res_idx = restype_orders[restype_3to1[res_name]]
        for atom14_idx, atom_name in enumerate(atoms):
            if atom_name in atom37_order:
                atom37_idx = atom37_order[atom_name]
                gather_indices[res_idx, atom37_idx] = atom14_idx
                gather_mask[res_idx, atom37_idx] = True
    return gather_indices, gather_mask


_gather_indices, _gather_mask = _get_atom14_to_atom37_mapping()


# Swappable atoms for each residue type
restype_ambiguous_atoms: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "ASP": (("OD1",), ("OD2",)),
    "GLU": (("OE1",), ("OE2",)),
    "PHE": (("CD1", "CE1"), ("CD2", "CE2")),
    "TYR": (("CD1", "CE1"), ("CD2", "CE2")),
    "ARG": (("NH1",), ("NH2",)),
    "LEU": (("CD1",), ("CD2",)),
    "VAL": (("CG1",), ("CG2",)),
}

# Reference conformers for each residue
restype_ref_atom_positions: dict[str, dict[str, tuple[float, float, float]]] = {
    "ALA": {
        "N": (-0.525, 1.363, 0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.526, -0.000, -0.000),
        "O": (0.627, 1.062, 0.000),
        "CB": (-0.529, -0.774, -1.205),
    },
    "ARG": {
        "N": (-0.524, 1.362, -0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.525, -0.000, -0.000),
        "O": (0.626, 1.062, 0.000),
        "CB": (-0.524, -0.778, -1.209),
        "CG": (0.616, 1.390, -0.000),
        "CD": (0.564, 1.414, 0.000),
        "NE": (0.539, 1.357, -0.000),
        "CZ": (0.758, 1.093, -0.000),
        "NH1": (0.206, 2.301, 0.000),
        "NH2": (2.078, 0.978, -0.000),
    },
    "ASN": {
        "N": (-0.536, 1.357, 0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.526, -0.000, -0.000),
        "O": (0.625, 1.062, 0.000),
        "CB": (-0.531, -0.787, -1.200),
        "CG": (0.584, 1.399, 0.000),
        "OD1": (0.633, 1.059, 0.000),
        "ND2": (0.593, -1.188, 0.001),
    },
    "ASP": {
        "N": (-0.525, 1.362, -0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.527, 0.000, -0.000),
        "O": (0.626, 1.062, -0.000),
        "CB": (-0.526, -0.778, -1.208),
        "CG": (0.593, 1.398, -0.000),
        "OD1": (0.610, 1.091, 0.000),
        "OD2": (0.592, -1.101, -0.003),
    },
    "CYS": {
        "N": (-0.522, 1.362, -0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.524, 0.000, 0.000),
        "O": (0.625, 1.062, -0.000),
        "CB": (-0.519, -0.773, -1.212),
        "SG": (0.728, 1.653, 0.000),
    },
    "GLN": {
        "N": (-0.526, 1.361, -0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.526, 0.000, 0.000),
        "O": (0.626, 1.062, -0.000),
        "CB": (-0.525, -0.779, -1.207),
        "CG": (0.615, 1.393, 0.000),
        "CD": (0.587, 1.399, -0.000),
        "OE1": (0.634, 1.060, 0.000),
        "NE2": (0.593, -1.189, -0.001),
    },
    "GLU": {
        "N": (-0.528, 1.361, 0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.526, -0.000, -0.000),
        "O": (0.626, 1.062, 0.000),
        "CB": (-0.526, -0.781, -1.207),
        "CG": (0.615, 1.392, 0.000),
        "CD": (0.600, 1.397, 0.000),
        "OE1": (0.607, 1.095, -0.000),
        "OE2": (0.589, -1.104, -0.001),
    },
    "GLY": {
        "N": (-0.572, 1.337, 0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.517, -0.000, -0.000),
        "O": (0.626, 1.062, -0.000),
    },
    "HIS": {
        "N": (-0.527, 1.360, 0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.525, 0.000, 0.000),
        "O": (0.625, 1.063, 0.000),
        "CB": (-0.525, -0.778, -1.208),
        "CG": (0.600, 1.370, -0.000),
        "ND1": (0.744, 1.160, -0.000),
        "CD2": (0.889, -1.021, 0.003),
        "CE1": (2.030, 0.851, 0.002),
        "NE2": (2.145, -0.466, 0.004),
    },
    "ILE": {
        "N": (-0.493, 1.373, -0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.527, -0.000, -0.000),
        "O": (0.627, 1.062, -0.000),
        "CB": (-0.536, -0.793, -1.213),
        "CG1": (0.534, 1.437, -0.000),
        "CG2": (0.540, -0.785, -1.199),
        "CD1": (0.619, 1.391, 0.000),
    },
    "LEU": {
        "N": (-0.520, 1.363, 0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.525, -0.000, -0.000),
        "O": (0.625, 1.063, -0.000),
        "CB": (-0.522, -0.773, -1.214),
        "CG": (0.678, 1.371, 0.000),
        "CD1": (0.530, 1.430, -0.000),
        "CD2": (0.535, -0.774, 1.200),
    },
    "LYS": {
        "N": (-0.526, 1.362, -0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.526, 0.000, 0.000),
        "O": (0.626, 1.062, -0.000),
        "CB": (-0.524, -0.778, -1.208),
        "CG": (0.619, 1.390, 0.000),
        "CD": (0.559, 1.417, 0.000),
        "CE": (0.560, 1.416, 0.000),
        "NZ": (0.554, 1.387, 0.000),
    },
    "MET": {
        "N": (-0.521, 1.364, -0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.525, 0.000, 0.000),
        "O": (0.625, 1.062, -0.000),
        "CB": (-0.523, -0.776, -1.210),
        "CG": (0.613, 1.391, -0.000),
        "SD": (0.703, 1.695, 0.000),
        "CE": (0.320, 1.786, -0.000),
    },
    "PHE": {
        "N": (-0.518, 1.363, 0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.524, 0.000, -0.000),
        "O": (0.626, 1.062, -0.000),
        "CB": (-0.525, -0.776, -1.212),
        "CG": (0.607, 1.377, 0.000),
        "CD1": (0.709, 1.195, -0.000),
        "CD2": (0.706, -1.196, 0.000),
        "CE1": (2.102, 1.198, -0.000),
        "CE2": (2.098, -1.201, -0.000),
        "CZ": (2.794, -0.003, -0.001),
    },
    "PRO": {
        "N": (-0.566, 1.351, -0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.527, -0.000, 0.000),
        "O": (0.621, 1.066, 0.000),
        "CB": (-0.546, -0.611, -1.293),
        "CG": (0.382, 1.445, 0.0),
        "CD": (0.477, 1.424, 0.0),
    },
    "SER": {
        "N": (-0.529, 1.360, -0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.525, -0.000, -0.000),
        "O": (0.626, 1.062, -0.000),
        "CB": (-0.518, -0.777, -1.211),
        "OG": (0.503, 1.325, 0.000),
    },
    "THR": {
        "N": (-0.517, 1.364, 0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.526, 0.000, -0.000),
        "O": (0.626, 1.062, 0.000),
        "CB": (-0.516, -0.793, -1.215),
        "OG1": (0.472, 1.353, 0.000),
        "CG2": (0.550, -0.718, -1.228),
    },
    "TRP": {
        "N": (-0.521, 1.363, 0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.525, -0.000, 0.000),
        "O": (0.627, 1.062, 0.000),
        "CB": (-0.523, -0.776, -1.212),
        "CG": (0.609, 1.370, -0.000),
        "CD1": (0.824, 1.091, 0.000),
        "CD2": (0.854, -1.148, -0.005),
        "NE1": (2.140, 0.690, -0.004),
        "CE2": (2.186, -0.678, -0.007),
        "CE3": (0.622, -2.530, -0.007),
        "CZ2": (3.283, -1.543, -0.011),
        "CZ3": (1.715, -3.389, -0.011),
        "CH2": (3.028, -2.890, -0.013),
    },
    "TYR": {
        "N": (-0.522, 1.362, 0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.524, -0.000, -0.000),
        "O": (0.627, 1.062, -0.000),
        "CB": (-0.522, -0.776, -1.213),
        "CG": (0.607, 1.382, -0.000),
        "CD1": (0.716, 1.195, -0.000),
        "CD2": (0.713, -1.194, -0.001),
        "CE1": (2.107, 1.200, -0.002),
        "CE2": (2.104, -1.201, -0.003),
        "CZ": (2.791, -0.001, -0.003),
        "OH": (4.168, -0.002, -0.005),
    },
    "VAL": {
        "N": (-0.494, 1.373, -0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.527, -0.000, -0.000),
        "O": (0.627, 1.062, -0.000),
        "CB": (-0.533, -0.795, -1.213),
        "CG1": (0.540, 1.429, -0.000),
        "CG2": (0.533, -0.776, 1.203),
    },
    "UNK": {
        "N": (-0.525, 1.363, 0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.526, -0.000, -0.000),
        "O": (0.627, 1.062, 0.000),
        "CB": (-0.529, -0.774, -1.205),
    },
}
restype_atom14_positions: dict[str, np.ndarray] = {}
for res_name, atom_positions in restype_ref_atom_positions.items():
    res_idx = restype_orders[restype_3to1[res_name]]
    atom14_positions = np.zeros((14, 3), dtype=np.float32)
    for atom_name, position in atom_positions.items():
        if atom_name in restype_atom14_order[res_name]:
            atom_idx = restype_atom14_order[res_name][atom_name]
            atom14_positions[atom_idx] = position
    restype_atom14_positions[res_name] = atom14_positions
