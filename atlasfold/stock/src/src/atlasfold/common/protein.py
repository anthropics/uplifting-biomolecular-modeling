import dataclasses
import pathlib
from collections import defaultdict

import gemmi
import numpy as np
from typing_extensions import Self

from atlasfold.common import file_io, residue_constants


def _default_chain_ids(num_chains: int) -> list[str]:
    chain_ids = []
    for i in range(num_chains):
        chain_id = ""
        n = i
        while True:
            chain_id = chr(ord("A") + (n % 26)) + chain_id
            n //= 26
            if n == 0:
                break
        chain_ids.append(chain_id)
    return chain_ids


@dataclasses.dataclass(kw_only=True)
class Protein:
    """A data structure representing a 3D protein structure"""

    name: str
    sequence: str
    coordinates: np.ndarray  # [L, 14, 3]
    b_factors: np.ndarray  # [L, 14]
    residue_index: np.ndarray | None = None  # [L], optional residue index
    chain_id: str = "A"

    def __post_init__(self):
        """Validate the input data."""
        L = len(self.sequence)
        if self.coordinates.shape not in [(L, 14, 3)]:
            raise ValueError(
                f"Invalid coordinates shape: {self.coordinates.shape}. "
                f"Expected (L, 14, 3) where L is the sequence length."
            )
        # TODO: support the structure with missing residues
        if self.residue_index is not None:
            if self.residue_index.shape != (L,):
                raise ValueError(
                    f"Invalid residue_index shape: {self.residue_index.shape}. "
                    f"Expected (L,) where L is the sequence length."
                )

    def __len__(self):
        """Return the number of residues in the structure."""
        return len(self.sequence)

    @property
    def num_residues(self) -> int:
        """Return the number of residues in the structure."""
        return len(self.sequence)

    @classmethod
    def get_empty(cls, name: str, sequence: str, chain_id: str = "A") -> Self:
        """Create an empty structure with NaN coordinates."""
        L = len(sequence)
        coordinates = np.full((L, 14, 3), np.nan, dtype=np.float32)
        b_factors = np.full((L, 14), np.nan, dtype=np.float32)
        return cls(
            name=name,
            sequence=sequence,
            coordinates=coordinates,
            b_factors=b_factors,
            chain_id=chain_id,
        )

    @classmethod
    def create(
        cls,
        name: str,
        sequence: str,
        coordinates: np.ndarray,
        b_factors: np.ndarray,
        residue_index: np.ndarray | None = None,
        chain_id: str = "A",
    ) -> Self:
        """Create a Protein instance with the given data."""
        return cls(
            name=name,
            sequence=sequence,
            coordinates=coordinates,
            b_factors=b_factors,
            residue_index=residue_index,
            chain_id=chain_id,
        )

    @property
    def atom_mask(self) -> np.ndarray:
        """Return a boolean mask indicating which atoms are present in the structure."""
        return np.isfinite(self.coordinates).all(axis=-1)

    def to_pdb(self, model: str | None = None) -> str:
        return file_io.to_pdb(self.to_gemmi_structure(), model=model)

    def to_mmcif(self, model: str | None = None) -> str:
        return file_io.to_mmcif(self.to_gemmi_structure(), model=model)

    def to_gemmi_structure(self) -> gemmi.Structure:
        cinfo = file_io.ChainInfo(
            label_asym_id="A",
            auth_asym_id=self.chain_id,
            entity_id=1,
            sequence=self.sequence,
            coordinates=self.coordinates,
            b_factors=self.b_factors,
        )
        return file_io.to_gemmi_structure(self.name, [cinfo])

    @classmethod
    def from_pdb(
        cls,
        path: str | pathlib.Path,
        chain_id: int | str | None = None,
        name: str | None = None,
    ) -> Self:
        """Load a predicted protein structure from single monomer file."""
        if name is None:
            name = pathlib.Path(path).name.split(".")[0]

        struct: gemmi.Structure = gemmi.read_structure(str(path))
        model: gemmi.Model = struct[0]

        # Get chain
        if chain_id is not None:
            raw_chain = model[chain_id]
        else:
            raw_chain = model[0]

        # Read chain
        aa_list, coords_list, biso_list, res_idx_list = [], [], [], []
        for res in raw_chain:
            aa1 = residue_constants.restype_3to1.get(res.name, "X")
            aa3 = residue_constants.restype_1to3[aa1]
            res_atom14_order = residue_constants.restype_atom14_order[aa3]
            res_coords = np.full((14, 3), np.nan, dtype=np.float32)
            res_b_factors = np.full((14,), np.nan, dtype=np.float32)
            for atom in res:
                atom_name: str = atom.name.upper().strip()
                if atom_name in res_atom14_order:
                    atom_i: int = res_atom14_order[atom_name]
                    res_coords[atom_i, :] = atom.pos.tolist()
                    res_b_factors[atom_i] = atom.b_iso
            aa_list.append(aa1)
            coords_list.append(res_coords)
            biso_list.append(res_b_factors)
            res_idx_list.append(res.seqid.num)

        sequence = "".join(aa_list)
        coordinates = np.stack(coords_list, axis=0)  # [L, 14, 3]
        b_factors = np.stack(biso_list, axis=0)  # [L, 14]
        residue_index = np.array(res_idx_list, dtype=np.int32)  # [L,]
        return cls.create(
            name,
            sequence,
            coordinates,
            b_factors,
            residue_index,
            chain_id=str(raw_chain.name),
        )


@dataclasses.dataclass(kw_only=True)
class ProteinMultimer:
    """A data structure representing a protein complex"""

    name: str
    chains: list[Protein]
    entity_ids: list[int] = dataclasses.field(default_factory=list)
    asym_ids: list[int] = dataclasses.field(default_factory=list)
    sym_ids: list[int] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        num_chains = len(self.chains)
        if num_chains == 0:
            raise ValueError("ProteinMultimer must contain at least one chain.")

        def check_length(ids, name):
            if ids is not None and len(ids) not in (0, num_chains):
                raise ValueError(
                    f"Length of {name} ({len(ids)}) must match "
                    f"the number of sequences ({num_chains})."
                )

        check_length(self.entity_ids, "entity_ids")
        check_length(self.asym_ids, "asym_ids")
        check_length(self.sym_ids, "sym_ids")

        if len(self.entity_ids) == 0:
            seq_to_eid = {}
            i = 1
            for c in self.chains:
                if c.sequence not in seq_to_eid:
                    seq_to_eid[c.sequence] = i
                    i += 1
            self.entity_ids = [seq_to_eid[c.sequence] for c in self.chains]
        if len(self.asym_ids) == 0:
            self.asym_ids = list(range(1, num_chains + 1))
        if len(self.sym_ids) == 0:
            seq_to_symid = defaultdict(int)
            sym_ids = []
            for c in self.chains:
                seq_to_symid[c.sequence] += 1
                sym_ids.append(seq_to_symid[c.sequence])
            self.sym_ids = sym_ids

    @classmethod
    def get_empty(
        cls,
        name: str,
        sequence: list[str],
        chain_ids: list[str] | None = None,
        entity_ids: list[int] | None = None,
        asym_ids: list[int] | None = None,
        sym_ids: list[int] | None = None,
    ) -> Self:
        """Create an empty structure with NaN coordinates."""
        # Check that the lengths of chain_ids and entity_ids match the
        # number of sequences
        num_chains = len(sequence)

        def check_length(ids, name):
            if ids is not None and len(ids) != num_chains:
                raise ValueError(
                    f"Length of {name} ({len(ids)}) must match "
                    f"the number of sequences ({num_chains})."
                )

        check_length(chain_ids, "chain_ids")
        check_length(entity_ids, "entity_ids")
        check_length(asym_ids, "asym_ids")
        check_length(sym_ids, "sym_ids")

        if chain_ids is None:
            chain_ids = _default_chain_ids(len(sequence))

        chains = [
            Protein.get_empty(chain_id, seq, chain_id=chain_id)
            for chain_id, seq in zip(chain_ids, sequence, strict=True)
        ]
        return cls(
            name=name,
            chains=chains,
            entity_ids=entity_ids or [],
            asym_ids=asym_ids or [],
            sym_ids=sym_ids or [],
        )

    def __len__(self):
        """Return the number of chains in the complex."""
        return self.num_chains

    @property
    def num_chains(self) -> int:
        """Return the number of chains in the complex."""
        return len(self.chains)

    @property
    def sequences(self) -> list[str]:
        """
        Return the list of sequences for each chain in the complex.
        """
        return [c.sequence for c in self.chains]

    @property
    def sequence(self) -> str:
        """
        Return the concatenated sequence of all chains in the complex,
        separated by colons.
        """
        return ":".join(self.sequences)

    @property
    def num_residues(self) -> int:
        """Return the number of residues in the structure."""
        return sum(c.num_residues for c in self.chains)

    def to_pdb(self, model: str | None = None) -> str:
        return file_io.to_pdb(self.to_gemmi_structure(), model=model)

    def to_mmcif(self, model: str | None = None) -> str:
        return file_io.to_mmcif(self.to_gemmi_structure(), model=model)

    def to_gemmi_structure(self) -> gemmi.Structure:
        label_asym_ids = _default_chain_ids(len(self.chains))
        cinfos = [
            file_io.ChainInfo(
                label_asym_id=label_asym_id,
                auth_asym_id=c.name,
                entity_id=eid,
                sequence=c.sequence,
                coordinates=c.coordinates,
                b_factors=c.b_factors,
            )
            for label_asym_id, eid, c in zip(
                label_asym_ids, self.entity_ids, self.chains, strict=True
            )
        ]
        return file_io.to_gemmi_structure(self.name, cinfos)
