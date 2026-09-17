# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Data-side of the input features processing."""

import dataclasses
from typing import Self, TypeAlias

import torch

from xfold.nn import atom_layout


BatchDict: TypeAlias = dict[str, torch.Tensor]


def _unwrap(obj):
    """Unwrap an object from a zero-dim np.ndarray."""
    if isinstance(obj, torch.Tensor) and obj.ndim == 0:
        return obj.item()
    else:
        return obj


@dataclasses.dataclass(frozen=True)
class MSA:
    """Dataclass containing MSA."""

    rows: torch.Tensor
    mask: torch.Tensor
    deletion_matrix: torch.Tensor
    # Occurrence of each residue type along the sequence, averaged over MSA rows.
    profile: torch.Tensor
    # Occurrence of deletions along the sequence, averaged over MSA rows.
    deletion_mean: torch.Tensor
    # Number of MSA alignments.
    num_alignments: torch.Tensor

    def index_msa_rows(self, indices: torch.Tensor) -> Self:
        assert indices.ndim == 1

        return MSA(
            rows=self.rows[indices, :],
            mask=self.mask[indices, :],
            deletion_matrix=self.deletion_matrix[indices, :],
            profile=self.profile,
            deletion_mean=self.deletion_mean,
            num_alignments=self.num_alignments,
        )

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        output = cls(
            rows=batch['msa'],
            mask=batch['msa_mask'],
            deletion_matrix=batch['deletion_matrix'],
            profile=batch['profile'],
            deletion_mean=batch['deletion_mean'],
            num_alignments=batch['num_alignments'],
        )
        return output

    def as_data_dict(self) -> BatchDict:
        return {
            'msa': self.rows,
            'msa_mask': self.mask,
            'deletion_matrix': self.deletion_matrix,
            'profile': self.profile,
            'deletion_mean': self.deletion_mean,
            'num_alignments': self.num_alignments,
        }


@dataclasses.dataclass(frozen=True)
class Templates:
    """Dataclass containing templates."""

    # aatype of templates, int32 w shape [num_templates, num_res]
    aatype: torch.Tensor
    # atom positions of templates, float32 w shape [num_templates, num_res, 24, 3]
    atom_positions: torch.Tensor
    # atom mask of templates, bool w shape [num_templates, num_res, 24]
    atom_mask: torch.Tensor

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        """Make Template from batch dictionary."""
        return cls(
            aatype=batch['template_aatype'],
            atom_positions=batch['template_atom_positions'],
            atom_mask=batch['template_atom_mask'],
        )

    def as_data_dict(self) -> BatchDict:
        return {
            'template_aatype': self.aatype,
            'template_atom_positions': self.atom_positions,
            'template_atom_mask': self.atom_mask,
        }
    
    def __getitem__(self, index: int) -> Self:
        return Templates(
            aatype=self.aatype[index],
            atom_positions=self.atom_positions[index],
            atom_mask=self.atom_mask[index],
        )


@dataclasses.dataclass(frozen=True)
class TokenFeatures:
    """Dataclass containing features for tokens."""

    residue_index: torch.Tensor
    token_index: torch.Tensor
    aatype: torch.Tensor
    mask: torch.Tensor
    seq_length: torch.Tensor

    # Chain symmetry identifiers
    # for an A3B2 stoichiometry the meaning of these features is as follows:
    # asym_id:    1 2 3 4 5
    # entity_id:  1 1 1 2 2
    # sym_id:     1 2 3 1 2
    asym_id: torch.Tensor
    entity_id: torch.Tensor
    sym_id: torch.Tensor

    # token type features
    is_protein: torch.Tensor
    is_rna: torch.Tensor
    is_dna: torch.Tensor
    is_ligand: torch.Tensor
    is_nonstandard_polymer_chain: torch.Tensor
    is_water: torch.Tensor

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        return cls(
            residue_index=batch['residue_index'],
            token_index=batch['token_index'],
            aatype=batch['aatype'],
            mask=batch['seq_mask'],
            entity_id=batch['entity_id'],
            asym_id=batch['asym_id'],
            sym_id=batch['sym_id'],
            seq_length=batch['seq_length'],
            is_protein=batch['is_protein'],
            is_rna=batch['is_rna'],
            is_dna=batch['is_dna'],
            is_ligand=batch['is_ligand'],
            is_nonstandard_polymer_chain=batch['is_nonstandard_polymer_chain'],
            is_water=batch['is_water'],
        )

    def as_data_dict(self) -> BatchDict:
        return {
            'residue_index': self.residue_index,
            'token_index': self.token_index,
            'aatype': self.aatype,
            'seq_mask': self.mask,
            'entity_id': self.entity_id,
            'asym_id': self.asym_id,
            'sym_id': self.sym_id,
            'seq_length': self.seq_length,
            'is_protein': self.is_protein,
            'is_rna': self.is_rna,
            'is_dna': self.is_dna,
            'is_ligand': self.is_ligand,
            'is_nonstandard_polymer_chain': self.is_nonstandard_polymer_chain,
            'is_water': self.is_water,
        }


@dataclasses.dataclass(frozen=True)
class RefStructure:
    """Contains ref structure information."""

    # Array with positions, float32, shape [num_res, max_atoms_per_token, 3]
    positions: torch.Tensor
    # Array with masks, bool, shape [num_res, max_atoms_per_token]
    mask: torch.Tensor
    # Array with elements, int32, shape [num_res, max_atoms_per_token]
    element: torch.Tensor
    # Array with charges, float32, shape [num_res, max_atoms_per_token]
    charge: torch.Tensor
    # Array with atom name characters, int32, [num_res, max_atoms_per_token, 4]
    atom_name_chars: torch.Tensor
    # Array with reference space uids, int32, [num_res, max_atoms_per_token]
    ref_space_uid: torch.Tensor

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        return cls(
            positions=batch['ref_pos'],
            mask=batch['ref_mask'],
            element=batch['ref_element'],
            charge=batch['ref_charge'],
            atom_name_chars=batch['ref_atom_name_chars'],
            ref_space_uid=batch['ref_space_uid'],
        )

    def as_data_dict(self) -> BatchDict:
        return {
            'ref_pos': self.positions,
            'ref_mask': self.mask,
            'ref_element': self.element,
            'ref_charge': self.charge,
            'ref_atom_name_chars': self.atom_name_chars,
            'ref_space_uid': self.ref_space_uid,
        }


@dataclasses.dataclass(frozen=True)
class PredictedStructureInfo:
    """Contains information necessary to work with predicted structure."""

    atom_mask: torch.Tensor
    residue_center_index: torch.Tensor

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        return cls(
            atom_mask=batch['pred_dense_atom_mask'],
            residue_center_index=batch['residue_center_index'],
        )

    def as_data_dict(self) -> BatchDict:
        return {
            'pred_dense_atom_mask': self.atom_mask,
            'residue_center_index': self.residue_center_index,
        }


@dataclasses.dataclass(frozen=True)
class PolymerLigandBondInfo:
    """Contains information about polymer-ligand bonds."""

    tokens_to_polymer_ligand_bonds: atom_layout.GatherInfo
    # Gather indices to convert from cropped dense atom layout to bonds layout
    # (num_tokens, 2)
    token_atoms_to_bonds: atom_layout.GatherInfo

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        return cls(
            tokens_to_polymer_ligand_bonds=atom_layout.GatherInfo.from_dict(
                batch, key_prefix='tokens_to_polymer_ligand_bonds'
            ),
            token_atoms_to_bonds=atom_layout.GatherInfo.from_dict(
                batch, key_prefix='token_atoms_to_polymer_ligand_bonds'
            ),
        )

    def as_data_dict(self) -> BatchDict:
        return {
            **self.tokens_to_polymer_ligand_bonds.as_dict(
                key_prefix='tokens_to_polymer_ligand_bonds'
            ),
            **self.token_atoms_to_bonds.as_dict(
                key_prefix='token_atoms_to_polymer_ligand_bonds'
            ),
        }


@dataclasses.dataclass(frozen=True)
class LigandLigandBondInfo:
    """Contains information about the location of ligand-ligand bonds."""

    tokens_to_ligand_ligand_bonds: atom_layout.GatherInfo

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        return cls(
            tokens_to_ligand_ligand_bonds=atom_layout.GatherInfo.from_dict(
                batch, key_prefix='tokens_to_ligand_ligand_bonds'
            )
        )

    def as_data_dict(self) -> BatchDict:
        return {
            **self.tokens_to_ligand_ligand_bonds.as_dict(
                key_prefix='tokens_to_ligand_ligand_bonds'
            )
        }


@dataclasses.dataclass(frozen=True)
class PseudoBetaInfo:
    """Contains information for extracting pseudo-beta and equivalent atoms."""

    token_atoms_to_pseudo_beta: atom_layout.GatherInfo

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        return cls(
            token_atoms_to_pseudo_beta=atom_layout.GatherInfo.from_dict(
                batch, key_prefix='token_atoms_to_pseudo_beta'
            ),
        )

    def as_data_dict(self) -> BatchDict:
        return {
            **self.token_atoms_to_pseudo_beta.as_dict(
                key_prefix='token_atoms_to_pseudo_beta'
            ),
        }


@dataclasses.dataclass(frozen=True)
class AtomCrossAtt:
    """Operate on flat atoms."""

    token_atoms_to_queries: atom_layout.GatherInfo
    tokens_to_queries: atom_layout.GatherInfo
    tokens_to_keys: atom_layout.GatherInfo
    queries_to_keys: atom_layout.GatherInfo
    queries_to_token_atoms: atom_layout.GatherInfo

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        return cls(
            token_atoms_to_queries=atom_layout.GatherInfo.from_dict(
                batch, key_prefix='token_atoms_to_queries'
            ),
            tokens_to_queries=atom_layout.GatherInfo.from_dict(
                batch, key_prefix='tokens_to_queries'
            ),
            tokens_to_keys=atom_layout.GatherInfo.from_dict(
                batch, key_prefix='tokens_to_keys'
            ),
            queries_to_keys=atom_layout.GatherInfo.from_dict(
                batch, key_prefix='queries_to_keys'
            ),
            queries_to_token_atoms=atom_layout.GatherInfo.from_dict(
                batch, key_prefix='queries_to_token_atoms'
            ),
        )

    def as_data_dict(self) -> BatchDict:
        return {
            **self.token_atoms_to_queries.as_dict(
                key_prefix='token_atoms_to_queries'
            ),
            **self.tokens_to_queries.as_dict(key_prefix='tokens_to_queries'),
            **self.tokens_to_keys.as_dict(key_prefix='tokens_to_keys'),
            **self.queries_to_keys.as_dict(key_prefix='queries_to_keys'),
            **self.queries_to_token_atoms.as_dict(
                key_prefix='queries_to_token_atoms'
            ),
        }


@dataclasses.dataclass(frozen=True)
class ConvertModelOutput:
    """Contains atom layout info."""

    # cleaned_struc: structure.Structure
    # token_atoms_layout: atom_layout.AtomLayout
    # flat_output_layout: atom_layout.AtomLayout
    # empty_output_struc: structure.Structure
    # polymer_ligand_bonds: atom_layout.AtomLayout
    # ligand_ligand_bonds: atom_layout.AtomLayout

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        """Construct atom layout object from dictionary."""

        return cls(
            # cleaned_struc=_unwrap(batch.get('cleaned_struc', None)),
            # token_atoms_layout=_unwrap(batch.get('token_atoms_layout', None)),
            # flat_output_layout=_unwrap(batch.get('flat_output_layout', None)),
            # empty_output_struc=_unwrap(batch.get('empty_output_struc', None)),
            # polymer_ligand_bonds=_unwrap(
            #     batch.get('polymer_ligand_bonds', None)),
            # ligand_ligand_bonds=_unwrap(
            #     batch.get('ligand_ligand_bonds', None)),
        )


@dataclasses.dataclass(frozen=True)
class Frames:
    """Features for backbone frames."""

    mask: torch.Tensor

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        return cls(mask=batch['frames_mask'])

    def as_data_dict(self) -> BatchDict:
        return {'frames_mask': self.mask}
