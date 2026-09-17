# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Batch dataclass."""
from typing import Self
import dataclasses

from xfold import features


@dataclasses.dataclass(frozen=True)
class Batch:
    """Dataclass containing batch."""

    msa: features.MSA
    templates: features.Templates
    token_features: features.TokenFeatures
    ref_structure: features.RefStructure
    predicted_structure_info: features.PredictedStructureInfo
    polymer_ligand_bond_info: features.PolymerLigandBondInfo
    ligand_ligand_bond_info: features.LigandLigandBondInfo
    pseudo_beta_info: features.PseudoBetaInfo
    atom_cross_att: features.AtomCrossAtt
    convert_model_output: features.ConvertModelOutput
    frames: features.Frames

    @property
    def num_res(self) -> int:
        return self.token_features.aatype.shape[-1]

    @classmethod
    def from_data_dict(cls, batch: features.BatchDict) -> Self:
        """Construct batch object from dictionary."""
        return cls(
            msa=features.MSA.from_data_dict(batch),
            templates=features.Templates.from_data_dict(batch),
            token_features=features.TokenFeatures.from_data_dict(batch),
            ref_structure=features.RefStructure.from_data_dict(batch),
            predicted_structure_info=features.PredictedStructureInfo.from_data_dict(
                batch
            ),
            polymer_ligand_bond_info=features.PolymerLigandBondInfo.from_data_dict(
                batch
            ),
            ligand_ligand_bond_info=features.LigandLigandBondInfo.from_data_dict(
                batch
            ),
            pseudo_beta_info=features.PseudoBetaInfo.from_data_dict(batch),
            atom_cross_att=features.AtomCrossAtt.from_data_dict(batch),
            convert_model_output=features.ConvertModelOutput.from_data_dict(
                batch),
            frames=features.Frames.from_data_dict(batch),
        )

    def as_data_dict(self) -> features.BatchDict:
        """Converts batch object to dictionary."""
        output = {
            **self.msa.as_data_dict(),
            **self.templates.as_data_dict(),
            **self.token_features.as_data_dict(),
            **self.ref_structure.as_data_dict(),
            **self.predicted_structure_info.as_data_dict(),
            **self.polymer_ligand_bond_info.as_data_dict(),
            **self.ligand_ligand_bond_info.as_data_dict(),
            **self.pseudo_beta_info.as_data_dict(),
            **self.atom_cross_att.as_data_dict(),
            **self.convert_model_output.as_data_dict(),
            **self.frames.as_data_dict(),
        }
        return output
