"""
Target-conditioned binder design feature pipeline.

Generates features for binder design tasks with interface conditioning.
"""

import logging
import numpy as np
from typing import Optional, Dict, Tuple

from genie3.generation.data.feature_pipeline.base import FeaturePipeline
from genie3.generation.np.protein_constants import ATOM_TYPES, PROTEIN_RESTYPES
from genie3.generation.utils.feat_utils import (
    ProteinRepresentationMode,
    concatenate_np_features,
    create_np_features_from_chain,
    create_np_features_from_target_config,
    update_np_features_with_ca_centering,
)
from genie3.generation.utils.pdb_utils import Chain, Structure


class TargetFeaturePipeline(FeaturePipeline):
    """
    Feature pipeline for binder design.
    """

    def __init__(
        self,
        prot_rep_mode: str,
        include_target_interface: Optional[bool] = None,
        include_target_interface_backbone: Optional[bool] = None,
        include_target_interface_sidechain: Optional[bool] = None,
        include_binder_interface: Optional[bool] = None,
        include_binder_interface_backbone: Optional[bool] = None,
        include_binder_interface_sidechain: Optional[bool] = None,
        include_binder_non_interface: Optional[bool] = None,
        interface_cb_distance_threshold_min: Optional[int] = None,
        interface_cb_distance_threshold_max: Optional[int] = None,
        interface_mask_strategy: Optional[str] = None,
        inference_interface_mask_strategy: Optional[str] = "interface",
        **kwargs,
    ):
        """
        Initialize target conditioning pipeline.

        Args:
            prot_rep_mode:
                Protein representation mode (Calpha or Atom14).
            include_target_interface:
                Whether to condition on target interface residues.
            include_target_interface_backbone:
                Include backbone atoms of target interface.
            include_target_interface_sidechain:
                Include sidechain atoms of target interface.
            include_binder_interface:
                Whether to condition on binder interface residues.
            include_binder_interface_backbone:
                Include backbone atoms of binder interface.
            include_binder_interface_sidechain:
                Include sidechain atoms of binder interface.
            include_binder_non_interface:
                Whether to condition on binder non-interface residues.
            interface_cb_distance_threshold_min / max:
                Randomized Cβ distance threshold range for interface detection.
            interface_mask_strategy:
                Strategy for randomly masking interface residues during training.
                Example: "random_10_30" masks 10–30% randomly.
            inference_interface_mask_strategy:
                Strategy used during inference.

        Raises:
            Exits if unsupported representation mode is provided.
        """
        super().__init__()
        self.prot_rep_mode = prot_rep_mode

        # Training-specific parameters: Target conditioning flags
        self.include_target_interface = include_target_interface
        self.include_target_interface_backbone = include_target_interface_backbone
        self.include_target_interface_sidechain = include_target_interface_sidechain

        # Training-specific parameters: Binder conditioning flags
        self.include_binder_interface = include_binder_interface
        self.include_binder_interface_backbone = include_binder_interface_backbone
        self.include_binder_interface_sidechain = include_binder_interface_sidechain
        self.include_binder_non_interface = include_binder_non_interface

        # Training-specific parameters: Interface masking
        self.interface_mask_strategy = interface_mask_strategy
        self.interface_cb_distance_threshold_min = interface_cb_distance_threshold_min
        self.interface_cb_distance_threshold_max = interface_cb_distance_threshold_max

        # Inference-specific parameters
        self.inference_interface_mask_strategy = inference_interface_mask_strategy

        # Sanity check on protein representation mode
        if self.prot_rep_mode not in [
            ProteinRepresentationMode.Calpha.value,
            ProteinRepresentationMode.Atom14.value,
        ]:
            logging.error(
                "Current target feature pipeline only supports calpha/atom14 representation"
            )
            exit(0)

    def create_np_features(self, structure: Structure) -> Dict[str, np.ndarray]:
        """
        Create binder–target conditioned NumPy features.

        Steps:
            1. Validate dimer structure
            2. Randomly assign binder/target chains
            3. Compute residue interface mask
            4. Build conditioning masks
            5. Generate per-chain features
            6. Concatenate features
            7. Center on target CA atoms

        Args:
            structure:
                Structure containing exactly 2 chains.

        Returns:
            np_features:
                Feature dictionary for model input.
        """

        # Sanity check
        if len(structure.chains) != 2:
            logging.error("Current binder design capability only supports dimer")
            exit(0)

        # Random chain permutation
        if np.random.rand() > 0.5:
            binder = structure.chains[1]
            target = structure.chains[0]
        else:
            binder = structure.chains[0]
            target = structure.chains[1]

        # Compute residue interface mask
        binder_residue_interface_mask, target_residue_interface_mask = (
            self._compute_residue_interface(binder=binder, target=target)
        )

        ###########################
        ###   Binder Features   ###
        ###########################

        # Sanity check
        if self.include_binder_interface and self.include_binder_non_interface:
            logging.error(
                """
                Conditioning on both binder interface and  non-interface is not allowed.
            """
            )
            exit(0)
        if self.include_binder_interface:
            if (
                not self.include_binder_interface_backbone
                and self.include_binder_interface_sidechain
            ):
                logging.error(
                    """
                    Conditioning on binder interface sidechain requires conditioning on binder
                    interface backbone enabled.
                """
                )
                exit(0)
        else:
            if (
                self.include_binder_interface_backbone
                or self.include_binder_interface_sidechain
            ):
                logging.error(
                    """
                    Conditioning on binder interface is disabled. This conflicts with
                    include_binder_interface_backbone and/or include_binder_interface_sidechain.
                """
                )
                exit(0)

        # Create conditional features
        if self.include_binder_interface:
            binder_residue_cond_group = binder_residue_interface_mask.astype(np.int32)
            binder_residue_cond_mask = binder_residue_cond_group > 0
            if self.prot_rep_mode == ProteinRepresentationMode.Calpha.value:
                binder_residue_cond_atomize_mask = np.zeros_like(
                    binder_residue_cond_mask
                )
                binder_residue_cond_include_backbone = (
                    binder_residue_cond_mask * self.include_binder_interface_backbone
                )
                binder_residue_cond_include_sidechain = np.zeros_like(
                    binder_residue_cond_mask
                )
            else:
                binder_residue_cond_atomize_mask = binder_residue_cond_mask
                binder_residue_cond_include_backbone = (
                    binder_residue_cond_mask * self.include_binder_interface_backbone
                )
                binder_residue_cond_include_sidechain = (
                    binder_residue_cond_mask * self.include_binder_interface_sidechain
                )
        elif self.include_binder_non_interface:
            binder_residue_cond_group = 1 - binder_residue_interface_mask.astype(
                np.int32
            )
            binder_residue_cond_mask = binder_residue_cond_group > 0
            binder_residue_cond_atomize_mask = np.zeros_like(binder_residue_cond_mask)
            binder_residue_cond_include_backbone = binder_residue_cond_mask
            binder_residue_cond_include_sidechain = np.zeros_like(
                binder_residue_cond_mask
            )
        else:
            binder_residue_cond_group = np.zeros_like(binder_residue_interface_mask)
            binder_residue_cond_mask = binder_residue_cond_group
            binder_residue_cond_atomize_mask = np.zeros_like(binder_residue_cond_mask)
            binder_residue_cond_include_backbone = np.zeros_like(
                binder_residue_cond_mask
            )
            binder_residue_cond_include_sidechain = np.zeros_like(
                binder_residue_cond_mask
            )

        # Create features
        binder_np_features = create_np_features_from_chain(
            asym_id=0,
            sym_id=0,
            entity_id=0,
            chain=binder,
            residue_cond_group=binder_residue_cond_group,
            residue_cond_atomize_mask=binder_residue_cond_atomize_mask,
            residue_cond_include_backbone=binder_residue_cond_include_backbone,
            residue_cond_include_sidechain=binder_residue_cond_include_sidechain,
        )
        token_index_offset = len(binder_np_features["token_index"])
        cond_group_offset = (
            0  # TODO: fix this to binder_np_features['cond_group'].max()
        )

        ###########################
        ###   Target Features   ###
        ###########################

        # Sanity check
        if self.include_target_interface:
            if (
                not self.include_target_interface_backbone
                and self.include_target_interface_sidechain
            ):
                logging.error(
                    """
                    Conditioning on target interface sidechain requires conditioning on target
                    interface backbone enabled.
                """
                )
                exit(0)
        else:
            if (
                self.include_target_interface_backbone
                or self.include_target_interface_sidechain
            ):
                logging.error(
                    """
                    Conditioning on target interface is disabled. This conflicts with
                    include_target_interface_backbone and/or include_target_interface_sidechain.
                """
                )
                exit(0)

        # Create conditional features
        target_residue_cond_group = (
            np.ones_like(target_residue_interface_mask) + cond_group_offset
        )
        if self.prot_rep_mode == ProteinRepresentationMode.Calpha.value:
            target_residue_cond_atomize_mask = np.zeros_like(
                target_residue_interface_mask
            )
            target_residue_cond_include_backbone = (
                target_residue_interface_mask * self.include_target_interface_backbone
                + (1 - target_residue_interface_mask)
            )
            target_residue_cond_include_sidechain = np.zeros_like(
                target_residue_interface_mask
            )
        else:
            target_residue_cond_atomize_mask = target_residue_interface_mask
            target_residue_cond_include_backbone = (
                target_residue_interface_mask * self.include_target_interface_backbone
                + (1 - target_residue_interface_mask)
            )
            target_residue_cond_include_sidechain = (
                target_residue_interface_mask * self.include_target_interface_sidechain
            )

        # Create features
        target_np_features = create_np_features_from_chain(
            asym_id=1,
            sym_id=0,
            entity_id=1,
            chain=target,
            residue_cond_group=target_residue_cond_group,
            residue_cond_atomize_mask=target_residue_cond_atomize_mask,
            residue_cond_include_backbone=target_residue_cond_include_backbone,
            residue_cond_include_sidechain=target_residue_cond_include_sidechain,
            residue_cond_interface_mask=target_residue_interface_mask,
            token_index_offset=token_index_offset,
        )

        # Concatenate
        np_features = concatenate_np_features([binder_np_features, target_np_features])

        # Center atom positions by the Ca atoms of the target structure
        np_features = update_np_features_with_ca_centering(np_features, mode="nonfirst")

        return np_features

    def create_np_features_from_config(self, filepath: str) -> Dict[str, np.ndarray]:
        """
        Create features from a configuration file (inference mode).

        Args:
            filepath:
                Path to target configuration file.

        Returns:
            np_features (dict):
                Feature dictionary ready for inference.
        """
        return create_np_features_from_target_config(
            filepath=filepath,
            prot_rep_mode=self.prot_rep_mode,
            interface_mode=self.inference_interface_mask_strategy,
        )

    ################################
    ###     Helper Functions     ###
    ################################

    def _compute_residue_interface(
        self, binder: Chain, target: Chain
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute residue-level interface masks using Cβ distances.

        Steps:
            1. Extract Cβ (or CA for GLY/unknown) positions
            2. Compute pairwise distances
            3. Apply randomized distance threshold
            4. Optionally apply random masking strategy

        Returns:
            binder_residue_interface_mask (np.ndarray)
            target_residue_interface_mask (np.ndarray)
        """

        def is_float(s: str) -> bool:
            try:
                float(s)
                return True
            except ValueError:
                return False

        def get_cbeta(chain: Chain) -> Tuple[np.ndarray, np.ndarray]:
            """
            Extract Cβ coordinates for each residue.
            For GLY or unknown residues, use CA instead.
            """
            cb_positions, cb_mask = [], []
            for residue in chain.residues:
                is_unk = np.sum(residue.restype) == 0
                is_gly = residue.restype[PROTEIN_RESTYPES.index("GLY")]
                cb_positions.append(
                    residue.gt_atom_positions[ATOM_TYPES.index("CA")]
                    if is_unk or is_gly
                    else residue.gt_atom_positions[ATOM_TYPES.index("CB")]
                )
                cb_mask.append(
                    residue.gt_atom_mask[ATOM_TYPES.index("CA")]
                    if is_unk or is_gly
                    else residue.gt_atom_mask[ATOM_TYPES.index("CB")]
                )
            return np.stack(cb_positions), np.stack(cb_mask)

        # Compute Cb atom positions with corresponding atom masks
        binder_cb_positions, binder_cb_mask = get_cbeta(binder)
        target_cb_positions, target_cb_mask = get_cbeta(target)

        # Compute pairwise distances between binder and target residues
        d_binder_target = (
            np.sum(
                (
                    binder_cb_positions[..., None, :]
                    - target_cb_positions[..., None, :, :]
                )
                ** 2,
                axis=-1,
            )
            ** 0.5
        )
        mask_binder_target = binder_cb_mask[..., None] * target_cb_mask[..., None, :]

        # Detect interface
        interface_cb_distance_threshold = (
            np.random.rand()
            * (
                self.interface_cb_distance_threshold_max
                - self.interface_cb_distance_threshold_min
            )
            + self.interface_cb_distance_threshold_min
        )
        is_interface = (
            d_binder_target <= interface_cb_distance_threshold
        ) * mask_binder_target

        # Create residue interface mask
        binder_residue_interface_mask = np.sum(is_interface, axis=1) > 0
        target_residue_interface_mask = np.sum(is_interface, axis=0) > 0

        # Apply interface mask
        if self.interface_mask_strategy:
            if self.interface_mask_strategy.startswith("random_"):
                if is_float(self.interface_mask_strategy.split("_")[-2]):
                    max_mask_ratio = (
                        float(self.interface_mask_strategy.split("_")[-1]) / 100
                    )
                    min_mask_ratio = (
                        float(self.interface_mask_strategy.split("_")[-2]) / 100
                    )
                    mask_ratio = (
                        min_mask_ratio
                        + (max_mask_ratio - min_mask_ratio) * np.random.rand()
                    )
                else:
                    mask_ratio = (
                        float(self.interface_mask_strategy.split("_")[-1]) / 100
                    )
                binder_residue_interface_mask = binder_residue_interface_mask * (
                    np.random.rand(*binder_residue_interface_mask.shape) > mask_ratio
                )
                target_residue_interface_mask = target_residue_interface_mask * (
                    np.random.rand(*target_residue_interface_mask.shape) > mask_ratio
                )
            else:
                logging.error(
                    f"Invalid interface mask strategy: {self.interface_mask_strategy}"
                )
                exit(0)

        return binder_residue_interface_mask, target_residue_interface_mask
