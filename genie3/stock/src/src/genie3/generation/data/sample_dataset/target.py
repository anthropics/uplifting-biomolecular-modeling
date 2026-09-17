"""
Target-conditioned binder design sample dataset.

Provides sampling interface for binder design inference tasks.
"""

import os
import glob
import json
import numpy as np
from typing import List, Optional, Dict, Tuple
from torch.utils.data import Dataset

from genie3.generation.data.feature_pipeline.target import TargetFeaturePipeline
from genie3.generation.utils.feat_utils import batchify_np_features


class TargetSampleDataset(Dataset):
    """
    PyTorch Dataset for target-conditioned binder sampling.

    This dataset:
        • Loads binder design problems from JSON configs
        • Supports filtering by tags and explicit selections
        • Repeats each problem `n_sample` times
        • Generates target-conditioned features
        • Returns batched NumPy features for inference + corresponding sample names

    Expected directory structure:

        datadir/
            problems/
                <problem_name>.json
    """

    def __init__(
        self,
        datadir: str,
        n_sample: int,
        prot_rep_mode: str,
        cond_strategy: str,
        tags: Optional[str] = None,
        selections: Optional[str] = None,
        sample_index_offset: int = 0,
        **kwargs,
    ):
        """
        Initialize TargetSampleDataset.

        Args:
            datadir:
                Directory containing problem JSON files.
            n_sample:
                Number of samples to generate per problem.
            prot_rep_mode:
                Protein representation mode (calpha or atom14).
            cond_strategy:
                Interface conditioning strategy used during inference.
            tags:
                Comma-separated tag filter. Only problems containing
                at least one of these tags will be included.
                If None, all problems are included.
            selections:
                Comma-separated list of explicit problem names to include.
            **kwargs:
                Reserved for future extensions.
        """
        super().__init__()
        self.datadir = datadir

        # Load problems from data directory
        if tags is None:
            problems = [
                filepath.strip().split("/")[-1].split(".")[0]
                for filepath in glob.glob(os.path.join(datadir, "problems", "*.json"))
            ]
        else:
            problems = []
            tags = set(tags.strip().split(","))
            for filepath in glob.glob(os.path.join(datadir, "problems", "*.json")):
                with open(filepath) as file:
                    info = json.load(file)
                if "tag" in info:
                    for tag in info["tag"]:
                        if tag in tags:
                            problems.append(
                                filepath.strip().split("/")[-1].split(".")[0]
                            )
                            break

        # Filter problems by selections
        if selections is not None and len(selections.strip()) > 0:
            selections = set([elt.strip() for elt in selections.split(",")])
            problems = [problem for problem in problems if problem in selections]

        # Create task with names
        self.tasks = [problem for problem in problems for _ in range(n_sample)]
        self.names = [
            f"{problem}_{i}"
            for problem in problems
            for i in range(sample_index_offset, sample_index_offset + n_sample)
        ]

        # Create feature pipeline
        self.featurizer = TargetFeaturePipeline(
            prot_rep_mode=prot_rep_mode,
            inference_interface_mask_strategy=cond_strategy,
        )

    def __len__(self) -> int:
        """
        Returns total number of samples in dataset.
        """
        return len(self.tasks)

    def __getitem__(
        self, batch_idx: List[int]
    ) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """
        Generate a batch of target-conditioned features.

        For each requested index:
            1. Load corresponding problem config
            2. Create target-conditioned NumPy features
            3. Collect for batching

        Args:
            batch_idx:
                List of dataset indices (used by custom batch sampler).

        Returns:
            batch (dict):
                Batched NumPy feature dictionary.
            names (list[str]):
                Unique sample names.
        """

        # Create
        names, list_np_features = [], []
        for sample_idx in batch_idx:
            problem = self.tasks[sample_idx]
            name = self.names[sample_idx]

            # Create features
            np_features = self.featurizer.create_np_features_from_config(
                os.path.join(self.datadir, "problems", f"{problem}.json")
            )

            # Add features
            names.append(name)
            list_np_features.append(np_features)

        # Batchify
        batch = batchify_np_features(list_np_features)

        return batch, names
