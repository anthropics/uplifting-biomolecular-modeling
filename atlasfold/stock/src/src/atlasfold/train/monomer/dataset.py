import dataclasses
import io
import logging
import pathlib

import lmdb
import msgpack
import numpy as np
import torch
import torch.nn.functional as F

from atlasfold.common import featurize, metadata, protein, residue_constants
from atlasfold.train.monomer.cropper import ProteinCropper
from atlasfold.utils.geometry.random_augment import do_centering_atom14
from atlaslm.alphabet import Alphabet


class DataPipeline:
    @staticmethod
    def save(prot: protein.Protein, path: str | pathlib.Path):
        np.savez_compressed(path, **DataPipeline._to_arr_dict(prot))

    @staticmethod
    def load(path: str | pathlib.Path | io.BytesIO) -> protein.Protein:
        # Load the structure from compressed npz format
        with np.load(path) as data:
            return DataPipeline._from_arr_dict(data)

    @staticmethod
    def _to_arr_dict(prot: protein.Protein) -> dict:
        name_arr = np.array([prot.name], dtype="S")
        seq_arr = np.array([prot.sequence], dtype="S")
        # Only save the valid atom coordinates (include unresolved atoms as NaN)
        pad_mask = residue_constants.get_atom14_mask_from_sequence(prot.sequence)
        coords_arr = prot.coordinates[pad_mask]  # [Natom, 3]
        b_factors_arr = prot.b_factors[pad_mask]

        data: dict[str, np.ndarray] = {
            "name": name_arr,
            "sequence": seq_arr,
            "coordinates": coords_arr,
            "b_factors": b_factors_arr,
        }
        if prot.residue_index is not None:
            L = len(prot.sequence)
            if not np.array_equal(prot.residue_index, np.arange(1, L + 1)):
                raise NotImplementedError(
                    "Structure with missing residues is not supported yet."
                )
        return data

    @staticmethod
    def _from_arr_dict(data: dict) -> protein.Protein:
        name = data["name"].item().decode("utf-8")
        sequence = data["sequence"].item().decode("utf-8")
        coords_arr = data["coordinates"]  # [Natom, 3]
        b_factors_arr = data["b_factors"]  # [Natom]
        L = len(sequence)
        full_coords = np.full((L, 14, 3), np.nan, dtype=np.float32)
        pad_mask = residue_constants.get_atom14_mask_from_sequence(sequence)
        full_coords[pad_mask] = coords_arr
        full_b_factors = np.full((L, 14), np.nan, dtype=np.float32)
        full_b_factors[pad_mask] = b_factors_arr
        coordinates = full_coords
        b_factors = full_b_factors
        residue_index = data.get("residue_index", None)
        if residue_index is not None:
            if not np.array_equal(residue_index, np.arange(1, L + 1)):
                raise NotImplementedError(
                    "Structure with missing residues is not supported yet."
                )
        return protein.Protein.create(name, sequence, coordinates, b_factors)


def pad_input(
    data: dict[str, torch.Tensor],
    max_length: int | None = None,
    multiple_of: int | None = None,
) -> dict[str, torch.Tensor]:
    def pad_fn(x: torch.Tensor) -> torch.Tensor:
        if max_length is not None:
            assert max_length >= x.shape[0], (
                f"Input length {x.shape[0]} exceeds max_length {max_length}"
            )
            pad_len = max_length - x.shape[0]
        elif multiple_of is not None:
            pad_len = (multiple_of - x.shape[0] % multiple_of) % multiple_of
        else:
            raise ValueError("Either max_length or multiple_of must be specified.")
        if pad_len > 0:
            pad = [0, 0] * (x.dim() - 1) + [0, pad_len]
            x = F.pad(x, pad)
        return x

    return {k: pad_fn(v) for k, v in data.items()}


@dataclasses.dataclass(slots=True, kw_only=True)
class DatasetConfig:
    """Configuration for the training dataset.

    Attributes
    ----------
    name: str
        Name of the dataset, e.g., "pdb", "afdb"
    data_dir: str
        Path to the preprocessed data directory.
    metadata_path: str | None
        Path to the custom metadata file in msgpack format.
    """

    name: str
    data_dir: str | None = None
    metadata_path: str | None = None


@dataclasses.dataclass(slots=True, kw_only=True)
class TrainingDatasetConfig(DatasetConfig):
    """Configuration for the training dataset.

    Attributes
    ----------
    is_distillation: bool
        Whether the dataset is for distillation,
        i.e., using predicted structures as labels.
    sampling_strategy: str
        Strategy for sampling training examples.
    """

    weight: float
    is_distillation: bool = True
    filters: list[dict] = dataclasses.field(default_factory=list)
    sampling_strategy: tuple[str, ...] = ("length",)


@dataclasses.dataclass(slots=True, kw_only=True)
class ValidationDatasetConfig(DatasetConfig):
    """Configuration for the validation dataset."""


class LMDBDataset(torch.utils.data.Dataset):
    def __init__(self, config: DatasetConfig):
        super().__init__()
        self.name: str = config.name
        self.lm_alphabet: Alphabet = Alphabet()

        # Set path
        if config.data_dir is None:
            raise ValueError("data_dir is not specified in the config.")
        data_dir = pathlib.Path(config.data_dir)
        self.lmdb_path: str = str(data_dir / "structure.lmdb")
        if config.metadata_path is not None:
            self.metadata_path = config.metadata_path
        else:
            self.metadata_path = str(data_dir / "manifest.msgpack")

        # Load metadatas
        with open(self.metadata_path, "rb") as f:
            self.metadatas: list[dict] = msgpack.unpackb(f.read(), raw=False)

    def __len__(self) -> int:
        return len(self.metadatas)

    @property
    def lmdb_env(self) -> lmdb.Environment:
        if not hasattr(self, "_lmdb_env"):
            self._lmdb_env = lmdb.open(str(self.lmdb_path), readonly=True, lock=False)
        return self._lmdb_env

    def fetch_protein(self, key: str) -> protein.Protein:
        with self.lmdb_env.begin() as txn:
            npz_bytes = txn.get(key.encode())
        if npz_bytes is None:
            raise KeyError(f"Key {key} not found in LMDB database.")
        with io.BytesIO(npz_bytes) as f:
            prot = DataPipeline.load(f)
        # Check if the residue indices are contiguous and start from 1.
        # For training data preparation, we already complete the missing residues
        # in PDB files to 'UNK' with 'NaN' coordinates.
        if prot.residue_index is not None:
            if not np.array_equal(prot.residue_index, np.arange(1, len(prot) + 1)):
                raise NotImplementedError(
                    "Structure with missing residues is not supported yet."
                )
        return prot


class TrainingDataset(LMDBDataset):
    def __init__(
        self,
        config: TrainingDatasetConfig,
        max_length: int = 256,
        max_seq_length: int = 384,
    ):
        super().__init__(config)
        self.config: TrainingDatasetConfig = config
        self.cropper = ProteinCropper(
            prob_spatial=0.6, prob_contiguous=0.2, prob_multi_contiguous=0.2
        )
        self.max_length: int = max_length  # Folding input length limit
        self.max_seq_length: int = max_seq_length  # LM input length limit

        self.logger = logging.getLogger(f"[Training Dataset:{self.name}]")

        self.sampling_strategy = config.sampling_strategy
        for strategy in config.sampling_strategy:
            if strategy not in ("length", "cluster", "plddt"):
                raise ValueError(f"Invalid sampling strategy: {strategy}")
        self.filters = config.filters
        for filter_info in config.filters:
            if filter_info["type"] not in ("resolution", "plddt"):
                raise ValueError(f"Invalid filter type: {filter_info['type']}")
        self.filter_entries()

    def filter_entries(self):
        """Filter the dataset entries based on the specified criteria in the config."""

        def filter_fn(m: dict) -> bool:
            for filter_info in self.filters:
                if filter_info["type"] == "resolution":
                    r = m["exp"]["resolution"]
                    # NOTE: use NMR structures (resolution=None) for training.
                    if r is not None and r > filter_info["threshold"]:
                        return False
                elif filter_info["type"] == "plddt":
                    plddt = m["pred"]["plddt"]
                    if plddt is None:
                        self.logger.warning(
                            f"PLDDT is missing for entry {m['id']}; treating as 0."
                        )
                        plddt = 0.0
                    if np.isnan(plddt) or plddt < filter_info["threshold"]:
                        return False
            return True

        self.metadatas = [m for m in self.metadatas if filter_fn(m)]

    def get_sampling_weights(self) -> np.ndarray:
        """Get sampling weights."""

        def get_weight(m: dict) -> float:
            w = 1.0
            for strategy in self.sampling_strategy:
                if strategy == "length":
                    length = m["num_residues"]
                    w *= min(max(length, 256), 512)
                elif strategy == "cluster":
                    try:
                        cluster_size = m["cluster_size"]
                    except KeyError as e:
                        print(self.name, m)
                        raise e
                    if cluster_size == 0:
                        self.logger.warning(f"Cluster size is 0 for entry {m['id']}.")
                        cluster_size = 1
                    w *= 1 / cluster_size
                elif strategy == "plddt":
                    plddt = m["pred"]["plddt"]
                    if plddt is None:
                        self.logger.warning(
                            f"PLDDT is missing for entry {m['id']}; treating as 0."
                        )
                        plddt = 0.0
                    w *= min(max(plddt - 30, 0), 40)

            return w

        weights = np.array([get_weight(m) for m in self.metadatas])
        weights /= weights.sum()
        return weights

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, dict[str, torch.Tensor]]:
        while True:
            feat = self.sample_item(index)
            if feat["label"]["resolved_mask"][:, 1].sum() < 4:
                # If there are less than 4 resolved residues with C-alpha coordinates,
                # resample.
                metadata_dict = self.metadatas[index]
                name = metadata_dict["id"]
                self.logger.warning(
                    f"Sample {name} ({index}) has less than 4 resolved residues; "
                    f"resampling."
                )
                index = np.random.choice(len(self))
            else:
                break
        return feat

    def sample_item(self, index: int) -> dict[str, dict[str, torch.Tensor]]:
        # Create a random number generator without a fixed seed.
        rng = np.random.default_rng()

        metadata_dict = self.metadatas[index]
        m = metadata.Metadata.from_dict(metadata_dict)
        prot = self.fetch_protein(m.id)

        feat = featurize.featurize(prot.sequence)
        label = self.prepare_labels(prot)
        loss_mask = self.prepare_loss_masks(m)

        fold_input = {k: v for k, v in feat.items() if not k.startswith("lm.")}
        lm_input = {k: v for k, v in feat.items() if k.startswith("lm.")}

        # Crop the folding input and label to the maximum length.
        crop_indices = self.cropper.crop(prot, self.max_length, rng)
        fold_input = {k: v[crop_indices] for k, v in fold_input.items()}
        label = {k: v[crop_indices] for k, v in label.items()}

        # Prepare the LM input with expanded crop indices and BOS/EOS tokens.
        lm_crop_indices = self._expand_crop_indices_for_lm(
            crop_indices, len(prot), self.max_seq_length
        )
        lm_input = {k: v[lm_crop_indices] for k, v in lm_input.items()}

        # Add lm to fold input mapping manually since the LM input are
        # cropped with expanded indices.
        is_in_crop = np.isin(lm_input["lm.pos_id"], fold_input["res_idx"])
        seq_tok_idx = np.where(is_in_crop)[0]
        fold_input["seq_tok_idx"] = seq_tok_idx

        # Convert to torch tensors.
        fold_input = {k: torch.from_numpy(v) for k, v in fold_input.items()}
        lm_input = {k: torch.from_numpy(v) for k, v in lm_input.items()}
        label = {k: torch.from_numpy(v) for k, v in label.items()}
        loss_mask = {k: torch.tensor(v) for k, v in loss_mask.items()}

        # Pad the input and label to the maximum length.
        fold_input = pad_input(fold_input, max_length=self.max_length)
        label = pad_input(label, max_length=self.max_length)
        lm_input = pad_input(lm_input, max_length=self.max_seq_length)
        return {
            "feat": {**fold_input, **lm_input},
            "label": label,
            "loss_mask": loss_mask,
        }

    def prepare_labels(self, prot: protein.Protein) -> dict[str, np.ndarray]:
        """Prepare the label tensors for training."""
        # Extract the coordinates and the mask for resolved residues.
        coords = prot.coordinates  # [L, 14, 3]
        resolved_mask = np.isfinite(coords).all(axis=-1)  # [L, 14]
        coords = np.nan_to_num(coords, nan=0.0)
        coords = do_centering_atom14(coords, resolved_mask, mask_to_zero=True)
        return {"coordinates": coords, "resolved_mask": resolved_mask}

    def prepare_loss_masks(self, m: metadata.Metadata) -> dict[str, bool]:
        """Prepare the confidence mask for training."""
        confidence_loss = False
        # Train confidence head only on high-quality X-ray crystal/Cyro-EM structures
        # NMR structures have resolution value of None or 0.0.
        if not self.config.is_distillation:
            assert m.exp is not None, "Experiment metadata is missing"
            resolution = m.exp.resolution
            if resolution is not None and 0.1 <= resolution <= 3.0:
                confidence_loss = True
        return {
            "confidence": confidence_loss,
        }

    @staticmethod
    def _expand_crop_indices_for_lm(
        crop_indices: np.ndarray,
        seqlen: int,
        max_seq_length: int = 384,
    ) -> np.ndarray:
        assert len(crop_indices) <= seqlen, "Crop indices cannot exceed sequence length"
        if seqlen <= max_seq_length - 2:
            # If the full sequence fits within the LM input limit, use the entire sequence
            return np.arange(seqlen + 2)

        # NOTE: We assume that there is no missing residue in the input sequence.
        # We already complete the missing residues to 'UNK' with 'NaN' coordinates.
        budget = max_seq_length - len(crop_indices)

        # Initialize a boolean mask for the LM input tokens
        seq_crop_mask = np.zeros(seqlen + 2, dtype=bool)
        shifted_crops = crop_indices + 1  # Shift by 1 to account for BOS token at index 0
        seq_crop_mask[shifted_crops] = True

        # Determine the segments of contiguous indices
        breaks = np.where(np.diff(shifted_crops) != 1)[0] + 1
        segments = np.split(shifted_crops, breaks)

        # Identify internal gaps between segments: [start_idx, end_idx]
        gaps = []
        for i in range(len(segments) - 1):
            gap_start = segments[i][-1] + 1
            gap_end = segments[i + 1][0] - 1
            if gap_start <= gap_end:
                gaps.append([gap_start, gap_end])

        # Phase 1: Try to completely fill internal gaps
        # Sort gaps by size (smallest first) to maximize the number of merged segments
        gaps.sort(key=lambda x: x[1] - x[0] + 1)

        remaining_gaps = []
        for gap_start, gap_end in gaps:
            gap_size = gap_end - gap_start + 1
            if budget >= gap_size:
                # Fully fill the gap
                seq_crop_mask[gap_start : gap_end + 1] = True
                budget -= gap_size
            else:
                remaining_gaps.append([gap_start, gap_end])

        # Restore original left-to-right order for the remaining gaps
        remaining_gaps.sort(key=lambda x: x[0])

        # Phase 2: Prioritize inner expansion
        # Expand inwards into the remaining gaps
        active_inner_edges = []
        for gap_start, gap_end in remaining_gaps:
            # Append left segment expanding right, and right segment expanding left
            active_inner_edges.append({"pos": gap_start, "dir": 1, "limit": gap_end})
            active_inner_edges.append({"pos": gap_end, "dir": -1, "limit": gap_start})

        while budget > 0 and active_inner_edges:
            for edge in list(active_inner_edges):
                if budget <= 0:
                    break

                curr_pos = edge["pos"]
                # Fill the position if not already filled
                if not seq_crop_mask[curr_pos]:
                    seq_crop_mask[curr_pos] = True
                    budget -= 1

                # Move cursor
                next_pos = curr_pos + edge["dir"]

                # Check limits and overlaps to remove inactive edges
                if (
                    (edge["dir"] == 1 and next_pos > edge["limit"])
                    or (edge["dir"] == -1 and next_pos < edge["limit"])
                    or seq_crop_mask[next_pos]
                ):
                    active_inner_edges.remove(edge)
                else:
                    edge["pos"] = next_pos

        # Phase 3: Expand outer edges (leftmost and rightmost) if budget still remains
        if budget > 0:
            left_cursor = np.where(seq_crop_mask)[0][0] - 1
            right_cursor = np.where(seq_crop_mask)[0][-1] + 1

            while budget > 0:
                expanded = False
                # Expand leftmost anchor to the left
                if left_cursor >= 0 and budget > 0:
                    if not seq_crop_mask[left_cursor]:
                        seq_crop_mask[left_cursor] = True
                        budget -= 1
                    left_cursor -= 1
                    expanded = True

                # Expand rightmost anchor to the right
                if right_cursor <= seqlen + 1 and budget > 0:
                    if not seq_crop_mask[right_cursor]:
                        seq_crop_mask[right_cursor] = True
                        budget -= 1
                    right_cursor += 1
                    expanded = True

                if not expanded:
                    break

        return np.where(seq_crop_mask)[0]


class MultiTrainingDataset(torch.utils.data.Dataset):
    """Training dataset with AF3-style sampling and cropping."""

    def __init__(
        self,
        configs: list[TrainingDatasetConfig],
        max_length: int = 256,
        max_seq_length: int = 384,
    ) -> None:
        """
        Parameters
        ----------
        configs : list[TrainingDatasetConfig]
            List of dataset configurations.
        max_length : int, optional
            Maximum sequence length for the folding model input (default: 256).
        max_seq_length : int, optional
            Maximum sequence length for the language model input (default: 384).
        """
        self.datasets: list[TrainingDataset] = [
            TrainingDataset(config, max_length, max_seq_length) for config in configs
        ]
        ds_weights: list[np.ndarray] = [ds.get_sampling_weights() for ds in self.datasets]
        self.weights: np.ndarray = np.concatenate(
            [w * cfg.weight for w, cfg in zip(ds_weights, configs, strict=True)]
        )
        self.cumulative_sizes: np.ndarray = np.cumsum([len(ds) for ds in self.datasets])

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int) -> dict[str, dict[str, torch.Tensor]]:
        """Get the folding input for the given index."""
        # Find the dataset index
        dataset_idx = np.searchsorted(self.cumulative_sizes, index, side="right")
        if dataset_idx == 0:
            sample_idx = index
        else:
            sample_idx = index - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


class ValidationDataset(LMDBDataset):
    def __init__(self, config: ValidationDatasetConfig):
        super().__init__(config)
        self.config = config
        self.name = config.name

        # Sort the metadatas by sequence length for efficient batching
        self.metadatas.sort(key=lambda m: m["num_residues"])

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, dict[str, torch.Tensor]]:
        metadata_dict = self.metadatas[index]
        m = metadata.Metadata.from_dict(metadata_dict)
        prot = self.fetch_protein(m.id)

        # NOTE: For validation, we directly use lm features from featurization.
        feat = featurize.featurize(prot.sequence)
        label = self.prepare_labels(prot)

        feat = {k: torch.from_numpy(v) for k, v in feat.items()}
        label = {k: torch.from_numpy(v) for k, v in label.items()}

        # Pad the input and label to multiple of 32.
        return {
            "feat": pad_input(feat, multiple_of=32),
            "label": pad_input(label, multiple_of=32),
        }

    def prepare_labels(self, prot: protein.Protein) -> dict[str, np.ndarray]:
        """Prepare the label tensors for training."""
        # Extract the coordinates and the mask for resolved residues.
        coords = prot.coordinates  # [L, 14, 3]
        resolved_mask = np.isfinite(coords).all(axis=-1)  # [L, 14]
        coords = np.nan_to_num(coords, nan=0.0)
        coords = do_centering_atom14(coords, resolved_mask, mask_to_zero=True)
        return {"coordinates": coords, "resolved_mask": resolved_mask}
