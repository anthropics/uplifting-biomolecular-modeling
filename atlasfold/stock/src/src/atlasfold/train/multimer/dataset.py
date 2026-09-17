import dataclasses
import io
import logging
import pathlib
import pickle
from collections import defaultdict
from typing import Any

import lmdb
import msgpack
import numpy as np
import torch
import torch.nn.functional as F

from atlasfold.common import featurize, metadata, protein
from atlasfold.common import templates as template_utils
from atlasfold.train.monomer.dataset import DataPipeline as MonomerDataPipeline
from atlasfold.train.multimer import train_alignment
from atlasfold.train.multimer.cropper import MultimerCropper
from atlasfold.utils.geometry.random_augment import do_centering_atom14
from atlaslm.alphabet import Alphabet

TEMPLATE_AATYPE_MISMATCH_SKIP_FRACTION = 0.5


class MultimerDataPipeline:
    @staticmethod
    def save(compl: protein.ProteinMultimer, path: str | pathlib.Path):
        # Save the structure to compressed npz format
        data = {}
        data["name"] = np.array([compl.name], dtype="S")
        data["num_chains"] = np.array([compl.num_chains], dtype=np.int64)
        for i, c in enumerate(compl.chains):
            c_dict = MonomerDataPipeline._to_arr_dict(c)
            for k, v in c_dict.items():
                data[f"{i}.{k}"] = v
        np.savez_compressed(path, **data)

    @staticmethod
    def load(path: str | pathlib.Path | io.BytesIO) -> protein.ProteinMultimer:
        """Load the protein complex structure from compressed npz format."""
        with np.load(path) as data:
            name = data["name"].item().decode("utf-8")
            num_chains = data["num_chains"].item()
            chain_dicts = defaultdict(dict)
            for key in data.files:
                if key in ["name", "num_chains"]:
                    continue
                assert "." in key
                idx_str, field = key.split(".", 1)
                assert idx_str.isdigit(), f"Invalid key format: {key}"
                chain_dicts[int(idx_str)][field] = data[key]

            chains = [
                MonomerDataPipeline._from_arr_dict(chain_dicts[i])
                for i in range(num_chains)
            ]

        return protein.ProteinMultimer(name=name, chains=chains)


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
    type: str
        Dataset implementation type.
        One of "rcsb", "multimer_distillation", or "monomer_distillation".
    is_multimer: bool
        Whether the dataset contains multimeric structures.
    data_dir: str
        Path to the preprocessed data directory.
    metadata_path: str | None
        Path to the custom metadata file in msgpack format.
    use_templates: bool
        Whether to load per-chain template features.
    """

    name: str
    type: str
    data_dir: str | None = None
    metadata_path: str | None = None
    is_multimer: bool = True
    use_templates: bool = True
    template_lmdb_path: str | None = None
    template_mapping_path: str | None = None


@dataclasses.dataclass(slots=True, kw_only=True)
class TrainingDatasetConfig(DatasetConfig):
    """Configuration for the training dataset.

    Attributes
    ----------
    is_distillation: bool
        Whether the dataset is for distillation,
        i.e., using predicted structures as labels.
    """

    weight: float
    is_distillation: bool = True
    template_prob: float = 0.2
    filters: list[dict] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(slots=True, kw_only=True)
class ValidationDatasetConfig(DatasetConfig):
    """Configuration for the validation dataset."""


class LMDBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config: DatasetConfig,
        max_templates: int = 0,
    ):
        super().__init__()
        self.name: str = config.name
        self.type: str = config.type
        self.lm_alphabet: Alphabet = Alphabet()
        self.is_multimer: bool = config.is_multimer
        self.use_templates = config.use_templates

        # Set path
        if config.data_dir is None:
            raise ValueError("data_dir is not specified in the config.")
        data_dir = pathlib.Path(config.data_dir)
        self.data_dir = data_dir
        self.lmdb_path: str = str(data_dir / "structure.lmdb")
        if config.metadata_path is not None:
            self.metadata_path = config.metadata_path
        else:
            self.metadata_path = str(data_dir / "manifest.msgpack")

        # Load metadatas
        with open(self.metadata_path, "rb") as f:
            self.metadatas: list[dict] = msgpack.unpackb(f.read(), raw=False)
        self.logger = logging.getLogger(f"[Dataset:{self.name}]")

        # Optional AlphaFold-Multimer-style per-chain template resources.
        self.max_templates = int(max_templates)
        if self.use_templates:
            assert self.max_templates > 0, (
                "max_templates must be > 0 if use_templates is True."
            )
            template_lmdb_path = pathlib.Path(
                config.template_lmdb_path or data_dir / "template.lmdb"
            )
            template_mapping_path = pathlib.Path(
                config.template_mapping_path or data_dir / "template_mapping.lmdb"
            )
            assert template_lmdb_path.exists(), (
                f"Template LMDB path {template_lmdb_path} does not exist."
            )
            assert template_mapping_path.exists(), (
                f"Template mapping LMDB path {template_mapping_path} does not exist."
            )
            self.template_lmdb_path = str(template_lmdb_path)
            self.template_mapping_path = str(template_mapping_path)
            self.use_templates = True

    def __len__(self) -> int:
        return len(self.metadatas)

    @property
    def lmdb_env(self) -> lmdb.Environment:
        if not hasattr(self, "_lmdb_env"):
            self._lmdb_env = lmdb.open(str(self.lmdb_path), readonly=True, lock=False)
        return self._lmdb_env

    @property
    def template_lmdb_env(self) -> lmdb.Environment:
        if not self.use_templates:
            raise RuntimeError("Template LMDB is not enabled for this dataset.")
        if not hasattr(self, "_template_lmdb_env"):
            self._template_lmdb_env = lmdb.open(
                self.template_lmdb_path, readonly=True, lock=False
            )
        return self._template_lmdb_env

    @property
    def template_mapping_lmdb_env(self) -> lmdb.Environment:
        if not self.use_templates:
            raise RuntimeError("Template mapping LMDB is not enabled for this dataset.")
        if not hasattr(self, "_template_mapping_lmdb_env"):
            self._template_mapping_lmdb_env = lmdb.open(
                self.template_mapping_path,
                readonly=True,
                lock=False,
                readahead=False,
            )
        return self._template_mapping_lmdb_env

    def fetch_complex(self, key: str) -> protein.ProteinMultimer:
        with self.lmdb_env.begin() as txn:
            npz_bytes = txn.get(key.encode())
        if npz_bytes is None:
            raise KeyError(f"Key {key} not found in LMDB database.")
        if self.is_multimer:
            with io.BytesIO(npz_bytes) as f:
                compl = MultimerDataPipeline.load(f)
        else:
            with io.BytesIO(npz_bytes) as f:
                prot = MonomerDataPipeline.load(f)
            compl = protein.ProteinMultimer(name=prot.name, chains=[prot])
        return compl

    def fetch_template(self, key: str) -> protein.Protein:
        with self.template_lmdb_env.begin() as txn:
            npz_bytes = txn.get(key.encode())
        if npz_bytes is None:
            raise KeyError(f"Template key {key} not found in LMDB database.")
        with io.BytesIO(npz_bytes) as f:
            return MonomerDataPipeline.load(f)

    def fetch_template_hits(self, chain_id: str) -> list[dict]:
        with self.template_mapping_lmdb_env.begin() as txn:
            value = txn.get(chain_id.encode())
        if value is None:
            return []
        return pickle.loads(value)

    @staticmethod
    def get_template_mapping_key(
        complex_id: str,
        chain_metadata: metadata.Metadata,
    ) -> str | None:
        if chain_metadata.entity_id is None:
            return None
        return f"{complex_id.lower()}_{int(chain_metadata.entity_id)}"

    def prepare_template_inputs(
        self,
        compl: protein.ProteinMultimer,
        m: metadata.MultimerMetadata,
        chain_crops: list[np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Prepare AF-Multimer/AF3-style per-chain template features."""
        if not self.use_templates:
            return {}

        chain_template_features: list[dict[str, np.ndarray]] = []
        for chain, chain_metadata, crop in zip(
            compl.chains, m.chains, chain_crops, strict=True
        ):
            mapping_key = self.get_template_mapping_key(m.id, chain_metadata)
            chain_feat = self.prepare_chain_template_inputs(chain, mapping_key)
            chain_feat = template_utils.crop_template_features(chain_feat, crop)
            chain_template_features.append(chain_feat)
        return template_utils.concat_chain_template_features(chain_template_features)

    def prepare_empty_template_inputs(
        self,
        compl: protein.ProteinMultimer,
        chain_crops: list[np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Prepare fixed-size zero template features for a cropped complex."""
        chain_template_features = []
        for chain, crop in zip(compl.chains, chain_crops, strict=True):
            chain_feat = template_utils.make_empty_template_features(
                num_templates=self.max_templates,
                num_residues=len(chain),
            )
            chain_feat = template_utils.crop_template_features(chain_feat, crop)
            chain_template_features.append(chain_feat)
        return template_utils.concat_chain_template_features(chain_template_features)

    def prepare_chain_template_inputs(
        self,
        chain: protein.Protein,
        mapping_key: str | None,
    ) -> dict[str, np.ndarray]:
        """Prepare fixed-size template slots for one query chain."""
        hits = (
            []
            if mapping_key is None
            else self.fetch_template_hits(mapping_key)[: self.max_templates]
        )
        template_features: list[dict[str, np.ndarray]] = []
        for hit_dict in hits:
            hit = template_utils.TemplateHit.from_dict(hit_dict)
            try:
                template = self.fetch_template(hit.template_id)
            except KeyError:
                self.logger.warning(
                    f"Template {hit.template_id} for {mapping_key} not found; skipping."
                )
                continue
            if self.should_skip_template_hit(chain, template, hit, mapping_key):
                continue
            template_features.append(
                template_utils.featurize_aligned_template(
                    template,
                    hit,
                    query_length=len(chain),
                )
            )

        return template_utils.pack_template_features(
            template_features,
            num_templates=self.max_templates,
            query_length=len(chain),
        )

    def should_skip_template_hit(
        self,
        chain: protein.Protein,
        template: protein.Protein,
        hit: template_utils.TemplateHit,
        mapping_key: str | None,
    ) -> bool:
        mismatch_fraction, mismatch_count, aligned_count = (
            template_utils.template_sequence_mismatch_stats(chain.sequence, template, hit)
        )
        if (
            aligned_count > 0
            and mismatch_fraction >= TEMPLATE_AATYPE_MISMATCH_SKIP_FRACTION
        ):
            self.logger.debug(
                "Skipping template %s for %s: %.1f%% aligned residue types "
                "mismatch (%d/%d; threshold %.1f%%).",
                hit.template_id,
                mapping_key,
                100.0 * mismatch_fraction,
                mismatch_count,
                aligned_count,
                100.0 * TEMPLATE_AATYPE_MISMATCH_SKIP_FRACTION,
            )
            return True
        return False

    def to_metadata(self, m_dict: dict) -> metadata.MultimerMetadata:
        if self.is_multimer:
            return metadata.MultimerMetadata.from_dict(m_dict)
        else:
            _m = metadata.Metadata.from_dict(m_dict)
            return metadata.MultimerMetadata(
                _m.id,
                chains=[_m],
                interfaces=[],
                exp=_m.exp,
                pred=_m.pred,
            )


class TrainingDataset(LMDBDataset):
    def __init__(
        self,
        config: TrainingDatasetConfig,
        max_length: int = 384,
        max_seq_length: int = 768,
        max_templates: int = 2,
        max_contiguous_chains: int = 6,
    ):
        super().__init__(config, max_templates=max_templates)
        self.config: TrainingDatasetConfig = config
        if self.is_multimer:
            self.cropper = MultimerCropper(
                prob_spatial=0.4,
                prob_interface_spatial=0.4,
                prob_contiguous=0.2,
                max_contiguous_chains=max_contiguous_chains,
            )
        else:
            self.cropper = MultimerCropper(
                prob_spatial=0.75, prob_interface_spatial=0.0, prob_contiguous=0.25
            )
        self.max_length: int = max_length  # Folding input length limit
        self.max_seq_length: int = max_seq_length  # LM input length limit
        self.template_prob = float(config.template_prob)
        if not 0.0 <= self.template_prob <= 1.0:
            raise ValueError("template_prob must be between 0 and 1.")

        self.logger = logging.getLogger(f"[Training Dataset:{self.name}]")

    def get_sampling_weights(self) -> np.ndarray:
        """Get sampling weights for the dataset."""
        w = 1 / len(self.metadatas)
        return np.full(len(self.metadatas), w, dtype=np.float32)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        while True:
            try:
                feat = self.sample_item(index)
            except Exception as e:
                self.logger.warning(
                    f"Failed to sample item at index {index}: {e}. Resampling."
                )
                feat = None
            if feat is None:
                index = np.random.choice(len(self))
            else:
                break
        return feat

    def sample_item(self, index: int) -> dict[str, Any] | None:
        # Create a random number generator without a fixed seed.
        rng = np.random.default_rng()
        metadata_dict = self.metadatas[index]
        m = metadata.MultimerMetadata.from_dict(metadata_dict)
        compl = self.fetch_complex(m.id)
        return self.prepare_input(compl, m, bias_chain_id=None, rng=rng)

    def prepare_input(
        self,
        compl: protein.ProteinMultimer,
        m: metadata.MultimerMetadata,
        bias_chain_id: int | tuple[int, int] | None,
        rng: np.random.Generator,
    ) -> dict[str, Any] | None:
        # Crop the input complex
        chain_crops: list[np.ndarray] = self.cropper.crop(
            compl, m, self.max_length, bias_chain_id, rng
        )
        if sum(len(crop) for crop in chain_crops) < 4:
            self.logger.warning(
                f"Sample {m.id} has less than 4 residues after cropping; skipping."
            )
            return None

        # Extend the crop for LM input
        chain_lengths = [len(c) for c in compl.chains]
        chain_lm_crops: list[np.ndarray] = self.expand_crop_indices_for_lm(
            chain_crops, chain_lengths
        )

        fold_input, lm_input = self.featurize(compl, chain_crops, chain_lm_crops)
        template_input = self.prepare_template_inputs(compl, m, chain_crops, rng)
        label = self.prepare_labels(compl, chain_crops)
        full_label = self.prepare_full_labels(compl, chain_crops)
        loss_mask = self.prepare_loss_masks(m)

        # Pad the input and label to the maximum length.
        if template_input:
            template_input = template_utils.pad_template_features(
                template_input, max_length=self.max_length
            )

        # Convert to torch tensors.
        fold_input = {k: torch.from_numpy(v) for k, v in fold_input.items()}
        lm_input = {k: torch.from_numpy(v) for k, v in lm_input.items()}
        template_input = {k: torch.from_numpy(v) for k, v in template_input.items()}
        label = {k: torch.from_numpy(v) for k, v in label.items()}
        full_label = {k: torch.from_numpy(v) for k, v in full_label.items()}
        if loss_mask["confidence"]:
            full_label["alignment_metadata"] = train_alignment.prepare_alignment_metadata(
                full_label
            )
        loss_mask = {k: torch.tensor(v) for k, v in loss_mask.items()}

        fold_input = pad_input(fold_input, max_length=self.max_length)
        label = pad_input(label, max_length=self.max_length)
        lm_input = pad_input(lm_input, max_length=self.max_seq_length)

        if label["resolved_mask"][:, 1].sum() < 4:
            self.logger.warning(
                f"Sample {m.id} has less than 4 resolved residues; skipping."
            )
            return None

        return {
            "feat": {**fold_input, **lm_input, **template_input},
            "label": label,
            "full_label": full_label,
            "loss_mask": loss_mask,
        }

    def featurize(
        self,
        compl: protein.ProteinMultimer,
        chain_crops: list[np.ndarray],
        chain_lm_crops: list[np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        # Featurize each chain separately
        feats: list[dict[str, np.ndarray]] = [
            featurize.featurize(seq, entity_id=eid, asym_id=aid, sym_id=sid)
            for seq, eid, aid, sid in zip(
                compl.sequences,
                compl.entity_ids,
                compl.asym_ids,
                compl.sym_ids,
                strict=True,
            )
        ]
        # Separate fold input and LM input features, and crop them accordingly.
        chain_fold_input = [
            {k: v for k, v in f.items() if not k.startswith("lm.")} for f in feats
        ]
        chain_lm_input = [
            {k: v for k, v in f.items() if k.startswith("lm.")} for f in feats
        ]
        # Crop
        chain_fold_input = [
            {k: v[crop] for k, v in f.items()}
            for f, crop in zip(chain_fold_input, chain_crops, strict=True)
        ]
        chain_lm_input = [
            {k: v[crop] for k, v in f.items()}
            for f, crop in zip(chain_lm_input, chain_lm_crops, strict=True)
        ]
        lm_offset = 0
        for fold_feat, lm_feat in zip(chain_fold_input, chain_lm_input, strict=True):
            if len(fold_feat["res_idx"]) == 0:
                fold_feat["seq_tok_idx"] = np.empty((0,), dtype=np.int64)
            else:
                tok_idx_by_pos = {
                    int(pos_id): tok_idx
                    for tok_idx, pos_id in enumerate(lm_feat["lm.pos_id"])
                }
                try:
                    seq_tok_idx = np.asarray(
                        [tok_idx_by_pos[int(pos_id)] for pos_id in fold_feat["res_idx"]],
                        dtype=np.int64,
                    )
                except KeyError as e:
                    raise ValueError(
                        "LM crop does not cover all folded residues. "
                        f"Missing residue position: {e.args[0]}"
                    ) from e
                fold_feat["seq_tok_idx"] = seq_tok_idx + lm_offset
            lm_offset += len(lm_feat["lm.input_ids"])
        # Concatenate chains
        fold_input = {
            k: np.concatenate([f[k] for f in chain_fold_input], axis=0)
            for k in chain_fold_input[0].keys()
        }
        lm_input = {
            k: np.concatenate([f[k] for f in chain_lm_input], axis=0)
            for k in chain_lm_input[0].keys()
        }
        if len(fold_input["seq_tok_idx"]) > 0:
            max_seq_tok_idx = int(fold_input["seq_tok_idx"].max())
            if max_seq_tok_idx >= len(lm_input["lm.input_ids"]):
                raise ValueError(
                    f"seq_tok_idx contains {max_seq_tok_idx}, but LM input length is "
                    f"{len(lm_input['lm.input_ids'])}."
                )
        return fold_input, lm_input

    def prepare_labels(
        self, compl: protein.ProteinMultimer, chain_crops: list[np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Prepare the label tensors for training."""
        # Extract the coordinates and the mask for resolved residues.
        coords_list = []
        for chain, crop in zip(compl.chains, chain_crops, strict=True):
            if len(crop) == 0:
                continue
            coords = chain.coordinates[crop]  # [crop_len, 14, 3]
            coords_list.append(coords)
        coords = np.concatenate(coords_list, axis=0)  # [L, 14, 3]
        resolved_mask = np.isfinite(coords).all(axis=-1)  # [L, 14]
        coords = np.nan_to_num(coords, nan=0.0)
        coords = do_centering_atom14(coords, resolved_mask, mask_to_zero=True)
        return {"coordinates": coords, "resolved_mask": resolved_mask}

    def prepare_full_labels(
        self, compl: protein.ProteinMultimer, chain_crops: list[np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Prepare unpadded full-complex labels for permutation alignment."""
        coords = np.concatenate(
            [chain.coordinates for chain in compl.chains], axis=0
        ).astype(np.float32)
        resolved_mask = np.isfinite(coords).all(axis=-1)
        coords = np.nan_to_num(coords, nan=0.0)

        full_feat = featurize.featurize_complex(
            compl.sequences,
            compl.entity_ids,
            compl.asym_ids,
            compl.sym_ids,
        )

        crop_to_full_idx = []
        chain_start = 0
        for chain, crop in zip(compl.chains, chain_crops, strict=True):
            if len(crop) > 0:
                crop_to_full_idx.append(crop + chain_start)
            chain_start += chain.num_residues
        if crop_to_full_idx:
            crop_to_full_idx = np.concatenate(crop_to_full_idx, axis=0)
        else:
            crop_to_full_idx = np.empty((0,), dtype=np.int64)

        return {
            "coordinates": coords,
            "resolved_mask": resolved_mask,
            "aatype_int": full_feat["aatype_int"],
            "asym_id": full_feat["asym_id"],
            "entity_id": full_feat["entity_id"],
            "res_idx": full_feat["res_idx"],
            "crop_to_full_idx": crop_to_full_idx.astype(np.int64),
        }

    def prepare_template_inputs(
        self,
        compl: protein.ProteinMultimer,
        m: metadata.MultimerMetadata,
        chain_crops: list[np.ndarray],
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        """Prepare AF-Multimer/AF3-style per-chain template features.

        Template tensors are indexed by cropped residues, not LM tokens.
        """
        if not self.use_templates:
            return self.prepare_empty_template_inputs(compl, chain_crops)

        chain_template_features: list[dict[str, np.ndarray]] = []
        for chain, chain_metadata, crop in zip(
            compl.chains, m.chains, chain_crops, strict=True
        ):
            mapping_key = self.get_template_mapping_key(m.id, chain_metadata)
            chain_feat = self.prepare_chain_template_inputs(chain, mapping_key, rng)
            chain_feat = template_utils.crop_template_features(chain_feat, crop)
            chain_template_features.append(chain_feat)
        return template_utils.concat_chain_template_features(chain_template_features)

    def prepare_chain_template_inputs(
        self,
        chain: protein.Protein,
        mapping_key: str | None,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        """Prepare fixed-size template slots for one query chain."""
        hits = [] if mapping_key is None else self.fetch_template_hits(mapping_key)
        if hits and rng.random() < self.template_prob:
            max_selected = min(len(hits), self.max_templates)
            num_selected = int(rng.integers(1, max_selected + 1))
            selected_indices = np.sort(
                rng.choice(len(hits), size=num_selected, replace=False)
            )
            hits = [hits[i] for i in selected_indices]
        else:
            hits = []
        template_features: list[dict[str, np.ndarray]] = []
        for hit_dict in hits:
            hit = template_utils.TemplateHit.from_dict(hit_dict)
            try:
                template = self.fetch_template(hit.template_id)
            except KeyError:
                self.logger.warning(
                    f"Template {hit.template_id} for {mapping_key} not found; skipping."
                )
                continue
            if self.should_skip_template_hit(chain, template, hit, mapping_key):
                continue
            template_features.append(
                template_utils.featurize_aligned_template(
                    template,
                    hit,
                    query_length=len(chain),
                )
            )

        return template_utils.pack_template_features(
            template_features,
            num_templates=self.max_templates,
            query_length=len(chain),
        )

    def prepare_loss_masks(self, m: metadata.MultimerMetadata) -> dict[str, bool]:
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

    def expand_crop_indices_for_lm(
        self,
        chain_crops: list[np.ndarray],
        chain_lengths: list[int],
    ) -> list[np.ndarray]:
        """
        Distribute the global sequence length budget across multiple chains
        and expand their crop indices accordingly. Skipped chains (empty crops)
        are ignored in budget calculations.
        """
        num_chains = len(chain_crops)
        chain_crop_sizes = [len(crop) for crop in chain_crops]
        total_crop_size = sum(chain_crop_sizes)

        # Skip empty chains
        max_chain_lengths = [
            length + 2 if size > 0 else 0
            for length, size in zip(chain_lengths, chain_crop_sizes, strict=True)
        ]

        # If the total length of all chains fits within the budget, no crop.
        if sum(max_chain_lengths) <= self.max_seq_length:
            return [
                np.arange(length + 2) if size > 0 else np.array([], dtype=int)
                for length, size in zip(chain_lengths, chain_crop_sizes, strict=True)
            ]

        # Calculate the maximum number of additional tokens each chain can absorb
        absorbable = [
            max_len - crop_size
            for max_len, crop_size in zip(
                max_chain_lengths, chain_crop_sizes, strict=True
            )
        ]
        total_absorbable = sum(absorbable)
        remaining_budget = max(0, self.max_seq_length - total_crop_size)

        # Distribute proportionally for 80% of budget, and uniformly for the rest 20%.
        num_active_chains = sum(1 for size in chain_crop_sizes if size > 0)
        uniform_weight = 1.0 / num_active_chains
        allocated_budgets = [0 for _ in range(num_chains)]
        for i in range(num_chains):
            if chain_crop_sizes[i] == 0:
                continue
            proportional_weight = absorbable[i] / total_absorbable
            w = (0.2 * uniform_weight) + (0.8 * proportional_weight)
            allocation = int(remaining_budget * w)
            allocated_budgets[i] = min(allocation, absorbable[i])

        # Expand each chain using its newly allocated total length limit
        expanded_crops = []
        for i in range(num_chains):
            if chain_crop_sizes[i] == 0:
                expanded_crops.append(np.array([], dtype=int))
            else:
                target_length = chain_crop_sizes[i] + allocated_budgets[i]
                expanded = self._expand_crop_indices_for_lm(
                    chain_crops[i], chain_lengths[i], max_seq_length=target_length
                )
                expanded_crops.append(expanded)

        return expanded_crops

    @staticmethod
    def _expand_crop_indices_for_lm(
        crop_indices: np.ndarray,
        seqlen: int,
        max_seq_length: int = 768,
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


class RCSBTrainingDataset(TrainingDataset):
    """Training dataset for RCSB PDB"""

    def __init__(
        self,
        config: TrainingDatasetConfig,
        max_length: int = 384,
        max_seq_length: int = 768,
        max_templates: int = 2,
        max_contiguous_chains: int = 6,
    ):
        super().__init__(
            config, max_length, max_seq_length, max_templates, max_contiguous_chains
        )
        assert config.is_multimer, (
            "RCSBTrainingDataset should be used for multimer datasets."
        )

        # Chain/Interface clustering for sampling
        chain_weights = 1.0
        interface_weights = 2.0
        samples: list[tuple[dict, int | tuple[int, int]]] = []
        weights: list[float] = []
        for m_dict in self.metadatas:
            for c_i, cm in enumerate(m_dict["chains"]):
                samples.append((m_dict, c_i))
                weights.append(1.0 / (cm.get("cluster_size", 1) or 1) * chain_weights)
            for im in m_dict["interfaces"]:
                samples.append((m_dict, tuple(im["chain_ids"])))
                weights.append(1.0 / (im.get("cluster_size", 1) or 1) * interface_weights)
        self.samples = samples
        self.weights = np.array(weights, dtype=np.float32)

    def get_sampling_weights(self) -> np.ndarray:
        return self.weights / self.weights.sum()

    def __len__(self) -> int:
        return len(self.samples)

    def sample_item(self, index: int) -> dict[str, dict[str, torch.Tensor]] | None:
        rng = np.random.default_rng()
        m_dict, chain_id = self.samples[index]
        m = metadata.MultimerMetadata.from_dict(m_dict)
        compl = self.fetch_complex(m.id)
        return self.prepare_input(compl, m, chain_id, rng)


class MultimerDistillationDataset(TrainingDataset):
    """Training dataset for multimer distillation"""

    def __init__(
        self,
        config: TrainingDatasetConfig,
        max_length: int = 384,
        max_seq_length: int = 768,
        max_templates: int = 0,
        max_contiguous_chains: int = 6,
    ):
        super().__init__(
            config, max_length, max_seq_length, max_templates, max_contiguous_chains
        )
        assert config.is_multimer, (
            "MultimerDistillationDataset should be used for multimer datasets."
        )


class MonomerTrainingDataset(TrainingDataset):
    """Training dataset for monomer distillation"""

    def sample_item(self, index: int) -> dict[str, dict[str, torch.Tensor]] | None:
        rng = np.random.default_rng()
        m = self.to_metadata(self.metadatas[index])
        compl = self.fetch_complex(m.id)
        return self.prepare_input(compl, m, None, rng)


class MultiTrainingDataset(torch.utils.data.Dataset):
    """Training dataset with AF3-style sampling and cropping."""

    def __init__(
        self,
        configs: list[TrainingDatasetConfig],
        max_length: int = 384,
        max_seq_length: int = 768,
        max_templates: int = 2,
        max_contiguous_chains: int = 6,
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
        max_contiguous_chains : int, optional
            Maximum number of chains considered by contiguous cropping (default: 6).
        """
        self.datasets: list[TrainingDataset] = []
        for config in configs:
            if config.type == "rcsb":
                ds = RCSBTrainingDataset(
                    config,
                    max_length,
                    max_seq_length,
                    max_templates,
                    max_contiguous_chains,
                )
            elif config.type == "multimer_distillation":
                ds = MultimerDistillationDataset(
                    config,
                    max_length,
                    max_seq_length,
                    max_templates,
                    max_contiguous_chains,
                )
            elif config.type == "monomer_distillation":
                ds = MonomerTrainingDataset(config, max_length, max_seq_length)
            else:
                raise ValueError(
                    f"Unsupported dataset type '{config.type}' "
                    f"for dataset '{config.name}'."
                )
            self.datasets.append(ds)

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
    """RCSB PDB validation dataset for multimeric structures."""

    def __init__(
        self,
        config: ValidationDatasetConfig,
        max_templates: int = 0,
    ):
        super().__init__(config, max_templates=max_templates)
        self.config = config
        self.name = config.name
        # Sort the metadatas by sequence length for efficient batching
        self.metadatas.sort(
            key=lambda m: m.get(
                "num_residues",
                sum(chain["num_residues"] for chain in m["chains"]),
            )
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, dict[str, torch.Tensor]]:
        metadata_dict = self.metadatas[index]
        m = metadata.MultimerMetadata.from_dict(metadata_dict)
        compl = self.fetch_complex(m.id)

        # NOTE: For validation, we directly use lm features from featurization.
        feat = featurize.featurize_complex(
            compl.sequences, compl.entity_ids, compl.asym_ids, compl.sym_ids
        )
        label = self.prepare_labels(compl, m)
        template_input = {}

        padded_length = len(feat["aatype"]) + (-len(feat["aatype"]) % 32)
        if template_input:
            template_input = template_utils.pad_template_features(
                template_input, max_length=padded_length
            )

        # Pad the input and label to multiple of 32.
        feat = {
            **pad_input(
                {k: torch.from_numpy(v) for k, v in feat.items()}, multiple_of=32
            ),
            **{k: torch.from_numpy(v) for k, v in template_input.items()},
        }
        label = pad_input(
            {k: torch.from_numpy(v) for k, v in label.items()}, multiple_of=32
        )
        return {
            "feat": feat,
            "label": label,
        }

    def prepare_labels(
        self,
        compl: protein.ProteinMultimer,
        m: metadata.MultimerMetadata,
    ) -> dict[str, np.ndarray]:
        """Prepare the label tensors for training."""
        # Extract the coordinates and the mask for resolved residues.
        coords = np.concatenate(
            [c.coordinates for c in compl.chains], axis=0
        )  # [L, 14, 3]
        resolved_mask = np.isfinite(coords).all(axis=-1)  # [L, 14]
        coords = np.nan_to_num(coords, nan=0.0)
        coords = do_centering_atom14(coords, resolved_mask, mask_to_zero=True)

        if len(m.chains) != len(compl.chains):
            raise ValueError(
                f"Validation metadata/structure chain count mismatch for {m.id}: "
                f"{len(m.chains)} metadata chains != {len(compl.chains)} chains."
            )
        if len(compl.asym_ids) != len(compl.chains):
            raise ValueError(
                f"Validation structure asym_id count mismatch for {m.id}: "
                f"{len(compl.asym_ids)} asym_ids != {len(compl.chains)} chains."
            )

        low_homology_chain_asym_ids = [
            int(compl.asym_ids[i])
            for i, chain in enumerate(m.chains)
            if chain.is_low_homology
        ]
        low_homology_interface_asym_ids = []
        for interface in m.interfaces:
            if not interface.is_low_homology:
                continue
            chain_i, chain_j = interface.chain_ids
            if chain_i == chain_j:
                raise ValueError(
                    f"Validation metadata contains intra-chain interface for {m.id}: "
                    f"{interface.chain_ids}."
                )
            asym_i = compl.asym_ids[chain_i]
            asym_j = compl.asym_ids[chain_j]
            low_homology_interface_asym_ids.append((int(asym_i), int(asym_j)))

        return {
            "coordinates": coords,
            "resolved_mask": resolved_mask,
            "low_homology_chain_asym_ids": np.asarray(
                low_homology_chain_asym_ids,
                dtype=np.int64,
            ),
            "low_homology_interface_asym_ids": np.asarray(
                low_homology_interface_asym_ids,
                dtype=np.int64,
            ).reshape(-1, 2),
        }
