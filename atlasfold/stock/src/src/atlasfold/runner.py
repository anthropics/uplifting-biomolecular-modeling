import contextlib
import dataclasses
import warnings
from collections.abc import Iterator, Sequence
from typing import NamedTuple, TypeAlias

import numpy as np
import torch

from atlasfold.common import featurize, protein, residue_constants
from atlasfold.model import AtlasFold, SamplingConfig

SampleKey = tuple[int, int]


class FoldingInput(NamedTuple):
    """Input for one monomer folding target."""

    name: str
    sequence: str


FoldingInputLike: TypeAlias = FoldingInput | tuple[str, str]


def _sanitize_sequence(name: str, sequence: str) -> str:
    sequence = "".join(sequence.split()).upper()
    nonstandard_residues = sorted(
        set(sequence).difference(residue_constants.restype_orders)
    )
    if nonstandard_residues:
        warnings.warn(
            f"{name}: Nonstandard residue(s) "
            f"{', '.join(nonstandard_residues)} replaced with 'X'.",
            stacklevel=2,
        )
    return "".join(
        aa if aa in residue_constants.restype_orders else "X" for aa in sequence
    )


@contextlib.contextmanager
def seed_context(seed: int, device: torch.device):
    if device.type == "cuda":
        with torch.random.fork_rng(devices=[device], device_type="cuda"):
            torch.manual_seed(seed)
            yield
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            yield


@contextlib.contextmanager
def autocast_context(device: torch.device):
    if device.type == "cuda":
        with torch.autocast(device.type, torch.bfloat16, enabled=True):
            yield
    else:
        yield


def get_sampling_config(length: int) -> SamplingConfig:
    """Return the default sampling configuration for AtlasFold."""
    base_cfg = SamplingConfig()
    if length <= 512:
        base_cfg.num_steps = 20
    elif length <= 1024:
        base_cfg.num_steps = 30
    else:
        base_cfg.num_steps = 100
    return base_cfg


@dataclasses.dataclass(kw_only=True)
class ProteinOutput(protein.Protein):
    """A data structure representing a predicted 3D protein structure"""

    name: str
    sequence: str
    seed: int | None = None
    sample_index: int | None = None
    coordinates: np.ndarray  # [L, 14, 3]
    b_factors: np.ndarray  # [L,] or [L, 14]
    plddt: np.ndarray  # [L]
    pae: np.ndarray  # [L, L]
    ptm: float
    residue_index: np.ndarray | None = None  # [L], optional residue index

    def __post_init__(self):
        """Validate the input data."""
        super().__post_init__()
        L = len(self.sequence)
        if self.plddt.shape != (L,):
            raise ValueError(
                f"Invalid pLDDT shape: {self.plddt.shape}. "
                f"Expected (L,) where L is the sequence length."
            )
        if self.pae.shape != (L, L):
            raise ValueError(
                f"Invalid PAE shape: {self.pae.shape}. "
                f"Expected (L, L) where L is the sequence length."
            )
        assert self.residue_index is None

    @property
    def avg_plddt(self) -> float:
        return float(self.plddt.mean() * 100)

    @property
    def ranking_score(self) -> float:
        return self.avg_plddt

    @property
    def confidence_scores(self) -> dict[str, float]:
        return {
            "avg_plddt": self.avg_plddt,
            "ptm": float(self.ptm),
        }


@dataclasses.dataclass()
class FoldingOutput:
    """Outputs for one folded target."""

    outputs: dict[SampleKey, ProteinOutput]
    ranking: list[SampleKey]
    distogram_logits: dict[int, np.ndarray] = dataclasses.field(default_factory=dict)
    distogram_boundaries: np.ndarray | None = None

    @property
    def name(self) -> str:
        return self.outputs[next(iter(self.outputs))].name

    @property
    def length(self) -> int:
        return self.outputs[next(iter(self.outputs))].num_residues

    @property
    def best_key(self) -> SampleKey:
        if len(self.ranking) == 0:
            raise ValueError("No ranked samples are available.")
        return self.ranking[0]

    @property
    def best(self) -> ProteinOutput:
        return self.outputs[self.best_key]


class FoldingRunner:
    """A class for running protein folding using the AtlasFold model."""

    def __init__(self, model: AtlasFold):
        self.model: AtlasFold = model
        self.device = self.model.device

    def fold(
        self,
        name: str,
        sequence: str,
        *,
        num_samples: int = 5,
        seeds: int | Sequence[int] = 1,
        num_recycles: int = 4,
        mlm_prob: float = 0.15,
        sampling_config: SamplingConfig | None = None,
        length_buckets: Sequence[int] | None = None,
        return_distogram: bool = False,
    ) -> FoldingOutput:
        return next(
            self.fold_iter(
                [FoldingInput(name, sequence)],
                num_samples=num_samples,
                seeds=seeds,
                num_recycles=num_recycles,
                mlm_prob=mlm_prob,
                sampling_config=sampling_config,
                length_buckets=length_buckets,
                return_distogram=return_distogram,
            )
        )

    def fold_iter(
        self,
        inputs: Sequence[FoldingInputLike],
        *,
        num_samples: int = 5,
        seeds: int | Sequence[int] = 1,
        num_recycles: int = 4,
        mlm_prob: float = 0.15,
        sampling_config: SamplingConfig | None = None,
        length_buckets: Sequence[int] | None = None,
        max_tokens_per_batch: int = 1024,
        return_distogram: bool = False,
    ) -> Iterator[FoldingOutput]:
        """Yield predictions one target at a time in bucket execution order."""
        for batch in self.fold_iter_batch(
            inputs,
            num_samples=num_samples,
            seeds=seeds,
            num_recycles=num_recycles,
            mlm_prob=mlm_prob,
            sampling_config=sampling_config,
            length_buckets=length_buckets,
            max_tokens_per_batch=max_tokens_per_batch,
            return_distogram=return_distogram,
        ):
            yield from batch

    def fold_iter_batch(
        self,
        inputs: Sequence[FoldingInputLike],
        *,
        num_samples: int = 5,
        seeds: int | Sequence[int] = 1,
        num_recycles: int = 4,
        mlm_prob: float = 0.15,
        sampling_config: SamplingConfig | None = None,
        length_buckets: Sequence[int] | None = None,
        max_tokens_per_batch: int = 1024,
        return_distogram: bool = False,
    ) -> Iterator[list[FoldingOutput]]:
        """Yield predictions grouped by model batch."""
        seeds = [seeds] if isinstance(seeds, int) else list(seeds)
        inputs = list(inputs)
        if len(inputs) == 0:
            raise ValueError("No inputs provided.")
        if len(seeds) == 0:
            raise ValueError("No seeds provided.")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        if not 0.0 < mlm_prob <= 1.0:
            raise ValueError(
                f"mlm_prob must be greater than 0 and at most 1, got {mlm_prob}."
            )
        if max_tokens_per_batch <= 0:
            raise ValueError(
                f"max_tokens_per_batch must be positive, got {max_tokens_per_batch}."
            )

        normalized_inputs = self._normalize_inputs(inputs)
        bucketed_inputs = self._bucket_inputs(normalized_inputs, length_buckets)
        for bucket_length, chunk in self._iter_batch(
            bucketed_inputs, max_tokens_per_batch
        ):
            sequences = [item.sequence for item in chunk]
            feat = self._make_batch_features(sequences, bucket_length)

            _config = sampling_config or get_sampling_config(bucket_length)

            batch_outputs: list[dict[SampleKey, ProteinOutput]] = [{} for _ in chunk]
            batch_distogram_logits: list[dict[int, np.ndarray]] = [{} for _ in chunk]
            distogram_boundaries = None
            for seed_value in seeds:
                out = self.model_run(
                    feat,
                    seed=seed_value,
                    num_samples=num_samples,
                    num_recycles=num_recycles,
                    mlm_prob=mlm_prob,
                    sampling_config=_config,
                    return_distogram=return_distogram,
                )

                if return_distogram:
                    distogram_boundaries = out["distogram.boundaries"]
                for batch_item_idx, item in enumerate(chunk):
                    if return_distogram:
                        length = len(item.sequence)
                        batch_distogram_logits[batch_item_idx][seed_value] = out[
                            "distogram.logits"
                        ][batch_item_idx, :length, :length]
                    batch_outputs[batch_item_idx].update(
                        self._make_outputs(
                            name=item.name,
                            sequence=item.sequence,
                            out=out,
                            batch_idx=batch_item_idx,
                            num_samples=num_samples,
                            seed=seed_value,
                        )
                    )

            model_outputs = []
            for batch_item_idx, outputs in enumerate(batch_outputs):
                ranking = sorted(
                    outputs.keys(), key=lambda k: outputs[k].ranking_score, reverse=True
                )
                model_outputs.append(
                    FoldingOutput(
                        outputs,
                        ranking,
                        distogram_logits=batch_distogram_logits[batch_item_idx],
                        distogram_boundaries=distogram_boundaries,
                    )
                )
            yield model_outputs

    def model_run(
        self,
        feat: dict[str, np.ndarray],
        seed: int,
        num_samples: int,
        num_recycles: int,
        mlm_prob: float,
        sampling_config: SamplingConfig,
        return_distogram: bool = False,
    ) -> dict[str, np.ndarray]:
        device = self.device
        feat: dict[str, torch.Tensor] = {
            k: torch.as_tensor(v, device=device) for k, v in feat.items()
        }
        with (
            torch.inference_mode(),
            seed_context(seed, device),
            autocast_context(device),
        ):
            out = self.model.inference(
                feat,
                num_samples=num_samples,
                num_recycles=num_recycles,
                mlm_prob=mlm_prob,
                sampling_config=sampling_config,
            )
        output_keys = {"sample_coords", "plddt", "pae", "ptm"}
        if return_distogram:
            output_keys.update({"distogram.logits", "distogram.boundaries"})
        return {
            key: value.cpu().float().numpy()
            for key, value in out.items()
            if key in output_keys
        }

    @staticmethod
    def _normalize_inputs(
        inputs: Sequence[FoldingInputLike],
    ) -> list[FoldingInput]:
        normalized: list[FoldingInput] = []
        for input_ in inputs:
            item = FoldingInput(*input_)
            sequence = _sanitize_sequence(item.name, item.sequence)
            if len(sequence) == 0:
                raise ValueError(f"Input ({item.name}) has an empty sequence.")
            normalized.append(FoldingInput(item.name, sequence))
        return normalized

    @classmethod
    def _bucket_inputs(
        cls,
        inputs: Sequence[FoldingInput],
        length_buckets: Sequence[int] | None,
    ) -> dict[int, list[FoldingInput]]:
        bucketed_inputs: dict[int, list[FoldingInput]] = {}
        for item in inputs:
            bucket_length = cls._get_length_bucket(len(item.sequence), length_buckets)
            bucketed_inputs.setdefault(bucket_length, []).append(item)
        return bucketed_inputs

    @staticmethod
    def _iter_batch(
        bucketed_inputs: dict[int, list[FoldingInput]],
        max_tokens_per_batch: int,
    ) -> Iterator[tuple[int, list[FoldingInput]]]:
        for bucket_length in sorted(bucketed_inputs):
            bucket_items = bucketed_inputs[bucket_length]
            batch_size = max(1, max_tokens_per_batch // bucket_length)
            for start in range(0, len(bucket_items), batch_size):
                yield bucket_length, bucket_items[start : start + batch_size]

    @staticmethod
    def _get_length_bucket(
        length: int,
        length_buckets: Sequence[int] | None = None,
    ) -> int:
        """Return the smallest residue budget that can contain ``length``."""
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}.")

        if length_buckets is None:
            for bucket in featurize.DEFAULT_BUCKETS:
                if length <= bucket:
                    return bucket
            return ((length + 127) // 128) * 128

        buckets = sorted(set(length_buckets))
        if len(buckets) == 0:
            raise ValueError("length_buckets must contain at least one bucket.")
        if buckets[0] <= 0:
            raise ValueError("length_buckets must contain only positive integers.")

        for bucket in buckets:
            if length <= bucket:
                return bucket

        raise ValueError(
            f"Sequence length {length} exceeds the largest configured length bucket "
            f"({buckets[-1]})."
        )

    @classmethod
    def _make_batch_features(
        cls,
        sequences: Sequence[str],
        bucket_length: int,
    ) -> dict[str, np.ndarray]:
        feats = [
            cls._pad_to_length(featurize.featurize(sequence), bucket_length)
            for sequence in sequences
        ]
        return {k: np.stack([feat[k] for feat in feats], axis=0) for k in feats[0]}

    @staticmethod
    def _make_outputs(
        *,
        name: str,
        sequence: str,
        out: dict[str, np.ndarray],
        batch_idx: int,
        num_samples: int,
        seed: int,
    ) -> dict[SampleKey, ProteinOutput]:
        length = len(sequence)
        samples = {}
        for sample_idx in range(num_samples):
            coords = out["sample_coords"][batch_idx, sample_idx, :length]
            plddt = out["plddt"][batch_idx, sample_idx, :length]
            b_factor = plddt * 100
            pae = out["pae"][batch_idx, sample_idx, :length, :length]
            ptm = float(out["ptm"][batch_idx, sample_idx].item())

            sample = ProteinOutput(
                name=name,
                sequence=sequence,
                seed=seed,
                sample_index=sample_idx,
                coordinates=coords,
                b_factors=b_factor,
                plddt=plddt,
                pae=pae,
                ptm=ptm,
            )
            samples[(seed, sample_idx)] = sample
        return samples

    @staticmethod
    def _pad_to_length(
        feat: dict[str, np.ndarray],
        length: int,
    ) -> dict[str, np.ndarray]:
        """Pad the input features to an exact residue length."""
        current_length = feat["aatype_int"].shape[0]
        if length < current_length:
            raise ValueError(
                f"Cannot pad features of length {current_length} to shorter "
                f"length {length}."
            )

        new_feat: dict[str, np.ndarray] = {}
        for k, v in feat.items():
            target_length = length + 2 if k.startswith("lm.") else length
            pad_len = target_length - v.shape[0]
            if pad_len < 0:
                raise ValueError(
                    f"Feature {k!r} has length {v.shape[0]}, which exceeds "
                    f"target length {target_length}."
                )
            if pad_len == 0:
                new_feat[k] = v
            else:
                pad_width = ((0, pad_len),) + ((0, 0),) * (v.ndim - 1)
                constant_values = featurize.PAD_IDX if k == "lm.input_ids" else 0
                new_feat[k] = np.pad(
                    v,
                    pad_width,
                    constant_values=constant_values,
                )
        return new_feat

    @staticmethod
    def _pad(feat: dict[str, np.ndarray], multiple_of: int = 32) -> dict[str, np.ndarray]:
        """Pad the input features to the specified length."""
        length = feat["aatype_int"].shape[0]
        pad_length = ((length + multiple_of - 1) // multiple_of) * multiple_of
        return FoldingRunner._pad_to_length(feat, pad_length)
