import dataclasses
import functools
import warnings
from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import NamedTuple, TypeAlias

import numpy as np
import torch

from atlasfold.common import featurize, protein, residue_constants
from atlasfold.model import AtlasFold_Multimer, SamplingConfig
from atlasfold.model.utils import confidence_metrics
from atlasfold.runner import SampleKey, autocast_context, seed_context
from atlasfold.utils import rasa


class MultimerInput(NamedTuple):
    """Input for one multimer folding target."""

    name: str
    chains: Sequence[str]

    @property
    def length(self) -> int:
        return sum(len(chain) for chain in self.chains)


MultimerInputLike: TypeAlias = MultimerInput | tuple[str, Sequence[str]]


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


@dataclasses.dataclass(kw_only=True)
class ProteinMultimerOutput(protein.ProteinMultimer):
    """A predicted protein complex with confidence scores."""

    name: str
    chains: list[protein.Protein]
    seed: int | None = None
    sample_index: int | None = None
    plddt: np.ndarray
    pae: np.ndarray
    pde: np.ndarray
    ptm: float
    iptm: float
    fraction_disordered: float
    chain_ptm: list[float] = dataclasses.field(default_factory=list)
    interface_iptm: dict[tuple[int, int], float] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        num_residues = self.num_residues
        if self.plddt.shape != (num_residues,):
            raise ValueError(
                f"Invalid pLDDT shape: {self.plddt.shape}. Expected ({num_residues},)."
            )
        if self.pae.shape != (num_residues, num_residues):
            raise ValueError(
                f"Invalid PAE shape: {self.pae.shape}. "
                f"Expected ({num_residues}, {num_residues})."
            )
        if self.pde.shape != (num_residues, num_residues):
            raise ValueError(
                f"Invalid PDE shape: {self.pde.shape}. "
                f"Expected ({num_residues}, {num_residues})."
            )
        if len(self.chain_ptm) != self.num_chains:
            raise ValueError(
                f"Invalid chain pTM length: {len(self.chain_ptm)}. "
                f"Expected {self.num_chains}."
            )
        missing_interface_iptm = {
            (chain_i, chain_j)
            for chain_i in range(self.num_chains)
            for chain_j in range(chain_i + 1, self.num_chains)
            if (chain_i, chain_j) not in self.interface_iptm
        }
        if missing_interface_iptm:
            raise ValueError(
                f"Missing interface ipTM for chain pairs: {missing_interface_iptm}."
            )

    @property
    def avg_plddt(self) -> float:
        return self.plddt.mean().item() * 100

    @property
    def avg_pde(self) -> float:
        return self.pde.mean().item()

    @property
    def ranking_score(self) -> float:
        return 0.8 * self.iptm + 0.2 * self.ptm + 0.5 * self.fraction_disordered

    @functools.cached_property
    def confidence_scores(self) -> dict:
        complex_scores = {
            "avg_plddt": self.avg_plddt,
            "avg_pde": self.avg_pde,
            "ptm": self.ptm,
            "iptm": self.iptm,
            "fraction_disordered": self.fraction_disordered,
            "ranking_score": self.ranking_score,
        }

        chain_scores: dict[str, dict] = {}
        chain_names = []
        start = 0
        for chain_idx, chain in enumerate(self.chains):
            end = start + chain.num_residues
            chain_mask = np.zeros(self.num_residues, dtype=bool)
            chain_mask[start:end] = True
            pair_mask = chain_mask[:, None] & chain_mask[None, :]
            chain_scores[chain.name] = {
                "avg_plddt": self.plddt[chain_mask].mean().item() * 100,
                "avg_pde": self.pde[pair_mask].mean().item(),
                "ptm": self.chain_ptm[chain_idx],
            }
            chain_names.append(chain.name)
            start = end

        interface_scores: dict[str, dict] = {}
        for chain_i, chain_name_i in enumerate(chain_names):
            for chain_j, chain_name_j in enumerate(
                chain_names[chain_i + 1 :], chain_i + 1
            ):
                interface_scores[f"{chain_name_i}-{chain_name_j}"] = {
                    "iptm": self.interface_iptm[(chain_i, chain_j)],
                }

        return {
            "complex": complex_scores,
            "chains": chain_scores,
            "interfaces": interface_scores,
        }


@dataclasses.dataclass
class MultimerFoldingOutput:
    """Outputs for one folded target."""

    outputs: dict[SampleKey, ProteinMultimerOutput]
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
    def best(self) -> ProteinMultimerOutput:
        return self.outputs[self.best_key]


class MultimerFoldingRunner:
    """Run AtlasFold-Multimer inference on pre-split chain sequences."""

    def __init__(self, model: AtlasFold_Multimer):
        self.model: AtlasFold_Multimer = model
        self.device = self.model.device

    def fold(
        self,
        name: str,
        chains: Sequence[str],
        *,
        num_samples: int = 5,
        seeds: int | Sequence[int] = 1,
        num_recycles: int = 10,
        sampling_config: SamplingConfig | None = None,
        mlm_prob: float = 0.20,
        length_buckets: Sequence[int] | None = None,
        return_distogram: bool = False,
    ) -> MultimerFoldingOutput:
        return next(
            self.fold_iter(
                [MultimerInput(name, chains)],
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
        inputs: Sequence[MultimerInputLike],
        *,
        num_samples: int = 5,
        seeds: int | Sequence[int] = 1,
        num_recycles: int = 10,
        sampling_config: SamplingConfig | None = None,
        mlm_prob: float = 0.20,
        length_buckets: Sequence[int] | None = None,
        max_tokens_per_batch: int = 1024,
        return_distogram: bool = False,
    ) -> Iterator[MultimerFoldingOutput]:
        """Yield predictions one target at a time in bucket execution order."""
        for batch in self.fold_iter_batch(
            inputs,
            num_samples=num_samples,
            seeds=seeds,
            num_recycles=num_recycles,
            sampling_config=sampling_config,
            mlm_prob=mlm_prob,
            length_buckets=length_buckets,
            max_tokens_per_batch=max_tokens_per_batch,
            return_distogram=return_distogram,
        ):
            yield from batch

    def fold_iter_batch(
        self,
        inputs: Sequence[MultimerInputLike],
        *,
        num_samples: int = 5,
        seeds: int | Sequence[int] = 1,
        num_recycles: int = 10,
        sampling_config: SamplingConfig | None = None,
        mlm_prob: float = 0.20,
        length_buckets: Sequence[int] | None = None,
        max_tokens_per_batch: int = 1024,
        return_distogram: bool = False,
    ) -> Iterator[list[MultimerFoldingOutput]]:
        seeds = [seeds] if isinstance(seeds, int) else list(seeds)

        if len(inputs) == 0:
            raise ValueError("No inputs provided.")
        if len(seeds) == 0:
            raise ValueError("No seeds provided.")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        if max_tokens_per_batch <= 0:
            raise ValueError(
                f"max_tokens_per_batch must be positive, got {max_tokens_per_batch}."
            )
        if sampling_config is None:
            sampling_config = SamplingConfig(num_steps=200)

        normalized_inputs = self._normalize_inputs(inputs)
        bucketed_inputs = self._bucket_inputs(normalized_inputs, length_buckets)
        for bucket_length, complexes in self._iter_batch(
            bucketed_inputs, max_tokens_per_batch
        ):
            batch_size = len(complexes)
            batch = self._make_batch_features(complexes, bucket_length)

            model_outputs: list[dict[SampleKey, ProteinMultimerOutput]] = [
                {} for _ in range(batch_size)
            ]
            batch_distogram_logits: list[dict[int, np.ndarray]] = [
                {} for _ in range(batch_size)
            ]
            distogram_boundaries = None
            for seed_value in seeds:
                out = self.model_run(
                    batch,
                    seed=seed_value,
                    num_samples=num_samples,
                    num_recycles=num_recycles,
                    mlm_prob=mlm_prob,
                    sampling_config=sampling_config,
                    return_distogram=return_distogram,
                )

                if return_distogram:
                    distogram_boundaries = out["distogram.boundaries"]
                for batch_i, complex_input in enumerate(complexes):
                    if return_distogram:
                        length = complex_input.num_residues
                        batch_distogram_logits[batch_i][seed_value] = out[
                            "distogram.logits"
                        ][batch_i, :length, :length]
                    model_outputs[batch_i].update(
                        self._make_outputs(
                            complex_input=complex_input,
                            out=out,
                            batch_idx=batch_i,
                            num_samples=num_samples,
                            seed=seed_value,
                        )
                    )
            batch_outputs = []
            for batch_i, outputs in enumerate(model_outputs):
                ranking = sorted(
                    outputs.keys(), key=lambda k: outputs[k].ranking_score, reverse=True
                )
                batch_outputs.append(
                    MultimerFoldingOutput(
                        outputs,
                        ranking,
                        distogram_logits=batch_distogram_logits[batch_i],
                        distogram_boundaries=distogram_boundaries,
                    )
                )
            yield batch_outputs

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
        tensor_feat: dict[str, torch.Tensor] = {
            k: torch.as_tensor(v, device=device) for k, v in feat.items()
        }
        with (
            torch.inference_mode(),
            seed_context(seed, device),
            autocast_context(device),
        ):
            out = self.model.inference(
                tensor_feat,
                num_samples=num_samples,
                num_recycles=num_recycles,
                mlm_prob=mlm_prob,
                sampling_config=sampling_config,
            )
        output_keys = {
            "sample_coords",
            "plddt",
            "pae",
            "pde",
            "ptm",
            "iptm",
            "chain_ptm",
            "interface_iptm",
        }
        if return_distogram:
            output_keys.update({"distogram.logits", "distogram.boundaries"})
        return {
            key: value.cpu().float().numpy()
            for key, value in out.items()
            if key in output_keys
        }

    @staticmethod
    def _normalize_inputs(
        inputs: Sequence[MultimerInputLike],
    ) -> list[MultimerInput]:
        normalized: list[MultimerInput] = []
        for input_ in inputs:
            item = MultimerInput(*input_)
            chains = [
                _sanitize_sequence(f"{item.name} chain {chain_index}", chain)
                for chain_index, chain in enumerate(item.chains, start=1)
            ]
            if any(len(chain) == 0 for chain in chains):
                raise ValueError(f"Input ({item.name}) has an empty chain.")
            normalized.append(MultimerInput(item.name, chains))
        return normalized

    @classmethod
    def _bucket_inputs(
        cls,
        inputs: Sequence[MultimerInput],
        length_buckets: Sequence[int] | None,
    ) -> dict[int, list[MultimerInput]]:
        bucketed_inputs: dict[int, list[MultimerInput]] = defaultdict(list)
        for item in inputs:
            length = sum(len(chain) for chain in item.chains)
            if length == 0:
                raise ValueError(f"Input {item.name} has no residues.")
            bucket_length = cls._get_length_bucket(length, length_buckets)
            bucketed_inputs[bucket_length].append(item)
        return bucketed_inputs

    def _iter_batch(
        self,
        bucketed_inputs: dict[int, list[MultimerInput]],
        max_tokens_per_batch: int,
    ) -> Iterator[tuple[int, list[protein.ProteinMultimer]]]:
        for bucket_length in sorted(bucketed_inputs):
            bucket_items = bucketed_inputs[bucket_length]
            batch_size = max(1, max_tokens_per_batch // bucket_length)
            for start in range(0, len(bucket_items), batch_size):
                chunk = bucket_items[start : start + batch_size]
                complexes = [
                    protein.ProteinMultimer.get_empty(item.name, list(item.chains))
                    for item in chunk
                ]
                yield bucket_length, complexes

    @staticmethod
    def _get_length_bucket(
        length: int,
        length_buckets: Sequence[int] | None = None,
    ) -> int:
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
            f"Complex length {length} exceeds the largest configured length bucket "
            f"({buckets[-1]})."
        )

    @classmethod
    def _make_batch_features(
        cls,
        complexes: Sequence[protein.ProteinMultimer],
        bucket_length: int,
    ) -> dict[str, np.ndarray]:
        feats = [
            featurize.featurize_complex(
                complex_input.sequences,
                complex_input.entity_ids,
                complex_input.asym_ids,
                complex_input.sym_ids,
            )
            for complex_input in complexes
        ]
        # Keep MLM random draws independent of batch composition (up to 16 chains).
        lm_length = bucket_length + 32
        feats = [
            cls._pad_to_lengths(feat, residue_length=bucket_length, lm_length=lm_length)
            for feat in feats
        ]
        return {k: np.stack([feat[k] for feat in feats], axis=0) for k in feats[0]}

    @staticmethod
    def _make_outputs(
        *,
        complex_input: protein.ProteinMultimer,
        out: dict[str, np.ndarray],
        batch_idx: int,
        num_samples: int,
        seed: int,
    ) -> dict[SampleKey, ProteinMultimerOutput]:
        chain_lengths = [len(chain.sequence) for chain in complex_input.chains]
        total_length = sum(chain_lengths)
        chain_ids = [chain.name for chain in complex_input.chains]
        num_chains = len(chain_lengths)
        samples = {}
        residue_rasa = rasa.compute_rasa(
            out["sample_coords"][batch_idx, :, :total_length],
            complex_input.sequences,
        )
        fraction_disordered = confidence_metrics.compute_fraction_disordered(
            residue_rasa, chain_lengths
        )

        for sample_idx in range(num_samples):
            coords = out["sample_coords"][batch_idx, sample_idx, :total_length]
            plddt = out["plddt"][batch_idx, sample_idx, :total_length]
            pae = out["pae"][batch_idx, sample_idx, :total_length, :total_length]
            pde = out["pde"][batch_idx, sample_idx, :total_length, :total_length]
            ptm = float(out["ptm"][batch_idx, sample_idx].item())
            iptm = float(out["iptm"][batch_idx, sample_idx].item())

            # Create Protein objects for each chain in the complex
            chains: list[protein.Protein] = []
            start = 0
            for chain_id, chain_sequence, chain_length in zip(
                chain_ids,
                complex_input.sequences,
                chain_lengths,
                strict=True,
            ):
                end = start + chain_length
                chains.append(
                    protein.Protein.create(
                        name=chain_id,
                        sequence=chain_sequence,
                        coordinates=coords[start:end],
                        b_factors=plddt[start:end] * 100,
                    )
                )
                start = end

            chain_ptm = [
                float(value)
                for value in out["chain_ptm"][batch_idx, sample_idx, :num_chains]
            ]
            interface_iptm = out["interface_iptm"][
                batch_idx, sample_idx, :num_chains, :num_chains
            ]
            interface_iptm_dict = {
                (chain_i, chain_j): float(interface_iptm[chain_i, chain_j])
                for chain_i in range(num_chains)
                for chain_j in range(chain_i + 1, num_chains)
            }

            samples[(seed, sample_idx)] = ProteinMultimerOutput(
                name=complex_input.name,
                chains=chains,
                seed=seed,
                sample_index=sample_idx,
                plddt=plddt,
                pae=pae,
                pde=pde,
                ptm=ptm,
                iptm=iptm,
                fraction_disordered=float(fraction_disordered[sample_idx]),
                chain_ptm=chain_ptm,
                interface_iptm=interface_iptm_dict,
            )
        return samples

    @staticmethod
    def _pad_to_lengths(
        feat: dict[str, np.ndarray],
        *,
        residue_length: int,
        lm_length: int,
    ) -> dict[str, np.ndarray]:
        current_length = feat["aatype_int"].shape[0]
        if residue_length < current_length:
            raise ValueError(
                f"Cannot pad features of length {current_length} to shorter "
                f"length {residue_length}."
            )

        new_feat: dict[str, np.ndarray] = {}
        for k, v in feat.items():
            target_length = lm_length if k.startswith("lm.") else residue_length
            pad_len = target_length - v.shape[0]
            if pad_len < 0:
                raise ValueError(
                    f"Feature {k} has length {v.shape[0]}, which exceeds "
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
