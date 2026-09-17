"""Typed configuration models for unified Genie3 experiment configs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


MappingData = dict[str, Any]


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment identity and reproducibility settings."""

    name: str
    seed: int | None = None
    extra: MappingData = field(default_factory=dict)


@dataclass(frozen=True)
class PathsConfig:
    """Shared input/output paths for generation and evaluation."""

    rootdir: str | None = None
    dataset: str | None = None
    extra: MappingData = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationConfig:
    """Generation settings in the unified experiment config."""

    base: MappingData
    dataset: MappingData
    io: MappingData
    inference: MappingData
    compile: bool = False
    extra: MappingData = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationInverseFoldingConfig:
    """Inverse-folding stage settings for evaluation."""

    mode: str = "all"
    model_name: str = "proteinmpnn"
    num_seq: int = 8
    extra: MappingData = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationFoldingConfig:
    """Structure-folding stage settings for evaluation."""

    mode: str = "template"
    model_name: str = "colabfold"
    backend: str = "subprocess"
    num_models: int = 5
    num_recycles: int = 20
    extra: MappingData = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationReductionConfig:
    """Reduction-stage settings for evaluation."""

    calc_novelty: bool = False
    skip_reduce: bool = False
    extra: MappingData = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation settings in the unified experiment config."""

    version: str | None = None
    verbose: bool = False
    inverse_folding: EvaluationInverseFoldingConfig = field(
        default_factory=EvaluationInverseFoldingConfig
    )
    folding: EvaluationFoldingConfig = field(default_factory=EvaluationFoldingConfig)
    reduction: EvaluationReductionConfig = field(default_factory=EvaluationReductionConfig)
    extra: MappingData = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeConfig:
    """Invocation-independent runtime settings."""

    num_devices: int = 1
    device: str | None = None
    workers: int | None = None
    extra: MappingData = field(default_factory=dict)


@dataclass(frozen=True)
class RoundConfig:
    """Configuration for a single round in an iterative binder design experiment."""

    cond_strategy: str
    id: str | None = None
    n_sample: int | None = None


@dataclass(frozen=True)
class ExperimentRunConfig:
    """Unified config loaded from one experiment YAML file."""

    config_path: Path
    raw: MappingData
    experiment: ExperimentConfig
    paths: PathsConfig
    generation: GenerationConfig | None = None
    evaluation: EvaluationConfig | None = None
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    rounds: list[RoundConfig] = field(default_factory=list)

    def rootdir(self) -> str | None:
        """Return the experiment root directory, preferring generation.io."""
        if self.generation is not None:
            outdir = self.generation.io.get("outdir")
            if outdir is not None:
                return str(outdir)
        return self.paths.rootdir
