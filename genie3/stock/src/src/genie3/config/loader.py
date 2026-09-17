"""Experiment configuration loading for the new Genie3 CLI."""

from __future__ import annotations

from pathlib import Path
import math
from typing import Any

import yaml

from genie3.config.models import (
    EvaluationConfig,
    EvaluationFoldingConfig,
    EvaluationInverseFoldingConfig,
    EvaluationReductionConfig,
    ExperimentConfig,
    ExperimentRunConfig,
    GenerationConfig,
    PathsConfig,
    RoundConfig,
    RuntimeConfig,
)
from genie3.config.schema import (
    ConfigError,
    optional_mapping,
    require_keys,
    require_mapping,
    require_nonempty_string,
)


def resolve_config_path(config_path: str) -> Path:
    """Return an absolute config path without loading or validating it yet."""
    return Path(config_path).expanduser().resolve()


def load_experiment_config(config_path: str) -> ExperimentRunConfig:
    """Load and validate a unified experiment config YAML file."""
    resolved_config_path = resolve_config_path(config_path)
    with resolved_config_path.open() as file:
        loaded = yaml.safe_load(file)

    raw = require_mapping(loaded, "config")
    experiment = _parse_experiment(raw.get("experiment"))
    paths = _parse_paths(raw.get("paths"))
    generation = _parse_generation(raw.get("generation"), paths)
    evaluation = _parse_evaluation(raw.get("evaluation"), paths)
    runtime = _parse_runtime(raw.get("runtime"))
    rounds = _parse_rounds(raw.get("rounds"))

    return ExperimentRunConfig(
        config_path=resolved_config_path,
        raw=raw,
        experiment=experiment,
        paths=paths,
        generation=generation,
        evaluation=evaluation,
        runtime=runtime,
        rounds=rounds,
    )


def validate_config_for_command(config: ExperimentRunConfig, command: str) -> None:
    """Validate that a loaded config has the sections needed for a CLI command."""
    if command in {"run", "generate"} and config.generation is None:
        raise ConfigError("Missing required config section: `generation`.")
    if command in {"run", "evaluate"} and config.evaluation is None:
        raise ConfigError("Missing required config section: `evaluation`.")
    if command in {"run", "evaluate"}:
        if config.evaluation is None:
            raise ConfigError("Missing required config section: `evaluation`.")
        if not config.evaluation.version:
            raise ConfigError("Missing required config field: `evaluation.version`.")
        if command == "evaluate" and not config.rootdir():
            raise ConfigError(
                "Missing evaluation root directory. Set `paths.rootdir`."
            )


def to_generation_config(
    config: ExperimentRunConfig,
    shard_id: int = 0,
    num_shards: int = 1,
) -> dict[str, Any]:
    """Convert unified generation settings into the sampling config shape.

    Args:
        config: Validated experiment run config.
        shard_id: Zero-based index of this shard (0 <= shard_id < num_shards).
        num_shards: Total number of shards. 1 means no sharding (default).
    """
    validate_config_for_command(config, "generate")
    assert config.generation is not None

    generation_config = {
        "base": dict(config.generation.base),
        "dataset": dict(config.generation.dataset),
        "io": dict(config.generation.io),
        "inference": dict(config.generation.inference),
        "compile": config.generation.compile,
    }
    if num_shards > 1:
        beam_handled = _apply_shard_to_config(generation_config, shard_id, num_shards)
        if not beam_handled:
            _normalize_beam_generation_config(generation_config)
    else:
        _normalize_beam_generation_config(generation_config)
    return generation_config


def to_evaluation_kwargs(
    config: ExperimentRunConfig,
    *,
    input_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Convert unified evaluation settings into evaluation runner kwargs."""
    if input_dir is None:
        validate_config_for_command(config, "evaluate")
    elif config.evaluation is None:
        raise ConfigError("Missing required config section: `evaluation`.")
    elif not config.evaluation.version:
        raise ConfigError("Missing required config field: `evaluation.version`.")
    assert config.evaluation is not None

    rootdir = str(input_dir) if input_dir is not None else config.rootdir()
    if rootdir is None:
        raise ConfigError("Missing evaluation root directory. Set `paths.rootdir`.")

    gen_selections = None
    if config.generation is not None:
        gen_selections = config.generation.dataset.get("selections")

    kwargs = {
        "rootdir": rootdir,
        "version": config.evaluation.version,
        "verbose": config.evaluation.verbose,
        "n_device": config.runtime.num_devices,
        "datadir": config.paths.dataset,
        "calc_novelty": config.evaluation.reduction.calc_novelty,
        "inverse_fold_mode": config.evaluation.inverse_folding.mode,
        "inverse_fold_model_name": config.evaluation.inverse_folding.model_name,
        "inverse_fold_num_seq": config.evaluation.inverse_folding.num_seq,
        "fold_mode": config.evaluation.folding.mode,
        "fold_model_name": config.evaluation.folding.model_name,
        "fold_backend": config.evaluation.folding.backend,
        "fold_num_models": config.evaluation.folding.num_models,
        "fold_num_recycles": config.evaluation.folding.num_recycles,
        "skip_reduce": config.evaluation.reduction.skip_reduce,
        "selections": gen_selections,
    }
    kwargs.update(config.evaluation.inverse_folding.extra)
    kwargs.update(config.evaluation.folding.extra)
    kwargs.update(config.evaluation.reduction.extra)
    kwargs.update(config.evaluation.extra)
    return kwargs


def _parse_experiment(data: Any) -> ExperimentConfig:
    section = require_mapping(data, "experiment")
    name = require_nonempty_string(section.pop("name", None), "experiment.name")
    seed = section.pop("seed", None)
    return ExperimentConfig(name=name, seed=seed, extra=section)


def _parse_paths(data: Any) -> PathsConfig:
    section = optional_mapping(data, "paths")
    rootdir = section.pop("rootdir", None)
    outdir = section.pop("outdir", None)
    if outdir is not None and rootdir is not None and outdir != rootdir:
        raise ConfigError(
            "Conflicting output directories. Use `paths.rootdir` as the single "
            "source of truth."
        )
    return PathsConfig(
        rootdir=rootdir or outdir,
        dataset=section.pop("dataset", None),
        extra=section,
    )


def _parse_generation(data: Any, paths: PathsConfig) -> GenerationConfig | None:
    if data is None:
        return None

    section = require_mapping(data, "generation")
    base = optional_mapping(section.pop("base", None), "generation.base")
    dataset = require_mapping(section.pop("dataset", None), "generation.dataset")
    io = optional_mapping(section.pop("io", None), "generation.io")
    inference = optional_mapping(section.pop("inference", None), "generation.inference")

    sampler = section.pop("sampler", None)
    if sampler is not None:
        if "sampler" in inference:
            raise ConfigError(
                "Use either `generation.sampler` or `generation.inference.sampler`, "
                "not both."
            )
        inference["sampler"] = require_mapping(sampler, "generation.sampler")

    # Apply generation.base defaults (pretrained v1)
    base.setdefault("checkpoint", "pretrained/v1/checkpoints/step=600000.ckpt")
    base.setdefault("config", "pretrained/v1/config.yaml")

    require_keys(dataset, "generation.dataset", ("source",))
    require_keys(inference, "generation.inference", ("sampler",))

    # Default sampler name to ddim
    inference["sampler"].setdefault("name", "ddim")

    # Default batch_size
    dataset.setdefault("batch_size", 1)

    # Propagate paths.dataset → dataset.datadir; disallow explicit datadir
    if dataset.get("datadir") is not None:
        raise ConfigError(
            "`generation.dataset.datadir` is no longer supported. "
            "Use `paths.dataset` instead."
        )
    dataset["datadir"] = paths.dataset

    # Apply beam search defaults
    search_wrapper = inference.get("search")
    if isinstance(search_wrapper, dict) and search_wrapper.get("name") == "beam":
        search = search_wrapper.setdefault("search", {})
        search.setdefault("beam_width", 4)
        search.setdefault("score_interval", 25)
        search.setdefault("n_output", -1)
        search.setdefault("branch_noise_scale", 0.0)

    # Apply colabfold reward defaults and propagate paths.dataset → reward.datadir
    reward_wrapper = inference.get("reward")
    if isinstance(reward_wrapper, dict) and reward_wrapper.get("name") == "colabfold":
        reward = reward_wrapper.setdefault("reward", {})
        if reward.get("datadir") is not None:
            raise ConfigError(
                "`reward.reward.datadir` is no longer supported. "
                "Use `paths.dataset` instead."
            )
        reward.setdefault("version", "binder")
        reward.setdefault("mode", "template")
        reward.setdefault("num_models", 1)
        reward.setdefault("num_recycles", 3)
        reward.setdefault("use_proteinmpnn", True)
        reward["datadir"] = paths.dataset

    configured_outdir = io.get("outdir")
    path_outdir = paths.rootdir
    if configured_outdir is not None and path_outdir is not None and configured_outdir != path_outdir:
        raise ConfigError(
            "Conflicting generation output directories. Use `paths.rootdir` as the "
            "single source of truth instead of repeating `generation.io.outdir`."
        )

    if configured_outdir is None:
        if path_outdir is None:
            raise ConfigError(
                "Missing generation output directory. Set `paths.rootdir`."
            )
        io["outdir"] = path_outdir

    return GenerationConfig(
        base=base,
        dataset=dataset,
        io=io,
        inference=inference,
        compile=section.pop("compile", False),
        extra=section,
    )


def _parse_evaluation(data: Any, paths: PathsConfig) -> EvaluationConfig | None:
    if data is None:
        return None

    section = require_mapping(data, "evaluation")
    inverse_folding = optional_mapping(
        section.pop("inverse_folding", None), "evaluation.inverse_folding"
    )
    folding = optional_mapping(section.pop("folding", None), "evaluation.folding")
    reduction = optional_mapping(section.pop("reduction", None), "evaluation.reduction")

    return EvaluationConfig(
        version=section.pop("version", None),
        verbose=section.pop("verbose", False),
        inverse_folding=EvaluationInverseFoldingConfig(
            mode=inverse_folding.pop("mode", "all"),
            model_name=inverse_folding.pop("model_name", "proteinmpnn"),
            num_seq=inverse_folding.pop("num_seq", 8),
            extra=inverse_folding,
        ),
        folding=EvaluationFoldingConfig(
            mode=folding.pop("mode", "template"),
            model_name=folding.pop("model_name", "colabfold"),
            backend=folding.pop("backend", "subprocess"),
            num_models=folding.pop("num_models", 5),
            num_recycles=folding.pop("num_recycles", 20),
            extra=folding,
        ),
        reduction=EvaluationReductionConfig(
            calc_novelty=reduction.pop("calc_novelty", False),
            skip_reduce=reduction.pop("skip_reduce", False),
            extra=reduction,
        ),
        extra=section,
    )


def _normalize_beam_generation_config(config: dict[str, Any]) -> None:
    """Normalize user-facing beam config into runner-facing sampling config."""
    inference = config.get("inference")
    if not isinstance(inference, dict):
        return

    search_wrapper = inference.get("search")
    if not isinstance(search_wrapper, dict):
        return
    if search_wrapper.get("name") != "beam":
        return

    search = search_wrapper.get("search")
    dataset = config.get("dataset")
    if not isinstance(search, dict) or not isinstance(dataset, dict):
        return

    beam_width = int(search.get("beam_width", 1))
    n_output = int(search.get("n_output", -1))
    outputs_per_run = beam_width if n_output <= 0 else n_output
    requested_sample_count = int(dataset["n_sample"])
    run_count = math.ceil(requested_sample_count / outputs_per_run)

    dataset["n_sample"] = run_count
    search["requested_n_sample"] = requested_sample_count


def _apply_shard_to_config(
    config: dict[str, Any],
    shard_id: int,
    num_shards: int,
) -> bool:
    """Slice the dataset so this node only generates its assigned shard.

    Returns True if beam normalization was handled here (caller skips it).
    Returns False for non-beam (caller runs normalization, which is a no-op).
    """
    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        return False

    inference = config.get("inference")
    search_wrapper = inference.get("search") if isinstance(inference, dict) else None
    is_beam = (
        isinstance(search_wrapper, dict)
        and search_wrapper.get("name") == "beam"
    )

    original_n_sample = int(dataset["n_sample"])

    if is_beam:
        search = search_wrapper.get("search", {})
        beam_width = int(search.get("beam_width", 1))
        n_output = int(search.get("n_output", -1))
        outputs_per_run = beam_width if n_output <= 0 else n_output

        total_runs = math.ceil(original_n_sample / outputs_per_run)
        runs_per_shard = math.ceil(total_runs / num_shards)
        run_start = min(shard_id * runs_per_shard, total_runs)
        run_end = min(run_start + runs_per_shard, total_runs)

        dataset["n_sample"] = run_end - run_start
        dataset["sample_index_offset"] = run_start
        # Global total so beam.py:527 trims the last run of the last shard correctly.
        # sample_idx in beam search is globally indexed (e.g. 399), not shard-local.
        search["requested_n_sample"] = original_n_sample
        return True
    else:
        samples_per_shard = math.ceil(original_n_sample / num_shards)
        sample_start = min(shard_id * samples_per_shard, original_n_sample)
        sample_end = min(sample_start + samples_per_shard, original_n_sample)

        dataset["n_sample"] = sample_end - sample_start
        dataset["sample_index_offset"] = sample_start
        return False


def _parse_rounds(data: Any) -> list[RoundConfig]:
    if not data:
        return []
    if not isinstance(data, list):
        raise ConfigError("`rounds` must be a list of round configs.")
    rounds = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ConfigError(f"`rounds[{i}]` must be a mapping.")
        cond_strategy = item.get("cond_strategy")
        if not cond_strategy:
            raise ConfigError(f"`rounds[{i}].cond_strategy` is required.")
        rounds.append(RoundConfig(
            cond_strategy=cond_strategy,
            id=item.get("id"),
            n_sample=item.get("n_sample"),
        ))
    return rounds


def _parse_runtime(data: Any) -> RuntimeConfig:
    section = optional_mapping(data, "runtime")
    section.pop("num_nodes", None)  # no longer used; each run is always single-node
    return RuntimeConfig(
        num_devices=section.pop("num_devices", 1),
        device=section.pop("device", None),
        workers=section.pop("workers", None),
        extra=section,
    )
