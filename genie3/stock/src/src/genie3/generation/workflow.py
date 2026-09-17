"""CLI-facing generation workflows for Genie3."""

from __future__ import annotations

import os
import sys
import datetime
import logging
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from genie3.config import (
    to_generation_config,
    validate_config_for_command,
)
from genie3.generation.stats import GenerationSummary, summarize_generation
from genie3.runtime.context import RunContext
from genie3.runtime.progress import get_active_reporter
from genie3.runtime.shards import (
    is_generation_shard_done,
    parse_selections,
    write_generation_shard_marker,
)


@dataclass(frozen=True)
class GenerationResult:
    """Minimal generation result passed from `genie3 run` into evaluation."""

    output_dir: Path | None = None
    summary: GenerationSummary | None = None


def run_generation(
    *,
    config_path: str,
    verbose: bool = False,
    log_dir: str | None = None,
    context: RunContext | None = None,
    shard_id: int = 0,
    num_shards: int = 1,
    num_devices: int | None = None,
) -> GenerationResult:
    """Run generation through the new CLI path.

    Args:
        config_path: Path to the experiment config YAML.
        verbose: Show detailed runtime output.
        log_dir: Root directory for run logs.
        context: Active RunContext if called from within a managed run.
        shard_id: Zero-based shard index for multi-node generation. Default 0.
        num_shards: Total number of shards. Default 1 (no sharding).
        num_devices: Override runtime.num_devices from config. Default None (use config).
    """
    del log_dir
    if context is not None:
        context.logger.debug("Preparing generation config")
        run_config = context.config
    else:
        from genie3.config import load_experiment_config

        run_config = load_experiment_config(config_path)

    validate_config_for_command(run_config, "generate")
    generation_config = to_generation_config(run_config, shard_id=shard_id, num_shards=num_shards)
    sample_config = _build_sample_config_from_dict(generation_config)

    selections_str = generation_config.get("dataset", {}).get("selections")
    selected_problems = parse_selections(selections_str)

    effective_num_devices = num_devices if num_devices is not None else run_config.runtime.num_devices
    args = Namespace(
        sample_config=sample_config,
        num_devices=effective_num_devices,
        num_nodes=1,
        verbose=context.invocation.verbose if context is not None else verbose,
        seed=run_config.experiment.seed,
    )

    outdir = Path(sample_config.io.outdir)

    if sample_config.dataset.n_sample == 0:
        logging.warning(
            "Shard %d/%d has no work (num_shards > total beam runs). "
            "Skipping generation.",
            shard_id,
            num_shards,
        )
        if num_shards > 1:
            write_generation_shard_marker(outdir, shard_id, num_shards, selected_problems=selected_problems)
        return GenerationResult(output_dir=outdir)

    if num_shards > 1 and is_generation_shard_done(outdir, shard_id, num_shards, selected_problems=selected_problems):
        logging.info(
            "Shard %d/%d already complete (marker found). Skipping generation.",
            shard_id,
            num_shards,
        )
        return GenerationResult(
            output_dir=outdir,
            summary=summarize_generation(sample_config.io.outdir),
        )

    if context is not None:
        context.logger.debug("Generation rootdir: %s", sample_config.io.outdir)
        context.logger.debug("Generation runtime args: %s", args)
        get_active_reporter().set_status("initialization")
        with context.profile.stage("generate"):
            _run_sample_main(args)
        _synchronize_generation_workers()
        get_active_reporter().set_status("done")
    else:
        _run_sample_main(args)

    if num_shards > 1:
        write_generation_shard_marker(outdir, shard_id, num_shards, selected_problems=selected_problems)

    return GenerationResult(
        output_dir=outdir,
        summary=summarize_generation(sample_config.io.outdir),
    )


def run_training(
    *,
    config_path: str,
    devices: int,
    num_nodes: int = 1,
    test: bool = False,
    mpi_plugin: bool = False,
    memory_snapshot: bool = False,
    reset_dataloader_state: bool = False,
) -> None:
    """Run training through the new CLI path."""
    args = Namespace(
        config=config_path,
        devices=devices,
        num_nodes=num_nodes,
        test=test,
        mpi_plugin=mpi_plugin,
        memory_snapshot=memory_snapshot,
        reset_dataloader_state=reset_dataloader_state,
    )
    _run_train_main(args)


def _build_sample_config_from_dict(config: dict):
    from genie3.generation.config.registry import build_sample_config_from_dict

    return build_sample_config_from_dict(config)


def _run_sample_main(args: Namespace) -> None:
    from lightning import Trainer, seed_everything
    from lightning.pytorch.strategies import DDPStrategy

    from genie3.generation.data.data_module import GenieDataModule
    from genie3.generation.runner.runner import GenieRunner

    worker_log_path = _configure_generation_logging(args.verbose)
    capture_stream = None
    original_stdout = None
    original_stderr = None
    if worker_log_path is not None and not args.verbose:
        capture_stream = open(worker_log_path, "a")
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = capture_stream
        sys.stderr = capture_stream

    try:
        config = args.sample_config

        if args.seed is not None:
            seed_everything(args.seed, workers=True)

        trainer = Trainer(
            devices=args.num_devices,
            num_nodes=args.num_nodes,
            accelerator="gpu",
            strategy=DDPStrategy(timeout=datetime.timedelta(hours=8)),
            deterministic=True,
            enable_progress_bar=args.verbose,
            inference_mode=False,
        )

        get_active_reporter().set_status("generation")
        runner = GenieRunner(config)
        dm = GenieDataModule(config.dataset)
        trainer.test(runner, dm)
        _synchronize_generation_workers()

        sampler_cfg = config.inference.sampler.sampler
        if not getattr(sampler_cfg, "predict_sidechain", False):
            get_active_reporter().set_status("done")
            logging.info("Sampling done!")
            return

        logging.info("[1 / 2] Main stage completed!")

        assert config.dataset.source == "unconditional"
        assert config.inference.sampler.name == "ddim"
        assert config.inference.sampler.sampler.predict_sequence

        config.dataset.source = "sidechain"
        config.dataset.datadir = config.io.outdir

        get_active_reporter().set_status("sidechain")
        runner = GenieRunner(config)
        dm = GenieDataModule(config.dataset)
        trainer.test(runner, dm)
        _synchronize_generation_workers()

        logging.info("[2 / 2] Sidechain stage completed!")
        get_active_reporter().set_status("done")
        logging.info("Sampling done!")
    finally:
        if capture_stream is not None:
            capture_stream.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            capture_stream.close()


def _run_train_main(args: Namespace) -> None:
    import torch
    from lightning import Trainer, seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.plugins.environments import MPIEnvironment

    from genie3.generation.config.registry import get_config
    from genie3.generation.data.data_module import GenieDataModule
    from genie3.generation.runner.runner import GenieRunner
    from genie3.generation.utils.callbacks import SaveOnTermination
    from genie3.generation.utils.model_io import get_ckpt_path

    config = get_config(args.config)

    seed_everything(config.training.seed, workers=True)

    logger = (
        WandbLogger(project=config.io.project, name=config.name, id=config.name)
        if not args.test
        else None
    )

    checkpoint_dirpath = os.path.join(
        config.io.rootdir, config.io.project, config.name, "checkpoints"
    )
    callbacks = [
        ModelCheckpoint(
            filename="{step}",
            save_top_k=-1,
            every_n_train_steps=config.training.checkpoint_every_n_step,
            dirpath=checkpoint_dirpath,
        )
    ]
    if not args.test:
        callbacks.append(SaveOnTermination(dirpath=checkpoint_dirpath))

    dm = GenieDataModule(config.data, reset_state_dict=args.reset_dataloader_state)
    runner = GenieRunner(config)

    trainer = Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        deterministic=True,
        enable_progress_bar=args.test,
        log_every_n_steps=config.training.log_every_n_step,
        max_steps=config.training.max_steps,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        gradient_clip_val=config.training.gradient_clip_val,
        gradient_clip_algorithm=config.training.gradient_clip_algorithm,
        callbacks=callbacks,
        plugins=MPIEnvironment() if args.mpi_plugin else None,
    )

    if args.memory_snapshot:
        torch.cuda.memory._record_memory_history(max_entries=100000)

    trainer.fit(runner, dm, ckpt_path=get_ckpt_path(checkpoint_dirpath))

    if args.memory_snapshot:
        torch.cuda.memory._dump_snapshot(
            os.path.join(
                config.io.rootdir, config.io.project, config.name, "memory_trace.pkl"
            )
        )


def _configure_generation_logging(verbose: bool):
    """Route generation runtime output into per-rank worker logs when available."""
    workers_dir = os.environ.get("GENIE3_WORKERS_DIR")
    local_rank = os.environ.get("LOCAL_RANK")
    rank = os.environ.get("RANK")
    worker_suffix = local_rank or rank or "launcher"
    primary_worker = worker_suffix in {"0", "launcher"}
    worker_log_path = None
    handlers = []

    if workers_dir:
        os.makedirs(workers_dir, exist_ok=True)
        worker_log_path = os.path.join(
            workers_dir,
            f"generate_device_{worker_suffix}.log",
        )
        handlers.append(logging.FileHandler(worker_log_path))

    if not handlers or (verbose and primary_worker):
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )
    return worker_log_path


def _synchronize_generation_workers() -> None:
    """Wait for all ranks before reading shared generation artifacts."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass
