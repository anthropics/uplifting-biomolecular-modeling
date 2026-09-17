"""Target distribution and GPU workers for the existing inference pipelines."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Literal


def validate_args(args: argparse.Namespace) -> None:
    gpu_ids = getattr(args, "gpu_ids", None)
    if gpu_ids is None:
        return
    if args.device is not None:
        raise ValueError("--gpu-ids cannot be combined with --device.")
    if not gpu_ids or any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError("GPU IDs must be a non-empty list of non-negative indices.")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"GPU IDs must be unique: {gpu_ids}.")


def distribute_targets(inputs: Sequence, num_workers: int) -> list[list]:
    if num_workers <= 0:
        raise ValueError("num_workers must be positive.")
    assignments = [[] for _ in range(num_workers)]
    workloads = [0] * num_workers

    def length(item) -> int:
        return len(item.sequence) if hasattr(item, "sequence") else item.length

    for item in sorted(inputs, key=lambda item: (-length(item), item.name)):
        worker = min(range(num_workers), key=lambda index: (workloads[index], index))
        assignments[worker].append(item)
        workloads[worker] += max(length(item), 1) ** 2
    return assignments


def prepare_weights(args: argparse.Namespace, model_type: str) -> argparse.Namespace:
    from atlasfold import pretrained

    model_name = pretrained.get_model_name(
        "atlasfold-260703" if model_type == "monomer" else "atlasfold-m-260725"
    )
    worker_args = argparse.Namespace(**vars(args))
    for field in ("model_path", "lm_path"):
        path = getattr(worker_args, field)
        if path is not None and not Path(path).is_file():
            raise FileNotFoundError(f"Local weight file does not exist: {path}")
    if worker_args.model_path is None:
        worker_args.model_path = pretrained.download_model_weights(
            model_name, args.cache_dir
        )
    if worker_args.lm_path is None:
        worker_args.lm_path = pretrained.download_lm_weights(
            pretrained.MODEL_CONFIGS[model_name].lm_name, args.cache_dir
        )
    worker_args.gpu_ids = None
    return worker_args


def worker_entry(
    worker_index: int,
    args: argparse.Namespace,
    assignments: Sequence[Sequence],
    gpu_ids: Sequence[int],
    model_type: str,
) -> None:
    import torch

    from atlasfold.cli import monomer, multimer

    # Keep the inherited CUDA visibility mask and use its logical device indices.
    gpu_id = gpu_ids[worker_index]
    torch.cuda.set_device(gpu_id)
    worker_args = argparse.Namespace(**vars(args))
    worker_args.device = f"cuda:{gpu_id}"
    worker_args.gpu_ids = None
    logger = logging.getLogger(f"atlasfold.{model_type}")
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            f"%(asctime)s | gpu-{gpu_id} | %(name)s | %(message)s",
            datefmt="%y/%m/%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    try:
        pipeline = monomer if model_type == "monomer" else multimer
        pipeline.run(worker_args, inputs=assignments[worker_index])
    finally:
        logger.removeHandler(handler)
        handler.close()


def run(
    args: argparse.Namespace,
    inputs: Sequence,
    model_type: Literal["monomer", "multimer"],
    logger: logging.Logger,
) -> None:
    import torch
    import torch.multiprocessing as mp

    validate_args(args)
    gpu_ids = list(args.gpu_ids)
    device_count = torch.cuda.device_count()
    if not torch.cuda.is_available() or any(index >= device_count for index in gpu_ids):
        raise ValueError(
            f"GPU IDs {gpu_ids} are unavailable; visible device count is {device_count}."
        )
    if not inputs:
        return

    logger.info("Preparing AtlasFold and AtlasLM weights before starting GPU workers.")
    worker_args = prepare_weights(args, model_type)
    gpu_ids = gpu_ids[: len(inputs)]
    assignments = distribute_targets(inputs, len(gpu_ids))
    for gpu_id, targets in zip(gpu_ids, assignments, strict=True):
        logger.info("Assigned %d targets to GPU %d.", len(targets), gpu_id)
    mp.spawn(
        worker_entry,
        args=(worker_args, assignments, gpu_ids, model_type),
        nprocs=len(gpu_ids),
        join=True,
    )
