import os
from typing import Any

import torch
import torch.distributed as dist


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size(group: Any = None) -> int:
    if os.environ.get("RANK", -1) == -1 or not is_dist_initialized():
        return 1
    return dist.get_world_size(group=group)


def get_rank(group: Any = None) -> int:
    if os.environ.get("RANK", -1) == -1 or not is_dist_initialized():
        return 0
    return dist.get_rank(group=group)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0)) if is_dist_initialized() else 0


def setup_dist() -> None:
    rank = int(os.environ.get("RANK", -1))
    if dist.is_available() and torch.cuda.is_available() and rank != -1:
        torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


def destroy_process_group() -> None:
    if is_dist_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if is_dist_initialized():
        dist.barrier()
