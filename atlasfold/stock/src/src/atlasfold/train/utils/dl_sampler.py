import math
import warnings
from collections.abc import Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler

__all__ = ["DistributedWeightedSampler"]


class DistributedWeightedSampler(Sampler[int]):
    def __init__(
        self,
        weights: np.ndarray | torch.Tensor | list[float],
        num_samples: int | None = None,
        replacement: bool = True,
        rank: int | None = None,
        world_size: int | None = None,
        epoch: int = 0,
        seed: int = 0,
    ) -> None:
        is_dist_initialized = dist.is_available() and dist.is_initialized()

        if rank is None:
            if not is_dist_initialized:
                warnings.warn(
                    "Distributed package is not available. Defaulting rank to 0."
                )
                rank = 0
            else:
                rank = dist.get_rank()

        if world_size is None:
            if not is_dist_initialized:
                warnings.warn(
                    "Distributed package is not available. Defaulting world_size to 1."
                )
                world_size = 1
            else:
                world_size = dist.get_world_size()

        self.rank: int = rank
        self.world_size: int = world_size
        self.epoch: int = epoch
        self.seed: int = seed
        self.replacement: bool = replacement

        if isinstance(weights, torch.Tensor):
            weights = weights.cpu().numpy()
        elif isinstance(weights, list):
            weights = np.array(weights)
        if weights.sum() == 0:
            raise ValueError("Sum of weights cannot be zero.")

        # Normalize weights
        weights = weights.astype(np.float64)
        self.weights: np.ndarray = weights / weights.sum()
        self.dataset_size: int = self.weights.shape[0]

        # Determine number of samples
        if num_samples is None:
            # If num_samples is not provided, set it so that each replica gets an equal
            # share
            total_global_samples = self.dataset_size
        else:
            total_global_samples = num_samples

        self.num_samples: int = int(math.ceil(total_global_samples / self.world_size))
        self.total_size: int = self.num_samples * self.world_size

    def __iter__(self) -> Iterator[int]:
        # deterministically shuffle based on epoch and seed
        seed = self.seed + self.epoch
        rng = np.random.default_rng(seed)

        indices = rng.choice(
            self.dataset_size,
            self.total_size,
            p=self.weights,
            replace=self.replacement,
        ).tolist()
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.world_size]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for this sampler.

        Changing the epoch ensures all replicas use a different random ordering for each
        epoch, as the random seed is determined by the combination of the `seed` and
        `epoch` attributes. If the epoch is not changed, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
