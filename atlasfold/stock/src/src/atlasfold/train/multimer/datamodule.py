import dataclasses
import gc
import logging

import lightning.pytorch as pl
from omegaconf import OmegaConf
from torch.utils.data import default_collate
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from atlasfold.train.multimer.dataset import (
    MultiTrainingDataset,
    TrainingDatasetConfig,
    ValidationDataset,
    ValidationDatasetConfig,
)
from atlasfold.train.utils.dl_sampler import DistributedWeightedSampler


def multimer_train_collate(samples: list[dict]) -> dict:
    """Collate train samples while preserving variable-length full labels."""
    return {
        "feat": default_collate([sample["feat"] for sample in samples]),
        "label": default_collate([sample["label"] for sample in samples]),
        "loss_mask": default_collate([sample["loss_mask"] for sample in samples]),
        "full_label": [sample["full_label"] for sample in samples],
    }


@dataclasses.dataclass(kw_only=True)
class DataModuleConfig:
    # === Common config for data modules === #
    batch_size: int = 1
    global_batch_size: int = 128
    num_workers: int = 0
    persistent_workers: bool = True
    pin_memory: bool = True

    # === Training hyperparameters === #
    max_length: int = 256
    max_seq_length: int = 384
    max_templates: int = 2
    max_contiguous_chains: int = 6

    # === Dataset configs === #
    data_root: str
    train_datasets: list[TrainingDatasetConfig] = dataclasses.field(default_factory=list)
    val_dataset: ValidationDatasetConfig


class TrainingDataModule(pl.LightningDataModule):
    _train_ds: MultiTrainingDataset
    _val_ds: ValidationDataset

    def __init__(self, config: DataModuleConfig) -> None:
        super().__init__()
        config = OmegaConf.to_object(OmegaConf.merge(DataModuleConfig, config))

        self.config = config
        self.logger = logging.getLogger("[DataModule]")

        # Set up the data directory based on root dir
        data_root = self.config.data_root
        for ds_cfg in self.config.train_datasets:
            if ds_cfg.data_dir is None:
                ds_cfg.data_dir = f"{data_root}/{ds_cfg.name}"
        val_ds_cfg = self.config.val_dataset
        if val_ds_cfg.data_dir is None:
            val_ds_cfg.data_dir = f"{data_root}/{val_ds_cfg.name}"

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit":
            self._train_ds = self.construct_train_dataset()
            self._val_ds = self.construct_val_dataset()
        elif stage == "validate":
            self._val_ds = self.construct_val_dataset()
        else:
            raise NotImplementedError("Not implemented yet.")
        gc.collect()
        gc.freeze()

    def construct_train_dataset(self) -> MultiTrainingDataset:
        """Construct training dataset."""
        if hasattr(self, "_train_ds"):
            return self._train_ds
        multi_ds = MultiTrainingDataset(
            configs=self.config.train_datasets,
            max_length=self.config.max_length,
            max_seq_length=self.config.max_seq_length,
            max_templates=self.config.max_templates,
            max_contiguous_chains=self.config.max_contiguous_chains,
        )
        # Print dataset info
        for d in multi_ds.datasets:
            self.print_rank_zero(
                f"Constructed training dataset '{d.name}':\n"
                f"  Weights: {d.config.weight}\n"
                f"  Num complexes: {len(d.metadatas)}\n"
                f"  Num samples: {len(d)}"
            )
        return multi_ds

    def construct_val_dataset(self) -> ValidationDataset:
        """Construct validation dataset."""
        if hasattr(self, "_val_ds"):
            return self._val_ds

        ds = ValidationDataset(
            config=self.config.val_dataset,
            max_templates=0,
        )
        self.print_rank_zero(
            f"Constructed validation dataset '{ds.name}':\n"
            f"  Num complexes: {len(ds.metadatas)}\n"
        )
        return ds

    def train_dataloader(self):
        dataset = self._train_ds

        sampler = DistributedWeightedSampler(
            weights=dataset.weights,
            rank=self.trainer.global_rank if self.trainer else 0,
            world_size=self.trainer.world_size if self.trainer else 1,
            epoch=self.trainer.current_epoch if self.trainer else 0,
            replacement=True,
        )
        persistent_workers = (
            self.config.persistent_workers and self.config.num_workers > 0
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            sampler=sampler,
            drop_last=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=multimer_train_collate,
        )

    def val_dataloader(self) -> DataLoader:
        dataset = self._val_ds
        sampler = None
        if self.trainer is not None:
            if self.trainer.world_size > 1:
                sampler = DistributedSampler(
                    dataset,
                    rank=self.trainer.global_rank,
                    num_replicas=self.trainer.world_size,
                    shuffle=False,
                    drop_last=False,
                )

        return DataLoader(
            dataset,
            batch_size=None,  # do not batch validation samples together
            sampler=sampler,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=False,
            persistent_workers=False,
        )

    def print_rank_zero(self, msg: str) -> None:
        if self.trainer is None or self.trainer.global_rank == 0:
            self.logger.info(f"{msg}")
