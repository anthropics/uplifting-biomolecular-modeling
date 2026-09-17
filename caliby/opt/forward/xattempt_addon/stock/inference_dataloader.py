"""
DataLoader for inference PDB featurization with multiprocess loading and prefetching.
"""

from typing import Any

from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from caliby.data.data import to
from caliby.data.datasets.atomworks_sd_dataset import sd_collator


class _InferenceDataset(Dataset):
    """Dataset wrapping get_sd_example() for individual PDBs."""

    def __init__(self, pdb_paths: list[str], data_cfg: DictConfig | None):
        self._pdb_paths = pdb_paths
        self._data_cfg = data_cfg

    def __len__(self):
        return len(self._pdb_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from caliby.eval.eval_utils.seq_des_utils import get_sd_example

        return get_sd_example(self._pdb_paths[idx], self._data_cfg)


class _EnsembleDataset(Dataset):
    """Dataset where each item is all conformers for one PDB, pre-collated via sd_collator()."""

    def __init__(
        self,
        pdb_names: list[str],
        pdb_to_conformers: dict[str, list[str]],
        data_cfg: DictConfig | None,
    ):
        self._pdb_names = pdb_names
        self._pdb_to_conformers = pdb_to_conformers
        self._data_cfg = data_cfg

    def __len__(self):
        return len(self._pdb_names)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from caliby.eval.eval_utils.seq_des_utils import get_sd_example

        pdb_name = self._pdb_names[idx]
        pdb_paths = self._pdb_to_conformers[pdb_name]
        examples = [get_sd_example(p, self._data_cfg) for p in pdb_paths]
        return sd_collator(examples)


def _identity_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Identity collate — each item from _EnsembleDataset is already a collated batch."""
    assert len(batch) == 1
    return batch[0]


class InferenceDataLoader:
    """DataLoader for inference PDB featurization with multiprocess loading and prefetching.

    Standard batched loading:
        loader = InferenceDataLoader(pdb_paths, data_cfg=..., device=..., batch_size=4, num_workers=2)
        for batch in loader:
            ...

    Ensemble loading (all conformers per PDB as one batch):
        loader = InferenceDataLoader.from_conformers(pdb_to_conformers, data_cfg=..., device=..., num_workers=2)
        for batch in loader:
            ...
    """

    def __init__(
        self,
        pdb_paths: list[str],
        *,
        data_cfg: DictConfig | None,
        device: str,
        batch_size: int,
        num_workers: int = 0,
        prefetch_factor: int = 1,
    ):
        self._device = device
        dataset = _InferenceDataset(pdb_paths, data_cfg)
        worker_kwargs = {}
        if num_workers > 0:
            worker_kwargs["persistent_workers"] = True
            worker_kwargs["prefetch_factor"] = prefetch_factor
        self._loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=sd_collator,
            **worker_kwargs,
        )

    @classmethod
    def from_conformers(
        cls,
        pdb_to_conformers: dict[str, list[str]],
        *,
        data_cfg: DictConfig | None,
        device: str,
        num_workers: int = 0,
        prefetch_factor: int = 1,
    ) -> "InferenceDataLoader":
        """Create a loader where each batch contains all conformers for one PDB."""
        pdb_names = list(pdb_to_conformers.keys())
        dataset = _EnsembleDataset(pdb_names, pdb_to_conformers, data_cfg)
        worker_kwargs = {}
        if num_workers > 0:
            worker_kwargs["persistent_workers"] = True
            worker_kwargs["prefetch_factor"] = prefetch_factor
        obj = cls.__new__(cls)
        obj._device = device
        obj._loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_identity_collate,
            **worker_kwargs,
        )
        return obj

    def __iter__(self):
        for batch in self._loader:
            yield to(batch, self._device)

    def __len__(self):
        return len(self._loader)

    @property
    def dataset_size(self) -> int:
        return len(self._loader.dataset)
