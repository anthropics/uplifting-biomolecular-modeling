import copy
from pathlib import Path

import torch

from atlasfold.configs.atlasfold import monomer_config
from atlasfold.configs.atlasfold_multimer import multimer_config
from atlasfold.model import (
    AtlasFold,
    AtlasFold_Multimer,
    AtlasFoldConfig,
    AtlasFoldMultimerConfig,
)
from atlasfold.runner import FoldingRunner
from atlasfold.runner_multimer import MultimerFoldingRunner
from atlaslm.model import AtlasLM
from atlaslm.pretrained import download_model_weights as download_lm_weights

ATLASFOLD_260703 = "SeonghwanSeo/atlasfold-260703"
ATLASFOLD_M_260725 = "SeonghwanSeo/atlasfold-m-260725"

SUPPORTED_MODELS = [ATLASFOLD_260703, ATLASFOLD_M_260725]

MODEL_NAME_MAP = {
    "atlasfold": ATLASFOLD_260703,
    "atlasfold-260703": ATLASFOLD_260703,
    "SeonghwanSeo/atlasfold-260703": ATLASFOLD_260703,
    "atlasfold-m": ATLASFOLD_M_260725,
    "atlasfold-m-260725": ATLASFOLD_M_260725,
    "SeonghwanSeo/atlasfold-m-260725": ATLASFOLD_M_260725,
}

MODEL_CONFIGS = {
    ATLASFOLD_260703: monomer_config,
    ATLASFOLD_M_260725: multimer_config,
}

WEIGHT_PATHS = {
    ATLASFOLD_260703: (
        "SeonghwanSeo/atlasfold-260703",
        "weights/atlasfold-260703.pth",
    ),
    ATLASFOLD_M_260725: (
        "SeonghwanSeo/atlasfold-m-260725",
        "weights/atlasfold-m-260725.pth",
    ),
}


def get_model_name(model_name: str) -> str:
    if model_name not in MODEL_NAME_MAP:
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Supported models: {', '.join(SUPPORTED_MODELS)}"
        )
    return MODEL_NAME_MAP[model_name]


def download_model_weights(
    model_name: str,
    cache_dir: str | Path | None = None,
) -> Path:
    from huggingface_hub import snapshot_download

    model_name = get_model_name(model_name)
    repo_id, filename = WEIGHT_PATHS[model_name]
    repo_path = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
    )
    return Path(repo_path) / filename


def load_model(
    model_name: str = ATLASFOLD_260703,
    device: torch.device | str = "cpu",
    *,
    kernel: str = "auto",
    cache_dir: str | Path | None = None,
    lm: AtlasLM | None = None,
    lm_path: str | Path | None = None,
    model_path: str | Path | None = None,
) -> AtlasFold | AtlasFold_Multimer:
    """Load a pretrained AtlasFold model by name.

    Parameters
    ----------
    model_name : str
        The name of the pretrained model to load. Options include:
            - "atlasfold" or "atlasfold-260703"
            - "atlasfold-m" or "atlasfold-m-260725"
    device : str | torch.device, optional
        The device to load the model onto.
    kernel : str, optional
        Triangle operation backend: auto, torch, or cuequiv. Auto selects
        cuequiv when available and otherwise uses torch.
    cache_dir : str, optional
        Directory used to cache downloaded model weights.
    lm : AtlasLM, optional
        An initialized language model to share with the loaded folding model.
    lm_path : str, optional
        Path to the local pretrained AtlasLM checkpoint.
    model_path : str, optional
        Path to the local pretrained AtlasFold model checkpoint.

    Returns
    -------
    model : AtlasFold | AtlasFold_Multimer
        The loaded AtlasFold model.
    """
    device = torch.device(device)

    model_name = get_model_name(model_name)
    cfg = copy.deepcopy(MODEL_CONFIGS[model_name])

    if lm is None:
        if lm_path is None:
            lm_path = download_lm_weights(cfg.lm_name, cache_dir)
        cfg.lm_path = str(lm_path)

    if model_path is None:
        model_path = download_model_weights(model_name, cache_dir)

    if isinstance(cfg, AtlasFoldMultimerConfig):
        model = AtlasFold_Multimer.from_pretrained(
            model_path,
            config=cfg,
            lm=lm,
            device=device,
            cache_dir=cache_dir,
        )
    elif isinstance(cfg, AtlasFoldConfig):
        model = AtlasFold.from_pretrained(
            model_path,
            config=cfg,
            lm=lm,
            device=device,
            cache_dir=cache_dir,
        )
    else:
        raise TypeError(f"Unsupported AtlasFold config type: {type(cfg)!r}.")

    model.set_kernel_backend(kernel)
    return model


def get_runner(
    model: AtlasFold | AtlasFold_Multimer,
) -> FoldingRunner | MultimerFoldingRunner:
    """Return the matching inference runner for an AtlasFold model."""
    if isinstance(model, AtlasFold_Multimer):
        return MultimerFoldingRunner(model)
    if isinstance(model, AtlasFold):
        return FoldingRunner(model)

    raise TypeError(f"Unsupported AtlasFold model type: {type(model)!r}.")
