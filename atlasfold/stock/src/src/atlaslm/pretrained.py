import dataclasses
from pathlib import Path

import torch

from atlaslm.model import AtlasLM


@dataclasses.dataclass(kw_only=True)
class AtlasLMConfig:
    """Configuration for AtlasLM models."""

    d_model: int
    n_heads: int
    n_layers: int


# MODEL LIST
ATLASLM_600M_BASE = "SeonghwanSeo/atlaslm-600m-base"
ATLASLM_3B_BASE = "SeonghwanSeo/atlaslm-3b-base"
SUPPORTED_MODELS = [
    ATLASLM_600M_BASE,
    ATLASLM_3B_BASE,
]

# User input -> model name mapping.
MODEL_NAME_MAP = {
    "atlaslm-600m": ATLASLM_600M_BASE,
    "atlaslm-600m-base": ATLASLM_600M_BASE,
    "SeonghwanSeo/atlaslm-600m-base": ATLASLM_600M_BASE,
    "atlaslm-3b": ATLASLM_3B_BASE,
    "atlaslm-3b-base": ATLASLM_3B_BASE,
    "SeonghwanSeo/atlaslm-3b-base": ATLASLM_3B_BASE,
}

WEIGHT_PATH = {
    ATLASLM_600M_BASE: (
        "SeonghwanSeo/atlaslm-600m-base",
        "weights/atlaslm_600m_base.pth",
    ),
    ATLASLM_3B_BASE: (
        "SeonghwanSeo/atlaslm-3b-base",
        "weights/atlaslm_3b_base.pth",
    ),
}
MODEL_CONFIG_MAP = {
    ATLASLM_600M_BASE: "600m",
    ATLASLM_3B_BASE: "3b",
}
MODEL_CONFIGS = {
    "600m": AtlasLMConfig(d_model=1152, n_heads=18, n_layers=36),
    "3b": AtlasLMConfig(d_model=2304, n_heads=36, n_layers=48),
}


def get_model_name(model_name: str) -> str:
    if model_name not in MODEL_NAME_MAP:
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Supported models: {', '.join(SUPPORTED_MODELS)}"
        )
    return MODEL_NAME_MAP[model_name]


def get_model(model_name: str) -> AtlasLM:
    model_name = get_model_name(model_name)
    config_name = MODEL_CONFIG_MAP[model_name]
    config = MODEL_CONFIGS[config_name]
    model = AtlasLM(config.d_model, config.n_heads, config.n_layers)
    return model.eval()


def download_model_weights(model_name: str, cache_dir: str | Path | None = None) -> Path:
    from huggingface_hub import snapshot_download

    model_name = get_model_name(model_name)
    repo_id, filename = WEIGHT_PATH[model_name]
    repo_path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        cache_dir=cache_dir,
    )
    return Path(repo_path) / filename


def load_model(
    pretrained_model_name_or_path: str | Path = ATLASLM_3B_BASE,
    *,
    config: AtlasLMConfig | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    cache_dir: str | Path | None = None,
) -> AtlasLM:
    """Load a pretrained AtlasLM model by name or local checkpoint path.

    Parameters
    ----------
    pretrained_model_name_or_path : str | Path
        The model name or local checkpoint path. Model name options include:
        - "atlaslm-600m"
        - "atlaslm-3b"
    config : AtlasLMConfig, optional
        Model architecture. Defaults to AtlasLM 3B for a local checkpoint and
        is inferred from the model name for a Hub model.
    device : str | torch.device, optional
        The device to load the model onto, by default "cpu".
    dtype : torch.dtype, optional
        The data type to use for the model parameters, by default None.
    cache_dir : str | Path, optional
        Directory to cache the downloaded model weights, by default None.

    Returns
    -------
    model : AtlasLM
        The loaded AtlasLM model.
    """
    device = torch.device(device)
    if dtype is None:
        dtype = torch.float32 if device.type == "cpu" else torch.bfloat16

    source = pretrained_model_name_or_path
    if isinstance(source, Path):
        if not source.is_file():
            raise FileNotFoundError(f"Model checkpoint does not exist: {source}")
        model_path = source
    elif Path(source).is_file():
        model_path = Path(source)
    else:
        model_name = get_model_name(source)
        model_path = download_model_weights(model_name, cache_dir)
        if config is None:
            config_name = MODEL_CONFIG_MAP[model_name]
            config = MODEL_CONFIGS[config_name]

    if config is None:
        config = MODEL_CONFIGS["3b"]

    # Initialize the model architecture
    with torch.device("meta"):
        model = AtlasLM(config.d_model, config.n_heads, config.n_layers).to(dtype)

    # Load the model weights
    model = model.to_empty(device=device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    for module in model.modules():
        if hasattr(module, "init_buffers"):
            module.init_buffers(device=device)

    return model
