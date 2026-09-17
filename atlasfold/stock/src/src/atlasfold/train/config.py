from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(
    path: str | Path,
    override_args: list[str] | None = None,
) -> DictConfig:
    """
    Load a configuration file from the given path with recursive _yaml_ inheritance.

    Args:
        path (str | Path): The path to the configuration file.
        override_args (list[str] | None): A list of dotlist strings to override specific
            configuration values.
        override_registry_defaults (bool): Whether to override registry defaults.

    Returns:
        DictConfig: The loaded configuration as a DictConfig object.
    """
    config: DictConfig = OmegaConf.load(path)

    if override_args is not None:
        # Override specific arguments in the config
        overrides = OmegaConf.from_dotlist(override_args)
        config = OmegaConf.merge(config, overrides)

    config = _resolve_yaml_inheritance(config, Path(path).parent)
    return config


def print_config(config: DictConfig) -> None:
    """Print the configuration in a human-readable format."""
    print(OmegaConf.to_yaml(config))


def save_config(config: DictConfig, save_path: str | Path) -> None:
    """Save the configuration to a YAML file."""
    OmegaConf.save(config, save_path)


def to_dict(config: DictConfig) -> dict:
    """Convert a DictConfig to a standard Python dictionary."""
    return OmegaConf.to_container(config, resolve=True)


def _resolve_yaml_inheritance(config: DictConfig, base_path: Path) -> DictConfig:
    container: dict = OmegaConf.to_container(config, resolve=True)

    def _resolve_yaml_inheritance(obj: Any, base_path: Path) -> Any:
        """Recursively resolve _yaml_ inheritance in nested structures."""
        if isinstance(obj, dict):
            # Check if current dict has _yaml_ and resolve it first
            if "_yaml_" in obj:
                yaml_path = base_path / obj.pop("_yaml_")
                base_config = load_config(yaml_path)
                obj = OmegaConf.merge(base_config, obj)
                obj = OmegaConf.to_container(obj, resolve=True)

            # Recursively process all nested dicts
            resolved = {}
            for key, value in obj.items():
                resolved[key] = _resolve_yaml_inheritance(value, base_path)
        elif isinstance(obj, list):
            resolved = [_resolve_yaml_inheritance(item, base_path) for item in obj]
        else:
            return obj

        return resolved

    resolved_container = _resolve_yaml_inheritance(container, base_path)
    return OmegaConf.create(resolved_container)
