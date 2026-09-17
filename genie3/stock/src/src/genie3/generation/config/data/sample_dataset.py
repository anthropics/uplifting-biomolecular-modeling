"""
Sample dataset configuration templates.

Provides configuration templates for different sampling scenarios:
- Unconditional: Generate proteins of varying lengths
- Motif: Scaffold design with motif constraints
- Target: Binder design for specific targets
- Multitarget: Binder design for multiple targets
"""

import logging
from ml_collections import ConfigDict
from ml_collections.config_dict import required_placeholder


def get_sample_dataset_config(source: str) -> ConfigDict:
    """
    Get configuration template for a sample dataset.

    Args:
        source: Dataset type ('unconditional', 'motif', 'target')

    Returns:
        ConfigDict with required and default parameters for the specified dataset
    """
    if source == "unconditional":
        return ConfigDict(
            {
                "min_length": required_placeholder(int),
                "max_length": required_placeholder(int),
                "length_step": required_placeholder(int),
                "n_sample": required_placeholder(int),
                "batch_size": required_placeholder(int),
            }
        )
    elif source == "motif":
        return ConfigDict(
            {
                "datadir": required_placeholder(str),
                "n_sample": required_placeholder(int),
                "selections": None,
                "batch_size": required_placeholder(int),
            }
        )
    elif source == "target":
        return ConfigDict(
            {
                "datadir": required_placeholder(str),
                "n_sample": required_placeholder(int),
                "tags": None,
                "selections": None,
                "cond_strategy": "extended",
            }
        )
    else:
        logging.error(f"Invalid sample dataset source: {source}")
        exit(0)
