from typing import Dict, Any

import os
import yaml

import utils.constants as constants


def load_config(
    name: str,
    kind: str
) -> Dict[str, Any]:
    """ Get a training configuration from a file.

    Args:
        name (str): name of the config to load
        kind (str): type of the config to load, either "model" or "train"

    Returns:
        Dict[str, Any]: dictionary containing the training configuration
    """
    if kind == "model":
        path = os.path.join(constants.MODEL_CONFIG_PATH, f"{name}.yaml")
    elif kind == "train":
        path = os.path.join(constants.TRAIN_CONFIG_PATH, f"{name}.yaml")
    else:
        raise ValueError(f"Unknown config kind: {kind}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config
