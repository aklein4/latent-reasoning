""" Models """

from models.zlm import ZLmConfig, ZLmModel
from models.gpt_zlm import GPTZLmConfig, GPTZLmModel
from models.uncond_zlm import UncondZLmConfig, UncondZLmModel
from models.zae import ZAEConfig, ZAEModel


CONFIG_DICT = {
    "zlm": ZLmConfig,
    "gpt_zlm": GPTZLmConfig,
    "uncond_zlm": UncondZLmConfig,
    "zae": ZAEConfig,
}


MODEL_DICT = {
    "zlm": ZLmModel,
    "gpt_zlm": GPTZLmModel,
    "uncond_zlm": UncondZLmModel,
    "zae": ZAEModel,
}


import torch
import os
import json

import utils.constants as constants


def load_checkpoint(
    path: str,
):
    cpkt_path = os.path.join(constants.BASE_PATH, path, "checkpoint.ckpt")
    config_path = os.path.join(constants.BASE_PATH, path, "config.json")

    with open(config_path, "r") as f:
        model_type = json.load(f)["model_type"]

    config = CONFIG_DICT[model_type].from_json_file(config_path)
    model = MODEL_DICT[model_type](config)

    model.load_state_dict(
        torch.load(cpkt_path, map_location="cpu")["model"],
        strict=True
    )

    return model
