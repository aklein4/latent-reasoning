""" Models """

from models.zlm import ZLmConfig, ZLmModel
from models.gpt_zlm import GPTZLmConfig, GPTZLmModel
from models.uncond_zlm import UncondZLmConfig, UncondZLmModel
from models.zae import ZAEConfig, ZAEModel
from models.zlm_full import ZLmFullConfig, ZLmFullModel
from models.zlm_contrast import ZLmContrastConfig, ZLmContrastModel


CONFIG_DICT = {
    "zlm": ZLmConfig,
    "gpt_zlm": GPTZLmConfig,
    "uncond_zlm": UncondZLmConfig,
    "zae": ZAEConfig,
    "zlm_full": ZLmFullConfig,
    "zlm_contrast": ZLmContrastConfig,
}


MODEL_DICT = {
    "zlm": ZLmModel,
    "gpt_zlm": GPTZLmModel,
    "uncond_zlm": UncondZLmModel,
    "zae": ZAEModel,
    "zlm_full": ZLmFullModel,
    "zlm_contrast": ZLmContrastModel,
}


import torch
import os
import json

import utils.constants as constants


def load_checkpoint(
    path: str,
    strict: bool = True,
    config: dict = None,
    **model_kwargs
):
    cpkt_path = os.path.join(constants.BASE_PATH, path, "checkpoint.ckpt")
    config_path = os.path.join(constants.BASE_PATH, path, "config.json")

    if config is None:
        with open(config_path, "r") as f:
            model_type = json.load(f)["model_type"]

        config = CONFIG_DICT[model_type].from_json_file(config_path)

    else:
        model_type = config["model_type"]

        config = CONFIG_DICT[model_type](**config)
        
    model = MODEL_DICT[model_type](config, **model_kwargs)

    loaded_state = torch.load(cpkt_path, map_location="cpu")["model"]
    if not strict:

        current_state = model.state_dict()
        filtered_state = {}
        for k, v in loaded_state.items():
            
            if (not k in current_state.keys()) or (loaded_state[k].shape == current_state[k].shape):
                filtered_state[k] = v
        
        loaded_state = filtered_state

    model.load_state_dict(
        torch.load(cpkt_path, map_location="cpu")["model"],
        strict=strict,
    )

    return model
