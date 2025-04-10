""" Models """

from models.zlm import ZLmConfig, ZLmModel
from models.gpt_zlm import GPTZLmConfig, GPTZLmModel
from models.uncond_zlm import UncondZLmConfig, UncondZLmModel


CONFIG_DICT = {
    "zlm": ZLmConfig,
    "gpt_zlm": GPTZLmConfig,
    "uncond_zlm": UncondZLmConfig,
}


MODEL_DICT = {
    "zlm": ZLmModel,
    "gpt_zlm": GPTZLmModel,
    "uncond_zlm": UncondZLmModel,
}
