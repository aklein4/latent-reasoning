""" Models """

from models.zlm import ZLmConfig, ZLmModel
from models.gpt_zlm import GPTZLmConfig, GPTZLmModel


CONFIG_DICT = {
    "zlm": ZLmConfig,
    "gpt_zlm": GPTZLmConfig,
}


MODEL_DICT = {
    "zlm": ZLmModel,
    "gpt_zlm": GPTZLmModel,
}
