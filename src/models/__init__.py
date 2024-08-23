""" Models """

# from models.base import BaseConfig, BaseLmModel
from models.base import BaseConfig, BaseLmModel
from models.rst import RSTConfig, RSTLmModel

CONFIG_DICT = {
    "base": BaseConfig,
    "rst": RSTConfig,
}

MODEL_DICT = {
    "base": BaseLmModel,
    "rst": RSTLmModel,
}
