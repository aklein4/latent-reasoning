""" Models """

# from models.base import BaseConfig, BaseLmModel
from models.base import BaseConfig, BaseLmModel
from models.rat import RatConfig, RatLmModel
from models.rst import RSTConfig, RSTLmModel
from models.asm import ASMConfig, ASMLmModel

CONFIG_DICT = {
    "base": BaseConfig,
    "rat": RatConfig,
    "rst": RSTConfig,
    "asm": ASMConfig,
}

MODEL_DICT = {
    "base": BaseLmModel,
    "rat": RatLmModel,
    "rst": RSTLmModel,
    "asm": ASMLmModel,
}
