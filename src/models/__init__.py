""" Models """

from models.hlm import HLmConfig, HLmModel
from models.patch_hlm import PatchHLmConfig, PatchHLmModel
from models.zlm import ZLmConfig, ZLmModel
from models.markovlm import MarkovLmConfig, MarkovLmModel

CONFIG_DICT = {
    "hlm": HLmConfig,
    "patch_hlm": PatchHLmConfig,
    "zlm": ZLmConfig,
    "markovlm": MarkovLmConfig,
}

MODEL_DICT = {
    "hlm": HLmModel,
    "patch_hlm": PatchHLmModel,
    "zlm": ZLmModel,
    "markovlm": MarkovLmModel,
}
