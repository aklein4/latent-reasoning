""" Models """

# from models.base import BaseConfig, BaseLmModel
from models.base import BaseConfig, BaseLmModel
from models.swift import SwiftConfig, SwiftModel
from models.vaelm import VaeLmConfig, VaeLmModel
from models.hlm import HLmConfig, HLmModel
from models.patch_hlm import PatchHLmConfig, PatchHLmModel

CONFIG_DICT = {
    "base": BaseConfig,
    "swift": SwiftConfig,
    "vaelm": VaeLmConfig,
    "hlm": HLmConfig,
    "patch_hlm": PatchHLmConfig,
}

MODEL_DICT = {
    "base": BaseLmModel,
    "swift": SwiftModel,
    "vaelm": VaeLmModel,
    "hlm": HLmModel,
    "patch_hlm": PatchHLmModel,
}
