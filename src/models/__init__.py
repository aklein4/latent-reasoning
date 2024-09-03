""" Models """

# from models.base import BaseConfig, BaseLmModel
from models.base import BaseConfig, BaseLmModel
from models.swift import SwiftConfig, SwiftModel

CONFIG_DICT = {
    "base": BaseConfig,
    "swift": SwiftConfig,
}

MODEL_DICT = {
    "base": BaseLmModel,
    "swift": SwiftModel,
}
