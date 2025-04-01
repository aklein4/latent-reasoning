import torch

import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get the base path of src
BASE_PATH = os.path.dirname( # src
    os.path.dirname( # utils
        __file__ # utils.constants
    )
)

# local data path
LOCAL_DATA_PATH = os.path.join(BASE_PATH, "local_data")

# paths to config files
MODEL_CONFIG_PATH = os.path.join(BASE_PATH, "model_configs")
TRAIN_CONFIG_PATH = os.path.join(BASE_PATH, "train_configs")

HF_ID = 'aklein4'
