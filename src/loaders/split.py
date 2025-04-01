
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.data_utils import load_byte_array
import utils.constants as constants


class SplitCollator:

    def __init__(
        self,
    ):
        pass
        

    def __call__(
        self,
        data,
    ):

        # get list tensors
        input_ids = [x['input_ids.npy'] for x in data]
        output_ids = [x['output_ids.npy'] for x in data]
        try:
            input_ids = [load_byte_array(x) for x in input_ids]
            output_ids = [load_byte_array(x) for x in output_ids]
        except:
            input_ids = [np.array(x) for x in input_ids]
            output_ids = [np.array(x) for x in output_ids]

        input_ids = [torch.tensor(x.astype(np.int64)).long() for x in input_ids]
        output_ids = [torch.tensor(x.astype(np.int64)).long() for x in output_ids]

        return {
            "input_ids": input_ids,
            "output_ids": output_ids,
        }
    