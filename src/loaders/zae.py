import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models import load_checkpoint
from utils.data_utils import load_byte_array
import utils.constants as constants


class ZAECollator:

    def __init__(
        self,
        checkpoint_path,
    ):
        self.model = load_checkpoint(checkpoint_path)
        
        self.model.eval()
        self.model.to(constants.DEVICE)
        

    @torch.no_grad()
    def __call__(
        self,
        data,
    ):

        # get list of arrays
        input_ids = [load_byte_array(x['input_ids.npy']) for x in data]
        output_ids = [load_byte_array(x['output_ids.npy']) for x in data]

        # get list of tensors
        input_ids = [torch.tensor(x.astype(np.int64)).long() for x in input_ids]
        output_ids = [torch.tensor(x.astype(np.int64)).long() for x in output_ids]

        # stack
        input_ids = torch.stack(input_ids, dim=0).to(constants.DEVICE)
        output_ids = torch.stack(output_ids, dim=0).to(constants.DEVICE)

        with torch.autocast("cuda", torch.bfloat16):
            target = self.model.encode(output_ids)
        target = target.float()

        return {
            "input_ids": input_ids,
            "output_ids": output_ids,
            "target": target,
        }
    