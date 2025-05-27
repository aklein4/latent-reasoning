
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.data_utils import load_byte_array
import utils.constants as constants


class MaskedCollator:

    def __init__(
        self,
        sequence_length: int,
        pad_token_id: int,
    ):
        self.seq_length = sequence_length
        self.pad_token_id = pad_token_id
        

    def __call__(
        self,
        data,
    ):

        # get list of arrays
        input_ids = [load_byte_array(x['input_ids.npy']) for x in data]

        # get list of tensors
        input_ids = [torch.tensor(x.astype(np.int64)).long() for x in input_ids]

        # save the segment lengths
        lengths = [x.shape[0] for x in input_ids]

        # pad into single tensor
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id
        )

        # apply seq_length constraint
        if input_ids.shape[1] < self.seq_length:
            input_ids = F.pad(
                input_ids,
                (0, self.seq_length - input_ids.shape[1]),
                value=self.pad_token_id
            )
        elif input_ids.shape[1] > self.seq_length:
            input_ids = input_ids[:, :self.seq_length]
        
        # create the mask
        mask = torch.zeros_like(input_ids).to(torch.float32)
        for i, length in enumerate(lengths):
            mask[i, :length] = 1.0

        return {
            "input_ids": input_ids.to(constants.DEVICE),
            "mask": mask.to(constants.DEVICE),
        }
    