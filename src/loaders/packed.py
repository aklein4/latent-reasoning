
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.data_utils import load_byte_array
import utils.constants as constants


class PackedCollator:

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
        segment_ids = [load_byte_array(x['segment_ids.npy']) for x in data]

        # get list of tensors
        input_ids = [torch.tensor(x.astype(np.int64)).long() for x in input_ids]
        segment_ids = [torch.tensor(x.astype(np.int64)).long() for x in segment_ids]

        # pad into single tensor
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id
        )
        segment_ids = torch.nn.utils.rnn.pad_sequence(
            segment_ids,
            batch_first=True,
            padding_value=-1
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

        if segment_ids.shape[1] < self.seq_length:
            segment_ids = F.pad(
                segment_ids,
                (0, self.seq_length - segment_ids.shape[1]),
                value=-1
            )
        elif segment_ids.shape[1] > self.seq_length:
            segment_ids = segment_ids[:, :self.seq_length]
        
        return {
            "input_ids": input_ids,
            "segment_ids": segment_ids,
        }
    