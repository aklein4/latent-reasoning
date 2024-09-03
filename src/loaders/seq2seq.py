
import torch

import numpy as np

from utils.data_utils import load_byte_array


class Seq2SeqCollator:

    def __init__(
        self,
        seq_length: int,
        pad_token_id: int,
    ):
        self.seq_length = seq_length
        self.pad_token_id = pad_token_id
        

    def __call__(
        self,
        data,
    ):

        # get list tensors
        input_ids = [x['input_ids.npy'] for x in data]
        try:
            input_ids = [load_byte_array(x) for x in input_ids]
        except:
            pass
        input_ids = [torch.tensor(x.astype(np.int64)).long() for x in input_ids]
        
        # apply max length
        for i in range(len(input_ids)):
            if input_ids[i].shape[0] > self.seq_length:
                input_ids[i] = input_ids[i][:self.seq_length]

        # create mask
        lengths = torch.tensor(np.random.randint(
            1,
            [x.shape[-1]-1 for x in input_ids],
        ))
        ar = torch.arange(self.seq_length)
        mask = torch.ones(len(input_ids), self.seq_length, dtype=torch.bool)
        mask = torch.where(ar[None] < lengths, mask, torch.zeros_like(mask))

        # pad into single tensor
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id
        )

        return input_ids, mask
    