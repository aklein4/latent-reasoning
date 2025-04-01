import torch

import io
import numpy as np


def load_byte_array(
    data: bytes
) -> torch.LongTensor:
    """ Convert the data from a byte stream to a tensor.
        - see npy_loads() in https://github.com/webdataset/webdataset/blob/main/webdataset/autodecode.py
    """
    stream = io.BytesIO(data)
    return np.lib.format.read_array(stream)
