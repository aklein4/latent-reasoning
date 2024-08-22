import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

from utils.model_utils import RotaryEmbedding


def main():
    
    rope = RotaryEmbedding(16, 1024, 10000)
    llama = LlamaRotaryEmbedding(16, 1024, 10000)

    q, k = torch.randn(512, 32, 64), torch.randn(512, 32, 64)
    q = q.view(512, 32, 4, 16).transpose(1, 2)
    k = k.view(512, 32, 4, 16).transpose(1, 2)

    p = torch.randint(0, 1024, (512, 32))

    rope_q, rope_k = rope(q, k, p)

    cos, sin = llama(q, p)
    llama_q, llama_k = apply_rotary_pos_emb(q, k, cos, sin)

    print((rope_q - llama_q).abs().max())
    print((rope_k - llama_k).abs().max())


if __name__ == "__main__":

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()