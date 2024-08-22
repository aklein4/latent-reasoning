import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt


def main():
    
    conv = nn.Conv1d(4, 12, 1, groups=2, bias=False)
    # conv.weight.data /= conv.weight.data.norm(p=2, dim=1, keepdim=True)

    v = torch.randn(4)
    # v[:4] /= v[:4].norm(p=2, dim=-1, keepdim=True)
    # v[4:] /= v[4:].norm(p=2, dim=-1, keepdim=True)

    # print(conv.weight.data)
    # print(conv.weight.data.shape)
    # print(v)

    print(conv(v[None].unsqueeze(-1)))
    # print(conv.weight.data[:2, :, 0] @ v[:6].unsqueeze(-1))
    # print(conv.weight.data[2:, :, 0] @ v[6:].unsqueeze(-1))

    print(torch.matmul(conv.weight.data[:6, :, 0], v[:2].unsqueeze(-1)))

if __name__ == "__main__":

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()