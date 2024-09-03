import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from utils.model_utils import PBitModule


def main():

    p = torch.rand(64)
    model = PBitModule(64, 8)

    dist = torch.distributions.Bernoulli(probs=p)
    sample = dist.sample((10000,))

    bit_examples = model(sample)

    p_expand = p[None].expand(10000, -1)
    model_examples = model.sample(p_expand)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].hist2d(
        bit_examples.detach().numpy()[:,0],
        bit_examples.detach().numpy()[:,1],
        bins=100
    )
    ax[0].set_xlim(-4, 4)
    ax[0].set_ylim(-4, 4)

    ax[1].hist2d(
        model_examples.detach().numpy()[:,0],
        model_examples.detach().numpy()[:,1],
        bins=100
    )
    ax[1].set_xlim(-4, 4)
    ax[1].set_ylim(-4, 4)
    plt.show()

    print(torch.mean(bit_examples, dim=0))
    print(torch.mean(model_examples, dim=0))

    print(torch.cov(bit_examples.T))
    print(torch.cov(model_examples.T))


if __name__ == '__main__':

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()