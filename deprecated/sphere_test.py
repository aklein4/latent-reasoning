import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


N_VEC = 48 * 32
D = 32 * 4


def main():
    
    vecs = nn.Parameter(torch.randn(N_VEC, D))
    vecs.data = F.normalize(vecs, p=2, dim=1)

    optimizer = torch.optim.Adam([vecs], lr=0.01)

    for i in (pbar:=tqdm(range(500))):

        cos = torch.mm(vecs, vecs.t()).abs()
        cos.fill_diagonal_(0)

        loss = cos.max(-1)[0].mean()
        print(loss)
        break

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            vecs.data = F.normalize(vecs.data, p=2, dim=1)

        pbar.set_postfix(loss=loss.item())

    plt.scatter(vecs[:, 0].detach().numpy(), vecs[:, 1].detach().numpy())
    plt.show()

if __name__ == "__main__":

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()