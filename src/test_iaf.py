import torch

from utils.model_utils import LinearIAF


D = 10
M = 3
I = D * M


def main():

    iaf = LinearIAF(D, M)

    x = torch.randn(D, requires_grad=True)
    bias = torch.randn(D)
    inner = torch.randn(I)

    out = iaf(x, inner, bias)
    out[1].backward()

    print(x.grad)


if __name__ == '__main__':
    main()