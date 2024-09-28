import torch

from loaders import get_loader


def main():

    x = torch.randn(8)
    x.requires_grad = True

    y = x + 5
    y[:4] = y[:4] * 0.5 + torch.randn(4)

    loss = y.sum()
    loss.backward()
    print(x.grad)
    return

    loader = get_loader(
        'fw-45b',
        "train",
        32,
        "seq2seq",
        {
            "sequence_length": 512,
        },
        True
    )

    for x, mask in loader:
        break


if __name__ == '__main__':
    main()