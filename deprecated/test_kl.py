import torch
import torch.nn as nn
import torch.nn.functional as F


def kl(enc_mu, enc_sigma, dec_mu, dec_sigma):
    """
    Compute KL divergence between two Gaussians.
    """
    return (
        torch.log(dec_sigma) - torch.log(enc_sigma)
        + 0.5 * (enc_sigma**2 + (enc_mu-dec_mu)**2) / (dec_sigma**2)
        - 0.5
    )


def main():
    
    enc_mu = nn.Parameter(torch.randn(1))
    enc_sigma = nn.Parameter(torch.rand(1))
    dec_mu = nn.Parameter(torch.randn(1))
    dec_sigma = nn.Parameter(torch.rand(1))
    
    optimizer = torch.optim.SGD([enc_mu, enc_sigma, dec_mu, dec_sigma], lr=1e-2)

    for i in range(1000):
        optimizer.zero_grad()

        loss = kl(enc_mu, enc_sigma.exp(), dec_mu, dec_sigma.exp())
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f"Step {i}: {loss.item()}")
            print(f"Enc mu: {enc_mu.item()}, Enc sigma: {enc_sigma.item()}")
            print(f"Dec mu: {dec_mu.item()}, Dec sigma: {dec_sigma.item()}")
            print()


if __name__ == '__main__':
    main()