
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class PBitModule(nn.Module):

    def __init__(self, bit_size, z_size):
        super().__init__()

        self.bit_size = bit_size
        self.z_size = z_size

        self.A = nn.Parameter(torch.zeros(self.z_size, self.bit_size))
        self.A.data.normal_(0, 1 / np.sqrt(self.bit_size))


    def forward(self, bits):
        return F.linear(2*bits-1, self.A)

    
    def sample(self, p, noise=None):
        
        shape = p.shape[:-1]
        p = p.view(-1, self.bit_size)
        if noise is not None:
            noise = noise.view(-1, self.z_size)

        mu = F.linear(2*p-1, self.A)

        var = 4 * p * (1 - p)
        cov = (self.A[None] * var.unsqueeze(-2)) @ self.A.T[None]
        
        chol = torch.linalg.cholesky_ex(cov)[0]

        if noise is None:
            noise = torch.randn_like(mu)

        sample = mu + (chol @ noise.unsqueeze(-1)).squeeze(-1)

        return sample.view(*shape, self.z_size)
        

class GaussianMixtureModule(nn.Module):

    def __init__(self, input_dim, output_dim, n_components):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_components = n_components

        self.proj = nn.Linear(input_dim, self.n_components, bias=False)

        self.mu = nn.Parameter(torch.randn(self.n_components, self.output_dim))
        self.log_sigma = nn.Parameter(torch.zeros(self.n_components, self.output_dim))


    def forward(self, x):
        logpi = torch.log_softmax(self.proj(x), dim=-1)

        return GaussianMixtureDistribution(
            logpi,
            self.mu,
            F.softplus(self.log_sigma)
        )


class GaussianMixtureDistribution:

    def __init__(self, logpi, mu, sigma):

        self.shape = logpi.shape[:-1]
        self.k = logpi.shape[-1]
        self.d = mu.shape[-1]

        self.logpi = logpi.view(-1, self.k) # [R, K]
        self.mu = mu # [K, D]
        self.sigma = sigma # [K, D]

    
    def sample(self, n_samples):

        # sample from categorical distribution [n, R]
        z = torch.distributions.Categorical(logits=self.logpi).sample((n_samples,))
        z = z.view(-1)

        # sample from gaussian distribution
        mu = self.mu[z].view(n_samples, *self.shape, -1)
        sigma = self.sigma[z].view(n_samples, *self.shape, -1)
        return mu + sigma * torch.randn_like(mu)
    

    def log_prob(self, x):

        n = x.shape[0]
        x = x.view(n, -1, 1, self.d) # [n, R, 1, D]

        mu_n = self.mu[None, None] # [1, 1, K, D]
        sigma_n = self.sigma[None, None] # [1, 1, K, D]
        logpi_n = self.logpi[None] # [1, R, K]

        log_probs = -0.5 * (
            2 * torch.log(sigma_n) +
            np.log(2 * np.pi) +
            ((x - mu_n) / sigma_n) ** 2
        ) # [n, R, K, D]

        log_probs = torch.sum(log_probs, dim=-1) # [n, R, K]
        log_probs = torch.logsumexp(logpi_n + log_probs, dim=-1)

        return log_probs.view(n, *self.shape)