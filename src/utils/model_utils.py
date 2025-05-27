import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def momentum_scan(x, beta):
    x_curr = torch.zeros_like(x[:, :1])
    x_accum = []
        
    for i in range(x.shape[1]):

        x_curr = (
            beta * x_curr
            + (1 - beta) * x[:, i:i+1]
        )
        x_accum.append(x_curr.clone())

    return torch.cat(x_accum, dim=1)


class _ScaleGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None

def scale_gradient(x, scale):
    return _ScaleGradient.apply(x, scale)


class _PrintGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, name):
        ctx.name = name
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.name is not None:
            print(f"{ctx.name}:")
        else:
            print("Gradient:")
        print(grad_output.abs().mean(-1))
        return grad_output, None

def print_gradient(x, name=None):
    return _PrintGradient.apply(x, name)


def unsqueeze_to_batch(x, target):
    while x.dim() < target.dim():
        x = x[None]

    return x


def expand_to_batch(x, target):
    og_shape = x.shape

    num_unsqueeze = 0
    while x.dim() < target.dim():
        x = x[None]
        num_unsqueeze += 1

    x = x.expand(
        *([target.shape[i] for i in range(num_unsqueeze)] + list(og_shape))
    )

    return x


class SoftmaxPooler(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        output_size
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.output_size = output_size

        self.q_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

        self.o_proj = nn.Linear(intermediate_size, output_size, bias=False)


    def forward(self, x, dim=-2):
        
        w = torch.softmax(self.q_proj(x), dim=dim)
        v = self.v_proj(x)

        h = torch.sum(w * v, dim=dim)

        return self.o_proj(h)
