import torch
import torch.nn as nn
import torch.nn.functional as F


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
