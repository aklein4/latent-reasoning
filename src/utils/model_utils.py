from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import (
    detach_variable,
    check_backward_validity,
)

try:
    import torch_xla.core.xla_model as xm
except:
    pass

import inspect

from utils.logging_utils import log_master_print
import utils.constants as constants


def _extract_tensors_from_list(inputs):
    tensor_inputs = []

    if torch.is_tensor(inputs):
        tensor_inputs.append(inputs)

    # tensor is Iterable so we need to avoid iterating through tensor
    elif isinstance(inputs, Iterable):
        for input in inputs:
            tensor_inputs += _extract_tensors_from_list(input)

    return tensor_inputs


def checkpoint_barrier(inputs):
    return
    xm.optimization_barrier_(
        _extract_tensors_from_list(inputs)
    )


class _FastCheckpointFunction(torch.autograd.Function):


  @staticmethod
  def forward(ctx, run_function, *args):
    check_backward_validity(args)
    ctx.run_function = run_function

    # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
    ctx.gpu_autocast_kwargs = {
        "enabled": torch.is_autocast_enabled(),
        "dtype": torch.get_autocast_gpu_dtype(),
        "cache_enabled": torch.is_autocast_cache_enabled()
    }
    ctx.cpu_autocast_kwargs = {
        "enabled": torch.is_autocast_cpu_enabled(),
        "dtype": torch.get_autocast_cpu_dtype(),
        "cache_enabled": torch.is_autocast_cache_enabled()
    }

    # Save non-tensor inputs in ctx, keep a placeholder None for tensors
    # to be filled out during the backward.
    ctx.inputs = []
    ctx.tensor_indices = []
    tensor_inputs = []
    for i, arg in enumerate(args):
      
      if torch.is_tensor(arg):
        tensor_inputs.append(arg)
        ctx.tensor_indices.append(i)
        ctx.inputs.append(None)

      else:
        ctx.inputs.append(arg)

    ctx.save_for_backward(*tensor_inputs)

    with torch.no_grad():
      outputs = run_function(*args)

    return outputs


  @staticmethod
  def backward(ctx, *args):
    if not torch.autograd._is_checkpoint_valid():
        raise RuntimeError(
            "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
            " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
            " argument."
        )
    
    # Copy the list to avoid modifying original list.
    inputs = list(ctx.inputs)
    tensor_indices = ctx.tensor_indices
    tensors = ctx.saved_tensors

    # Fill in inputs with appropriate saved tensors.
    for i, idx in enumerate(tensor_indices):
      inputs[idx] = tensors[i]


    # It may be more efficient to call a single optimization barrier after
    # the model forward pass, rather than after the original layers.
    # optimization_barrier_ is needed to separate the original forward pass with
    # the next forward + backward pass.
    weights = []
    buffers = []
    if (
       inspect.ismethod(ctx.run_function) and
       isinstance(ctx.run_function.__self__, torch.nn.Module)
    ):
        weights = list(ctx.run_function.__self__.parameters())
        buffers = list(ctx.run_function.__self__.buffers())
    if constants.XLA_AVAILABLE:
        xm.optimization_barrier_(
            _extract_tensors_from_list(
                inputs + list(args) +
                weights + buffers
            )
        )

    detached_inputs = detach_variable(tuple(inputs))
    with torch.enable_grad(), \
        torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
        torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
            outputs = ctx.run_function(*detached_inputs)

    if isinstance(outputs, torch.Tensor):
      outputs = (outputs,)

    # run backward() with only tensor that requires grad
    outputs_with_grad = []
    args_with_grad = []
    for i in range(len(outputs)):
      
      if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
        outputs_with_grad.append(outputs[i])
        args_with_grad.append(args[i])

    if len(outputs_with_grad) == 0:
      raise RuntimeError(
         "none of output has requires_grad=True, this checkpoint() is not necessary"
        )
    
    torch.autograd.backward(outputs_with_grad, args_with_grad)
    grads = tuple(
        inp.grad if isinstance(inp, torch.Tensor) else None
        for inp in detached_inputs
    )

    return (None,) + grads


def fast_checkpoint(function, *args):
    """ This is an optimized version of the xla gradient checkpointing function.
     - removes per-section optimization barriers
        (requires a single optimization barrier after the model forward pass!)
     - removes rng management (not needed for this project)
    """
    return _FastCheckpointFunction.apply(function, *args)


class RotaryEmbedding(nn.Module):

    def __init__(self, total_dim, frac, max_position_embeddings, base):
        super().__init__()

        assert total_dim % frac == 0, f'dimension {total_dim} must be divisible by frac {frac}'
        self.total_dim = total_dim
        self.dim = total_dim // frac
        assert self.dim % 2 == 0, f'dimension {self.dim} must be divisible by 2'

        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # inverse frequencies for rotations
        freq_ar = torch.arange(0, self.dim, 2).float()
        inv_freq = (
            1.0 /
            (self.base ** (freq_ar / self.dim))
        ) # [D/2]

        # only use integer positions, so we cache sin/cos as embeddings
        pos = torch.arange(0, self.max_position_embeddings).float()
        freqs = torch.matmul(inv_freq[:, None], pos[None, :]) # [D/2, L]
        freqs = freqs.permute(1, 0) # [L, D/2]

        freqs = torch.cat((freqs, freqs), dim=-1) # [L, D]
        sin = freqs.sin()
        cos = freqs.cos()
        
        self.sin_emb = nn.Embedding(self.max_position_embeddings, self.dim)
        self.sin_emb.weight.data = sin.contiguous()

        self.cos_emb = nn.Embedding(self.max_position_embeddings, self.dim)
        self.cos_emb.weight.data = cos.contiguous()


    def _get_sin_cos(self, position_ids):
        return (
            self.sin_emb(position_ids).detach(),
            self.cos_emb(position_ids).detach()
        )


    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


    def forward(self, q, k, position_ids):
        assert q.shape[-1] == self.total_dim, f'q shape {q.shape} does not match total_dim {self.total_dim}'
        assert k.shape[-1] == self.total_dim, f'k shape {k.shape} does not match total_dim {self.total_dim}'

        sin, cos = self._get_sin_cos(position_ids)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        if self.dim == self.total_dim:
            q = (q * cos) + (self._rotate_half(q) * sin)
            k = (k * cos) + (self._rotate_half(k) * sin)
            return q, k

        q_rot, q_no_rot = q[..., : self.dim], q[..., self.dim :]
        k_rot, k_no_rot = k[..., : self.dim], k[..., self.dim :]

        q_rot = (q_rot * cos) + (self._rotate_half(q_rot) * sin)
        k_rot = (k_rot * cos) + (self._rotate_half(k_rot) * sin)

        q = torch.cat((q_rot, q_no_rot), dim=-1)
        k = torch.cat((k_rot, k_no_rot), dim=-1)

        return q, k
    