
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torch_xla.core.xla_model as xm
except ImportError:
    pass

import numpy as np

from transformers.activations import ACT2FN

import utils.constants as constants


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, affine=True):
        super().__init__()

        self.affine = affine
        self.weight = None
        if affine:
            self.weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
        self.variance_epsilon = eps


    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        
        if self.affine:
            hidden_states = hidden_states * (1+self.weight)
        
        return hidden_states.to(input_dtype)


class FusedLinear(nn.Module):

    def __init__(
        self,
        in_feature_list,
        out_feature_list,
        bias=True,
        contiguous=False
    ):
        super().__init__()

        if isinstance(in_feature_list, int):
            in_feature_list = [in_feature_list]
        if isinstance(out_feature_list, int):
            out_feature_list = [out_feature_list]

        self.in_feature_list = in_feature_list
        self.out_feature_list = out_feature_list
        self.bias = bias
        self.contiguous = contiguous

        self.total_in = sum(in_feature_list)
        self.total_out = sum(out_feature_list)

        self.linear = nn.Linear(self.total_in, self.total_out, bias=bias)
    

    def _error_message(self, inputs):
        raise ValueError(f'expected inputs of size {self.in_feature_list}, got {[v.shape[-1] for v in inputs]}')


    def forward(self, *inputs, in_scale=None, scale=None, bias=None):
        if len(inputs) != len(self.in_feature_list):
            self._error_message(inputs)

        if len(self.in_feature_list) > 1:
            x = torch.cat(inputs, dim=-1)
        else:
            x = inputs[0]
        if x.shape[-1] != self.total_in:
            self._error_message(inputs)

        if in_scale is not None:
            x = x * in_scale

        x = self.linear(x)

        if scale is not None:
            x = x * scale
        if bias is not None:
            x = x + bias

        if len(self.out_feature_list) == 1:
            return x
        
        out = torch.split(x, self.out_feature_list, dim=-1)
        if self.contiguous:
            out = tuple(v.contiguous() for v in out)
        
        return out


class RotaryAttention(nn.Module):

    def __init__(
        self,
        hidden_size,
        attention_head_size,
        num_attention_heads,
        num_registers,
        use_rope,
        rope_fraction,
        max_sequence_length,
        rope_base,
        layer_idx
    ):
        super().__init__()

        self.layer_idx = layer_idx

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = attention_head_size
        self.total_dim = self.num_heads * self.head_dim

        self.use_rope = use_rope
        if self.use_rope:
            self.rope = RotaryEmbedding(
                self.head_dim, rope_fraction,
                max_sequence_length,
                rope_base
            )
        else:
            self.rope = None

        self.num_registers = num_registers
        if self.num_registers > 0:
            self.k_registers = nn.Parameter(
                torch.randn(1, self.num_heads, self.num_registers, self.head_dim)
            )
            self.v_registers = nn.Parameter(
                torch.randn(1, self.num_heads, self.num_registers, self.head_dim)
            )


    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask=None,
        past_key_value=None,
    ):

        # get shapes
        bsz, q_len, _ = query_states.shape

        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # apply rope
        if self.use_rope:
            query_states, key_states = self.rope(query_states, key_states, position_ids)

        # update/apply cache
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        # apply registers
        if self.num_registers > 0:
            key_states = torch.cat((key_states, self.k_registers.expand(bsz, -1, -1, -1)), dim=2)
            value_states = torch.cat((value_states, self.v_registers.expand(bsz, -1, -1, -1)), dim=2)

            if attention_mask is not None:
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.zeros(*attention_mask.shape[:-1], self.num_registers, dtype=attention_mask.dtype, device=attention_mask.device)
                    ],
                    dim=-1
                )

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3) / np.sqrt(self.head_dim))
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.total_dim)

        return attn_output


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
        sin = freqs.sin().contiguous()
        cos = freqs.cos().contiguous()
        
        self.register_buffer('sin_emb', sin, persistent=True)
        self.register_buffer('cos_emb', cos, persistent=True)


    def _get_sin_cos(self, x, position_ids):
        if position_ids is None:
            return (
                self.sin_emb[:x.shape[2]][None].detach(),
                self.cos_emb[:x.shape[2]][None].detach()
            )

        return (
            F.embedding(position_ids, self.sin_emb).detach(),
            F.embedding(position_ids, self.cos_emb).detach()
        )


    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


    def forward(self, q, k, position_ids):
        assert q.shape[-1] == self.total_dim, f'q shape {q.shape} does not match total_dim {self.total_dim}'
        assert k.shape[-1] == self.total_dim, f'k shape {k.shape} does not match total_dim {self.total_dim}'

        sin, cos = self._get_sin_cos(q, position_ids)
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


class GLU(nn.Module):

    def __init__(self, activation):
        super().__init__()
        self.activation = ACT2FN[activation]
    

    def forward(self, gate, value):
        return self.activation(gate) * value


from typing import Iterable
from torch.utils.checkpoint import (
    detach_variable,
    check_backward_validity,
)


def _extract_tensors_from_list(inputs):
    tensor_inputs = []

    if torch.is_tensor(inputs):
        tensor_inputs.append(inputs)

    # tensor is Iterable so we need to avoid iterating through tensor
    elif isinstance(inputs, Iterable):
        for input in inputs:
            tensor_inputs += _extract_tensors_from_list(input)

    return tensor_inputs


class _ModelCheckpointFunction(torch.autograd.Function):


  @staticmethod
  def forward(ctx, model, run_function, *args):
    ctx.model = model
    
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
    weights = list(ctx.model.parameters())
    buffers = list(ctx.model.buffers())
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

    return (None,None) + grads


def model_checkpoint(model, function, *args):
    """ This is an optimized version of the xla gradient checkpointing function.
     - removes per-section optimization barriers
        (requires a single optimization barrier after the model forward pass!)
     - removes rng management (not needed for this project)
    """
    return _ModelCheckpointFunction.apply(model, function, *args)
