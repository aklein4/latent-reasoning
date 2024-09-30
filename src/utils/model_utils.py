from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers.activations import ACT2FN


class ReZeroIO(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: Optional[float]=1e-5
    ):
        """ Implements affine LayerNorm input with ReZero output.

        Args:
            hidden_size (int): size of hidden dimension
            eps (float, optional): epsilon for normalization. Defaults to 1e-5.
        """
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=True)
        self.filter = nn.Parameter(torch.zeros(1, 1, hidden_size))


    def enter(
        self, 
        hidden_states: torch.FloatTensor
    ) -> torch.FloatTensor:
        """ Enters the block with layer norm.

        Args:
            x (torch.FloatTensor): residual stream

        Returns:
            torch.FloatTensor: normalized tensor
        """
        return self.norm(hidden_states)
    

    def exit(
        self,
        hidden_states: torch.FloatTensor,
        y: torch.FloatTensor
    ) -> torch.FloatTensor:
        """ Exits the block with ReZero.

        Args:
            hidden_states (torch.FloatTensor): residual stream
            y (torch.FloatTensor): output tensor from the block

        Returns:
            torch.FloatTensor: residual stream with y included
        """
        return hidden_states + self.filter * y


class FusedLinear(nn.Module):

    def __init__(
        self,
        in_feature_list: List[int],
        out_feature_list: List[int],
        bias: bool=True,
        mask: Optional[torch.FloatTensor]=None
    ):
        """ A linear layer that fuses multiple inputs and multiple outputs.
        Also supports a mask for for things like autoregressive networks.
        
        Args:
            in_feature_list (List[int]): Dimensions of each input feature (can be a single int)
            out_feature_list (List[int]): Dimensions of each output feature (can be a single int)
            bias (bool, optional): Whether to use bias in the linear layer. Defaults to True.
            mask (Optional[torch.FloatTensor], optional): A mask to multiply the linear weight by. Defaults to None.
        """
        super().__init__()

        # check for single values
        if isinstance(in_feature_list, int):
            in_feature_list = [in_feature_list]
        if isinstance(out_feature_list, int):
            out_feature_list = [out_feature_list]

        # save attributes
        self.in_feature_list = in_feature_list
        self.out_feature_list = out_feature_list
        self.bias = bias

        self.total_in = sum(in_feature_list)
        self.total_out = sum(out_feature_list)

        # parameters
        self.linear = nn.Linear(self.total_in, self.total_out, bias=bias)
    
        # save mask
        self.use_mask = mask is not None
        if self.use_mask:
            assert mask.shape == self.linear.weight.shape, f'mask shape {mask.shape} does not match weight shape {self.linear.weight.shape}'
            self.register_buffer('mask', mask)


    def _error_message(
        self,
        inputs: List[torch.Tensor]
    ) -> None:
        """ Raise an error message for incorrect input sizes.

        Args:
            inputs (List[torch.Tensor]): Input tensors

        Raises:
            ValueError: Incorrect input sizes
        """
        raise ValueError(f'expected inputs of size {self.in_feature_list}, got {[v.shape[-1] for v in inputs]}')


    def forward(
        self,
        *inputs: List[torch.FloatTensor]
    ) -> List[torch.FloatTensor]:
        """ Forward pass for the fused linear layer.

        Args:
            *inputs (List[torch.FloatTensor]): Input tensors (can be a single tensor)

        Returns:
            List[torch.FloatTensor]: Output tensors (single tensor if only one output)
        """

        # check inputs
        if len(inputs) != len(self.in_feature_list):
            self._error_message(inputs)

        # convert to single tensor and check again
        if len(self.in_feature_list) > 1:
            x = torch.cat(inputs, dim=-1)
        else:
            x = inputs[0]
        if x.shape[-1] != self.total_in:
            self._error_message(inputs)

        # apply linear
        if self.use_mask:
            x = F.linear(
                x,
                self.linear.weight * self.mask,
                self.linear.bias
            )
        else:
            x = self.linear(x)

        # convert outputs
        if len(self.out_feature_list) == 1:
            return x
        return torch.split(x, self.out_feature_list, dim=-1)


class RotaryAttention(nn.Module):

    def __init__(
        self,
        attention_head_size,
        num_attention_heads,
        num_registers,
        use_rope,
        rope_fraction,
        max_sequence_length,
        rope_base,
        layer_idx,
        position_scale=1.0
    ):
        super().__init__()

        self.layer_idx = layer_idx

        self.num_heads = num_attention_heads
        self.head_dim = attention_head_size
        self.total_dim = self.num_heads * self.head_dim

        self.use_rope = use_rope
        if self.use_rope:
            self.rope = RotaryEmbedding(
                self.head_dim, rope_fraction,
                max_sequence_length,
                rope_base,
                position_scale=position_scale
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
        position_ids=None,
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
            query_states = self.rope(query_states, position_ids)
            key_states = self.rope(key_states, position_ids)

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

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / np.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.total_dim)

        return attn_output


class RotaryEmbedding(nn.Module):

    def __init__(self, total_dim, frac, max_position_embeddings, base, position_scale=1):
        super().__init__()

        assert total_dim % frac == 0, f'dimension {total_dim} must be divisible by frac {frac}'
        self.total_dim = total_dim
        self.dim = total_dim // frac
        assert self.dim % 2 == 0, f'dimension {self.dim} must be divisible by 2'

        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.position_scale = position_scale

        # inverse frequencies for rotations
        freq_ar = torch.arange(0, self.dim, 2).float()
        inv_freq = (
            1.0 /
            (self.base ** (freq_ar / self.dim))
        ) # [D/2]

        # only use integer positions, so we cache sin/cos as embeddings
        pos = torch.arange(0, self.max_position_embeddings).float() * self.position_scale
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


    def forward(self, x, position_ids):
        assert x.shape[-1] == self.total_dim, f'shape {q.shape} does not match total_dim {self.total_dim}'

        sin, cos = self._get_sin_cos(x, position_ids)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        if self.dim == self.total_dim:
            return (x * cos) + (self._rotate_half(x) * sin)

        rot, no_rot = x[..., : self.dim], x[..., self.dim :]

        rot = (rot * cos) + (self._rotate_half(rot) * sin)

        return torch.cat((rot, no_rot), dim=-1)


class GLU(nn.Module):

    def __init__(self, activation):
        super().__init__()
        self.activation = ACT2FN[activation]
    

    def forward(self, gate, value):
        return self.activation(gate) * value
