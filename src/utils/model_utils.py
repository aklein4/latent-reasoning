
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers.activations import ACT2FN


class FusedLinear(nn.Module):

    def __init__(
        self,
        in_feature_list,
        out_feature_list,
        bias=True
    ):
        super().__init__()

        if isinstance(in_feature_list, int):
            in_feature_list = [in_feature_list]
        if isinstance(out_feature_list, int):
            out_feature_list = [out_feature_list]

        self.in_feature_list = in_feature_list
        self.out_feature_list = out_feature_list
        self.bias = bias

        self.total_in = sum(in_feature_list)
        self.total_out = sum(out_feature_list)

        self.linear = nn.Linear(self.total_in, self.total_out, bias=bias)
    

    def _error_message(self, inputs):
        raise ValueError(f'expected inputs of size {self.in_feature_list}, got {[v.shape[-1] for v in inputs]}')


    def forward(self, *inputs):
        if len(inputs) != len(self.in_feature_list):
            self._error_message(inputs)

        if len(self.in_feature_list) > 1:
            x = torch.cat(inputs, dim=-1)
        else:
            x = inputs[0]
        if x.shape[-1] != self.total_in:
            self._error_message(inputs)

        x = self.linear(x)

        if len(self.out_feature_list) == 1:
            return x
        return torch.split(x, self.out_feature_list, dim=-1)


class RotaryAttention(nn.Module):

    def __init__(
        self,
        hidden_size,
        attention_head_size,
        num_attention_heads,
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

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # apply rope
        if self.use_rope:
            query_states, key_states = self.rope(query_states, key_states, position_ids)

        # update/apply cache
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

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


class GLU(nn.Module):

    def __init__(self, activation):
        super().__init__()
        self.activation = ACT2FN[activation]
    

    def forward(self, gate, value):
        return self.activation(gate) * value
    