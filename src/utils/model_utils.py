
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers.activations import ACT2FN


class GaussianIAF(nn.Module):

    def __init__(self, hidden_size, z_size, mlp_mult, activation):
        super().__init__()

        self.hidden_size = hidden_size
        self.z_size = z_size
        self.mlp_mult = mlp_mult

        self.cat_size = hidden_size + z_size
        self.z_mlp_size = mlp_mult * z_size

        self.up = FusedLinear(
            [self.hidden_size, self.z_size],
            [2*self.z_size] + [self.z_mlp_size]*2,
            bias=False,
            mask=self._get_up_mask()
        )
        self.down = FusedLinear(
            self.z_mlp_size,
            2*self.z_size,
            bias=False,
            mask=self._get_down_mask()
        )

        self.mlp = GLU(activation)


    @torch.no_grad()
    def _get_up_mask(self):
        full_size = 2*self.z_size + 2*self.z_mlp_size

        # hidden states can apply to anything
        hidden_mask = torch.ones(full_size, self.hidden_size)

        # bias is auto-regressive (no diagonal)
        noise_bias_mask = torch.tril(torch.ones(self.z_size, self.z_size), diagonal=-1)
        noise_bias_mask = noise_bias_mask.repeat(2, 1)

        # mlp is auto-regressive (with diagonal)
        noise_mlp_mask = torch.tril(torch.ones(self.z_size, self.z_size), diagonal=0)
        noise_mlp_mask = noise_mlp_mask.repeat_interleave(self.mlp_mult, dim=0)
        noise_mlp_mask = noise_mlp_mask.repeat(2, 1)

        # combine noise masks
        noise_mask = torch.cat([noise_bias_mask, noise_mlp_mask], dim=0)

        # combine all masks
        return torch.cat([hidden_mask, noise_mask], dim=1)


    @torch.no_grad()
    def _get_down_mask(self):
        mask = torch.tril(torch.ones(self.z_size, self.z_size), diagonal=-1)
        mask = mask.repeat_interleave(self.mlp_mult, dim=1)
        out = mask.repeat(2, 1)
        return out


    def forward(self, hidden_states, noise): 
        z_bias, z_gate, z_val = self.up(
            hidden_states, noise
        )

        # returns concatination of mu and log_sigma
        return z_bias + self.down(self.mlp(z_gate, z_val))


class RMSNorm(nn.Module):
    pass


class FusedLinear(nn.Module):

    def __init__(
        self,
        in_feature_list,
        out_feature_list,
        bias=True,
        mask=None
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
    
        self.use_mask = mask is not None
        if self.use_mask:
            assert mask.shape == self.linear.weight.shape, f'mask shape {mask.shape} does not match weight shape {self.linear.weight.shape}'
            self.register_buffer('mask', mask)


    def _error_message(self, inputs):
        raise ValueError(f'expected inputs of size {self.in_feature_list}, got {[v.shape[-1] for v in inputs]}')


    def forward(self, *inputs):
        if len(inputs) != len(self.in_feature_list):
            self._error_message(inputs)

        # configure and check inputs
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

        # configure outputs
        if len(self.out_feature_list) == 1:
            return x
        return torch.split(x, self.out_feature_list, dim=-1)


class FullRotaryAttention(nn.Module):

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
        layer_idx,
        matrix_mask=None,
        out_size=None,
        position_scale=1.0
    ):
        super().__init__()

        qkv_size = attention_head_size * num_attention_heads
        
        self.QKV = FusedLinear(hidden_size, [qkv_size]*3, bias=False, mask=matrix_mask)
        self.O = nn.Linear(qkv_size, out_size if out_size is not None else hidden_size, bias=False)

        self.attn = RotaryAttention(
            hidden_size,
            attention_head_size,
            num_attention_heads,
            num_registers,
            use_rope,
            rope_fraction,
            max_sequence_length,
            rope_base,
            layer_idx,
            position_scale=position_scale
        )

    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids=None,
        attention_mask=None,
        past_key_value=None,
    ):
        q, k, v = self.QKV(hidden_states)

        attn_output = self.attn(q, k, v, position_ids, attention_mask, past_key_value)

        return self.O(attn_output)


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
        layer_idx,
        position_scale=1.0
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


class FullGLU(nn.Module):

    def __init__(self, hidden_size, mlp_size, activation, out_size=None):
        super().__init__()

        self.gate_proj = nn.Linear(hidden_size, mlp_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, mlp_size, bias=False)
        self.down_proj = nn.Linear(mlp_size, out_size if out_size is not None else hidden_size, bias=False)

        self.glu = GLU(activation)


    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        return self.down_proj(self.glu(gate, up))


class GLU(nn.Module):

    def __init__(self, activation):
        super().__init__()
        self.activation = ACT2FN[activation]
    

    def forward(self, gate, value):
        return self.activation(gate) * value
