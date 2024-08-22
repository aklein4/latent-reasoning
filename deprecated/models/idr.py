from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import (
    BaseConfig, BaseTransformer, BaseLmModel,
    BaseAttention, BaseMLP
)


class IDRConfig(BaseConfig):
    model_type = 'idr'


class IDRInputNorm(nn.Module):

    def __init__(self, hidden_size, eps):
        super().__init__()

        self.register_norm = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=False)
        self.register_scales = nn.Parameter(torch.ones(1, 1, hidden_size))
        
        self.skip_norm = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=False)
        self.skip_scales = nn.Parameter(torch.ones(1, 1, hidden_size))

        self.bias = nn.Parameter(torch.zeros(1, 1, hidden_size))


    def forward(self, hidden_states):
        register_states, skip_states = hidden_states.chunk(2, dim=-1)

        register_states = self.register_norm(register_states) * self.register_scales
        skip_states = self.skip_norm(skip_states) * self.skip_scales

        return (register_states + skip_states)/2 + self.bias


class IDRSkip(nn.Module):

    def __init__(self, hidden_size, eps):
        super().__init__()

        self.gates = nn.Parameter(torch.zeros(1, 1, hidden_size))


    def forward(self, hidden_states, y):
        register_states, skip_states = hidden_states.chunk(2, dim=-1)

        register_states = (
            self.gates * register_states +
            (1 - self.gates) * y
        )
        
        return torch.cat([register_states, skip_states], dim=-1)


class IDRLayer(nn.Module):
    def __init__(self, config: IDRConfig, layer_idx: int):
        super().__init__()

        self.hidden_size = config.hidden_size

        self.attn = BaseAttention(config, layer_idx)
        self.mlp = BaseMLP(config)

        h = config.hidden_size
        eps = config.layer_norm_eps

        self.attn_input_norm = IDRInputNorm(h, eps)
        self.mlp_input_norm = IDRInputNorm(h, eps)

        self.attn_skip = IDRSkip(h, eps)
        self.mlp_skip = IDRSkip(h, eps)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value = None,
    ):

        # Self Attention
        attn_out = self.attn(
            self.attn_input_norm(hidden_states),
            attention_mask,
            past_key_value=past_key_value
        )
        hidden_states = self.attn_skip(hidden_states, attn_out)

        # GLU MLP
        mlp_out = self.mlp(
            self.mlp_input_norm(hidden_states)
        )
        hidden_states = self.mlp_skip(hidden_states, mlp_out)

        return hidden_states


class IDRTransformer(BaseTransformer):
    layer_type = IDRLayer

    def get_norm(self, config):
        return IDRInputNorm(config.hidden_size, config.layer_norm_eps)


    def get_hidden_states(self, input_ids: torch.LongTensor, position_ids: torch.LongTensor) -> torch.Tensor:
        out = super().get_hidden_states(input_ids, position_ids)
    
        return torch.cat([out, out], dim=-1)


class IDRLmModel(BaseLmModel):
    transformer_type = IDRTransformer
