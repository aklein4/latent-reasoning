from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import (
    BaseConfig, BaseTransformer, BaseLmModel,
    BaseAttention, BaseMLP
)
import numpy as np


class ASMConfig(BaseConfig):

    model_type = 'asm'

    def __init__(
        self,
        k_rank: int = 1,
        *args,
        **kwargs,
    ):
        
        self.k_rank = k_rank

        super().__init__(*args, **kwargs)


class ASMWriter(nn.Module):

    def special_init_weights(self, config):

        self.k_up.weight.data.zero_()
        self.k_up.bias.data.fill_(np.log(np.exp(1) - 1))


    def __init__(self, config, y_size, is_embedding=False):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.y_size = y_size
        self.k_rank = config.k_rank
        self.is_embedding = is_embedding

        if is_embedding:
            self.W = nn.Linear(self.y_size, config.k_rank, bias=False)
        else:
            self.W = nn.Linear(self.y_size, self.hidden_size + config.k_rank, bias=False)    
        
        self.k_up = nn.Linear(self.k_rank, self.hidden_size, bias=True)


    def forward(self, hidden_states, y):
        hidden_states, normalizer = hidden_states.chunk(2, dim=0)

        # calculate output and down_y in same operation
        if self.is_embedding:
            y_out, k_down = y, self.W(y)
        else:
            y_out, k_down = self.W(y).split([self.hidden_size, self.k_rank], dim=-1)
        
        k = F.softplus(self.k_up(k_down))

        hidden_states = hidden_states + k * y_out
        normalizer = normalizer + k

        return torch.cat([hidden_states, normalizer], dim=0)


class ASMReader(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


    def forward(self, hidden_states):
        hidden_states, normalizer = hidden_states.chunk(2, dim=0)

        hidden_states = hidden_states / (1 + normalizer)
        hidden_states = self.norm(hidden_states)

        return hidden_states.squeeze(0)


class ASMAttention(BaseAttention):

    def init_o_proj(self, config):
        pass

    def get_o(self, hidden_states):
        return hidden_states


class ASMMLP(BaseMLP):

    def init_mlp_output(self, config):
        pass

    def get_mlp_output(self, hidden_states):
        return hidden_states


class ASMLayer(nn.Module):

    def special_init_weights(self, config):
        if config.identity_init:
            raise ValueError("Identity init not supported for SSMConnection!")

        self.attn_writer.special_init_weights(config)
        self.mlp_writer.special_init_weights(config)


    def post_step(self):
        pass


    def __init__(self, config, layer_idx: int):
        super().__init__()

        self.hidden_size = config.hidden_size

        self.attn = ASMAttention(config, layer_idx)
        self.mlp = ASMMLP(config)

        self.attn_reader = ASMReader(config)
        self.mlp_reader = ASMReader(config)

        self.attn_writer = ASMWriter(config, config.hidden_size)
        self.mlp_writer = ASMWriter(config, config.mlp_size)


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value = None,
    ):

        # Self Attention
        attn_out = self.attn(
            self.attn_reader(hidden_states),
            position_ids,
            attention_mask,
            past_key_value=past_key_value
        )
        hidden_states = self.attn_writer(hidden_states, attn_out)

        # GLU MLP
        mlp_out = self.mlp(
            self.mlp_reader(hidden_states)
        )
        hidden_states = self.mlp_writer(hidden_states, mlp_out)

        return hidden_states


class ASMTransformer(BaseTransformer):

    layer_type = ASMLayer


    def special_init_weights(self, config: BaseConfig):
        super().special_init_weights(config)

        self.emb_writer.special_init_weights(config)


    def get_extras(self, config):
        self.emb_writer = ASMWriter(config, config.hidden_size, is_embedding=True)
        self.norm = ASMReader(config)


    def get_hidden_states(self, input_ids: torch.LongTensor, position_ids: torch.LongTensor) -> torch.Tensor:
        hidden_states = super().get_hidden_states(input_ids, position_ids)
        return self.emb_writer(
            torch.zeros(2, *hidden_states.shape, device=hidden_states.device, dtype=hidden_states.dtype),
            hidden_states
        )


class ASMLmModel(BaseLmModel):

    transformer_type = ASMTransformer
