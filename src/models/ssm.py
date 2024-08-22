from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.base import (
    BaseConfig, BaseTransformer, BaseLmModel,
    BaseAttention, BaseMLP
)
from utils.model_utils import fast_checkpoint


class RatConfig(BaseConfig):

    model_type = 'rat'

    def __init__(
        self,
        residual_channels: int = 4,
        residual_heads: int = 16,
        delta_rank: int = 8,
        ssm_eps: float = 1e-5,
        *args,
        **kwargs,
    ):
        
        self.residual_channels = residual_channels
        self.residual_heads = residual_heads
        self.delta_rank = delta_rank
        self.ssm_eps = ssm_eps

        super().__init__(*args, **kwargs)


class RatRead(nn.Module):


    def special_init_weights(self, config):
        self.conv.weight.data.normal_(
            0.0, 1 / np.sqrt(config.residual_channels)
        )


    def __init__(self, config: RatConfig, num_outputs):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.residual_channels = config.residual_channels
        self.residual_size = self.hidden_size * self.residual_channels
        
        self.residual_heads = config.residual_heads
        assert self.hidden_size % self.residual_heads == 0

        self.num_outputs = num_outputs
        self.output_size = self.hidden_size * self.num_outputs

        self.conv = nn.Conv1d(
            self.residual_size, self.output_size,
            kernel_size=1, bias=False,
            groups=self.hidden_size
        )

        self.norm = nn.GroupNorm(
            self.num_outputs,
            self.output_size,
            eps=config.layer_norm_eps
        )


    def compute(self, hidden_states):
        bs, l, _ = hidden_states.shape

        # [B, L, HN]
        hidden_states = hidden_states.view(bs*l, self.residual_size, 1)
        hidden_states = self.conv(hidden_states)
        hidden_states = hidden_states.view(bs, l, self.output_size)

        # [B, L, N, H]
        hidden_states = (
            hidden_states
            .view(bs, l, self.hidden_size, self.num_outputs)
            .permute(0, 1, 3, 2)
            .reshape(bs, l, self.output_size)
        )

        hidden_states = hidden_states.view(bs*l, self.output_size)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(bs, l, self.output_size)
    
        return hidden_states
    

    def forward(self, hidden_states):
        return fast_checkpoint(self.compute, hidden_states)


class RatWrite(nn.Module):


    def special_init_weights(self, config):
        # o_d are both fine, no bias

        # zero on up rank
        self.delta_up.weight.data.zero_()

        # TODO: this is hardcoded from MAMBA
        t = 0.001 + (
            torch.rand_like(self.delta_up.bias.data) * (0.1 - 0.001)
        )
        self.delta_up.bias.data = torch.log(t.exp() - 1)

        # TODO: is this is best way?
        self.WA_WB.weight.data.zero_()

        self.WA_WB.bias.data.zero_()
        self.WA_WB.bias.data[:self.A_size].normal_(
            0.0, config.initializer_range
        )
        self.WA_WB.bias.data[self.A_size:].normal_(
            0.0, 1 / np.sqrt(config.residual_channels)
        )
        

    def __init__(self, config, y_size):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.residual_channels = config.residual_channels
        self.residual_heads = config.residual_heads
        self.residual_size = self.hidden_size * self.residual_channels

        self.head_size = self.hidden_size // self.residual_heads
        assert self.hidden_size % self.residual_heads == 0
        
        self.A_size = (self.residual_channels - 1) * self.residual_heads
        self.B_size = self.residual_channels * self.residual_heads

        self.y_size = y_size
        self.ssm_eps = config.ssm_eps
        self.delta_rank = config.delta_rank

        # projections for y
        self.Wo_Wd = nn.Linear(self.y_size, self.hidden_size + config.delta_rank, bias=False)
        self.WA_WB = nn.Linear(self.y_size, self.A_size + self.B_size, bias=True)
        
        # up projection for delta
        self.delta_up = nn.Linear(config.delta_rank, self.hidden_size, bias=True)


    def compute(self, hidden_states, o_d, A_B):
        bs, l, _ = hidden_states.shape

        # split o_d
        out, delta_low = torch.split(
            o_d, [self.hidden_size, self.delta_rank], dim=-1
        )

        # split A_B
        A_logit, B_raw = torch.split(
            A_B, [self.A_size, self.B_size], dim=-1
        )
        B_proj, B_skip = torch.split(
            B_raw, [self.A_size, self.residual_heads], dim=-1
        )

        # get delta
        delta = self.delta_up(delta_low)
        delta = F.softplus(delta)

        # reshape for SSM
        hidden_states = hidden_states.view(bs, l, self.head_size, self.residual_heads, self.residual_channels)
        out = out.view(bs, l, self.head_size, self.residual_heads, 1)
        
        delta = delta.view(bs, l, self.head_size, self.residual_heads, 1)
        
        A_logit = A_logit.view(bs, l, 1, self.residual_heads, self.residual_channels-1)
        B_proj = B_proj.view(bs, l, 1, self.residual_heads, self.residual_channels-1)
        B_skip = B_skip.view(bs, l, 1, self.residual_heads, 1)

        # calculate SSM matrices
        A_neg = -F.softplus(A_logit)
        A_bar = torch.exp(delta * A_neg)
        
        B_bar = (A_bar - 1) / (A_neg - self.ssm_eps) * B_proj
        B_bar_skip = delta * B_skip

        # add residuals
        A_bar = torch.cat([A_bar, torch.ones_like(A_bar[:, :, :, :, :1])], dim=-1)
        B_bar = torch.cat([B_bar, B_bar_skip], dim=-1)

        out = hidden_states * A_bar + out * B_bar

        return out.view(bs, l, self.residual_size)


    def forward(self, hidden_states, y):

        # do major projections
        o_d = self.Wo_Wd(y)
        A_B = self.WA_WB(y)

        return fast_checkpoint(self.compute, hidden_states, o_d, A_B)


class RatEmbedding(nn.Module):


    def special_init_weights(self, config):
        v = self.proj.weight.data.view(
            self.num_embeddings,
            1,
            self.residual_heads,
            self.residual_channels
        )

        v[..., :-1].zero_()
        v[..., -1].fill_(1.0)  


    def __init__(self, num_embeddings, hidden_size, residual_channels, residual_heads):
        super().__init__()

        self.num_embeddings = num_embeddings

        self.hidden_size = hidden_size
        self.residual_channels = residual_channels
        self.residual_heads = residual_heads
        self.residual_size = hidden_size * residual_channels

        self.head_size = hidden_size // residual_heads
        assert hidden_size % residual_heads == 0

        self.embedding = nn.Embedding(num_embeddings, hidden_size)
        self.proj = nn.Embedding(num_embeddings, residual_heads * residual_channels)
    

    def forward(self, input_ids):
        bs, l = input_ids.shape

        out = self.embedding(input_ids)
        proj = self.proj(input_ids)

        out = out.view(bs, l, self.head_size, self.residual_heads, 1)
        proj = proj.view(bs, l, 1, self.residual_heads, self.residual_channels)

        hidden_states = out * proj
        hidden_states = hidden_states.view(bs, l, self.residual_size)

        return hidden_states


class RatAttention(BaseAttention):

    def init_qkv_proj(self, config):
        self.qkv_proj = nn.Conv1d(
            3 * config.hidden_size, 3 * config.hidden_size,
            kernel_size=1, bias=config.use_qkv_bias,
            groups=3
        )
    

    def get_qkv(self, hidden_states):
        bs, l, _ = hidden_states.shape

        hidden_states = hidden_states.view(bs*l, 3 * self.hidden_size, 1)
        hidden_states = self.qkv_proj(hidden_states)
        hidden_states = hidden_states.view(bs, l, 3 * self.hidden_size)

        return hidden_states.chunk(3, dim=-1)


    def init_o_proj(self, config):
        pass
    

    def get_o(self, hidden_states):
        return hidden_states


class RatMLP(BaseMLP):

    def init_mlp_input(self, config):
        self.in_proj = nn.Conv1d(
            2*config.hidden_size, 2*config.mlp_size,
            kernel_size=1, bias=False,
            groups=2
        )

    
    def get_mlp_input(self, hidden_states):
        bs, l, _ = hidden_states.shape

        hidden_states = hidden_states.view(bs*l, 2 * self.hidden_size, 1)
        hidden_states = self.in_proj(hidden_states)
        hidden_states = hidden_states.view(bs, l, 2 * self.mlp_size)

        return hidden_states.chunk(2, dim=-1)


    def init_mlp_output(self, config):
        pass


    def get_mlp_output(self, hidden_states):
        return hidden_states


class RatLayer(nn.Module):


    def special_init_weights(self, config: BaseConfig):
        if config.identity_init:
            raise ValueError("identity_init not supported for RatLayer!")

        self.attn_read.special_init_weights(config)
        self.mlp_read.special_init_weights(config)

        self.attn_write.special_init_weights(config)
        self.mlp_write.special_init_weights(config)


    def post_step(self):
        pass


    def __init__(self, config: BaseConfig, layer_idx: int):
        super().__init__()

        self.hidden_size = config.hidden_size

        self.attn = RatAttention(config, layer_idx)
        self.mlp = RatMLP(config)

        self.attn_read = RatRead(config, 3)
        self.mlp_read = RatRead(config, 2)

        self.attn_write = RatWrite(config, self.hidden_size)
        self.mlp_write = RatWrite(config, config.mlp_size)


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value=None,
    ):

        # Self Attention
        attn_out = self.attn(
            self.attn_read(hidden_states),
            position_ids,
            attention_mask,
            past_key_value=past_key_value
        )
        hidden_states = self.attn_write(hidden_states, attn_out)

        # GLU MLP
        mlp_out = self.mlp(
            self.mlp_read(hidden_states)
        )
        hidden_states = self.mlp_write(hidden_states, mlp_out)

        return hidden_states


class RatTransformer(BaseTransformer):
    
    layer_type = RatLayer


    def get_extras(self, config):
        self.vocab_embs = RatEmbedding(
            config.vocab_size, config.hidden_size,
            config.residual_channels, config.residual_heads
        )

        self.norm = RatRead(config, 1)


    # @ torch.no_grad()()
    def special_init_weights(self, config):
        super().special_init_weights(config)

        self.vocab_embs.special_init_weights(config)
        self.norm.special_init_weights(config)


class RatLmModel(BaseLmModel):

    transformer_type = RatTransformer

    requires_barrier = True
    