from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import (
    BaseConfig, BaseTransformer, BaseLmModel,
    BaseAttention, BaseMLP
)


class RSTConfig(BaseConfig):

    model_type = 'RST'

    def __init__(
        self,
        delta_rank: int = 1,
        ssm_epsilon: float = 1e-5,
        *args,
        **kwargs,
    ):
        
        self.delta_rank = delta_rank
        self.ssm_epsilon = ssm_epsilon

        super().__init__(*args, **kwargs)


class SplitNorm(nn.GroupNorm):
    def __init__(self, c, eps=1e-5, affine=True):
        super().__init__(2, c, eps, affine)

    def forward(self, x):
        bs, l, d = x.shape
        out = super().forward(x.view(bs*l, d))
        return out.view(bs, l, d)


class SSMConnection(nn.Module):

    def special_init_weights(self, config):

        # TODO: is this is best way?
        self.A.data.normal_(0.0, config.initializer_range)

        self.delta_up.weight.data.zero_()

        # TODO: this is hardcoded from MAMBA
        t = 0.001 + (
            torch.rand_like(self.delta_up.bias.data) * (0.1 - 0.001)
        )

        # residual gets t=1
        t[..., self.half_size:].fill_(1.0)

        self.delta_up.bias.data = torch.log(t.exp() - 1)


    def __init__(self, config, y_size):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.half_size = self.hidden_size // 2
        assert self.half_size * 2 == self.hidden_size, f"hidden_size {self.hidden_size} must be even!"

        self.y_size = y_size
        self.ssm_epsilon = config.ssm_epsilon

        # decay factor
        self.A = nn.Parameter(torch.zeros(1, 1, self.half_size))

        # project y into the stream
        self.W = nn.Linear(self.y_size, self.hidden_size + config.delta_rank, bias=False)
        
        # calculate delta from y and hidden states
        self.delta_norm = SplitNorm(self.hidden_size, config.layer_norm_eps)
        self.delta_down_h = nn.Linear(self.hidden_size, config.delta_rank, bias=False)
        
        self.delta_up = nn.Linear(config.delta_rank, self.hidden_size, bias=True)


    def forward(self, hidden_states, y):

        # calculate output and down_y in same operation
        y_w = self.W(y)
        y_out = y_w[:, :, :self.hidden_size]
        down_y = y_w[:, :, self.hidden_size:]

        # get low rank delta from hidden states
        down_h = self.delta_down_h(self.delta_norm(hidden_states))

        # get delta
        delta = self.delta_up(down_h + down_y)
        delta = F.softplus(delta)

        # calculate SSM matrices
        A_neg = -F.softplus(self.A)
        A_bar = torch.exp(delta[:, :, :self.half_size] * A_neg)
        B_bar = (A_bar - 1) / (A_neg - self.ssm_epsilon)

        # add residuals
        A_bar = torch.cat([A_bar, torch.ones_like(A_bar)], dim=-1)
        B_bar = torch.cat([B_bar, delta[:, :, self.half_size:]], dim=-1)

        return (
            A_bar * hidden_states +
            B_bar * y_out
        )


class RSTAttention(BaseAttention):

    def init_o_proj(self, config):
        pass

    def get_o(self, hidden_states):
        return hidden_states


class RSTMLP(BaseMLP):

    def init_mlp_output(self, config):
        pass

    def get_mlp_output(self, hidden_states):
        return hidden_states


class RSTLayer(nn.Module):

    def special_init_weights(self, config):
        if config.identity_init:
            raise ValueError("Identity init not supported for SSMConnection!")

        self.attn_connection.special_init_weights(config)
        self.mlp_connection.special_init_weights(config)

    def post_step(self):
        pass


    def __init__(self, config, layer_idx: int):
        super().__init__()

        self.hidden_size = config.hidden_size

        self.attn = RSTAttention(config, layer_idx)
        self.mlp = RSTMLP(config)

        h = config.hidden_size
        eps = config.layer_norm_eps
        self.attn_norm = SplitNorm(h, eps)
        self.mlp_norm = SplitNorm(h, eps)

        self.attn_connection = SSMConnection(config, self.hidden_size)
        self.mlp_connection = SSMConnection(config, config.mlp_size)


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value = None,
    ):

        # Self Attention
        attn_out = self.attn(
            self.attn_norm(hidden_states),
            position_ids,
            attention_mask,
            past_key_value=past_key_value
        )
        hidden_states = self.attn_connection(hidden_states, attn_out)

        # GLU MLP
        mlp_out = self.mlp(
            self.mlp_norm(hidden_states)
        )
        hidden_states = self.mlp_connection(hidden_states, mlp_out)

        return hidden_states


class RSTTransformer(BaseTransformer):

    layer_type = RSTLayer


    def special_init_weights(self, config: BaseConfig):
        super().special_init_weights(config)

        self.vocab_embs.weight.data[:, :config.hidden_size//2].zero_()

        if self.pos_embs is not None:
            self.pos_embs.weight.data[:, :config.hidden_size//2].zero_()


    def get_extras(self, config):
        self.norm = SplitNorm(config.hidden_size, config.layer_norm_eps)


class RSTLmModel(BaseLmModel):

    transformer_type = RSTTransformer
