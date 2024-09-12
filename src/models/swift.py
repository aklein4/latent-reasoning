import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.xla import XLAConfig, XLAModel
from utils.model_utils import (
    RMSNorm,
    FusedLinear,
    RotaryAttention,
    GLU
)


class SwiftConfig(XLAConfig):
    """
    Args:
        hidden_size (`int`):
            Number of hidden layers in the Transformer decoder.
        mlp_size (`int`):
            Dimension of the MLP representations.
        attention_head_size (`int`):
            Size of the attention heads in the Transformer encoder
        num_attention_heads (`int`):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_registers (`int`):
            Number of registers to use in the attention layer.
        num_layers (`int`):
            Number of hidden layers in the Transformer decoder.
        hidden_act (`str` or `function`):
            The non-linear activation function (function or string).
        norm_eps (`float`):
            The epsilon used by the normalization layers.
        rope_fraction (`int`):
            The fraction of the hidden size to use for the RoPE embeddings.
        rope_base (`float`):
            The base period of the RoPE embeddings.
        z_size (`int`):
            The size of the latent space.
        z_init_scale (`float`):
            The scale of the initial latent space.
    """

    model_type = 'base'

    def __init__(
        self,
        hidden_size=None,
        mlp_size=None,
        attention_head_size=None,
        num_attention_heads=None,
        num_registers=None,
        num_layers=None,
        hidden_act=None,
        norm_eps=None,
        rope_fraction=None,
        rope_base=None,
        z_size=None,
        z_init_scale=None,
        *args,
        **kwargs,
    ):

        self.hidden_size = hidden_size
        self.mlp_size = mlp_size

        self.attention_head_size = attention_head_size
        self.num_attention_heads = num_attention_heads
        self.num_registers = num_registers

        self.num_layers = num_layers

        self.hidden_act = hidden_act
        self.norm_eps = norm_eps
        
        self.rope_fraction = rope_fraction
        self.rope_base = rope_base

        self.z_size = z_size
        self.z_init_scale = z_init_scale

        # derived
        assert self.z_size % self.num_layers == 0
        self.layer_z_size = self.z_size // self.num_layers
        self.qkv_size = self.attention_head_size * self.num_attention_heads

        super().__init__(*args, **kwargs)


class BaseSwiftLayer(nn.Module):

    def special_init(self, config: SwiftConfig):

        self.attn_scale.weight.data.zero_()
        self.qkv_scale.weight.data.zero_()
        self.qkv_bias.weight.data.zero_()
        self.attn_filter.weight.data.zero_()

        self.mlp_scale.weight.data.zero_()
        self.mlp_bias.weight.data.zero_()
        self.mlp_filter.weight.data.zero_()

        self.attn_scale.special_inited = True
        self.qkv_scale.special_inited = True
        self.qkv_bias.special_inited = True
        self.attn_filter.special_inited = True

        self.mlp_scale.special_inited = True
        self.mlp_bias.special_inited = True
        self.mlp_filter.special_inited = True


    def __init__(self, config: SwiftConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.mlp_size = config.mlp_size
        self.qkv_size = config.qkv_size

        self.attn_norm = RMSNorm(self.hidden_size, eps=config.norm_eps, affine=False)
        self.attn_scale = nn.Embedding(3, self.hidden_size)
        self.qkv_scale = nn.Embedding(3, 3*self.qkv_size)
        self.qkv_bias = nn.Embedding(3, 3*self.qkv_size)
        self.attn_filter = nn.Embedding(3, self.hidden_size)

        self.QKV = FusedLinear(
            self.hidden_size,
            [self.qkv_size]*3,
            bias=False
        )
        self.O = nn.Linear(self.qkv_size, self.hidden_size, bias=False)

        self.mlp_norm = nn.LayerNorm(self.hidden_size, eps=config.norm_eps, elementwise_affine=False)
        self.mlp_scale = nn.Embedding(3, self.hidden_size)
        self.mlp_bias = nn.Embedding(3, self.hidden_size)
        self.mlp_filter = nn.Embedding(3, self.hidden_size)

        self.mlp_gate = nn.Linear(self.hidden_size, self.mlp_size, bias=False)
        self.mlp_up = nn.Linear(self.hidden_size, self.mlp_size, bias=False)
        self.mlp_down = nn.Linear(self.mlp_size, self.hidden_size, bias=False)

        self.attn = RotaryAttention(
            self.hidden_size,
            config.attention_head_size,
            config.num_attention_heads,
            None,
            True,
            config.rope_fraction,
            config.max_sequence_length,
            config.rope_base,
            self.layer_idx
        )
        self.mlp = GLU(config.hidden_act)


    def forward(
        self,
        hidden_states,
        state_types,
        position_ids: torch.LongTensor,
        attention_mask=None,
        past_key_value=None,
    ):

        # attention
        q, k, v = self.QKV(
            self.attn_norm(hidden_states),
            in_scale=(1+self.attn_scale(state_types)),
            scale=(1+self.qkv_scale(state_types)),
            bias=self.qkv_bias(state_types)
        )

        attn_out = self.attn(
            q, k, v,
            position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value
        )
        y = self.O(attn_out) * self.attn_filter(state_types)

        hidden_states = hidden_states + y

        # mlp
        x = self.mlp_norm(hidden_states) * (1+self.mlp_scale(state_types)) + self.mlp_bias(state_types)
        
        gate = self.mlp_gate(x)
        up = self.mlp_up(x)
        
        mlp_out = self.mlp(gate, up)
        y = self.mlp_down(mlp_out)

        hidden_states = hidden_states + y

        return hidden_states


class EncoderSwiftLayer(BaseSwiftLayer):


    def __init__(self, config: SwiftConfig, layer_idx: int):
        super().__init__()

        self.layer_z_size = config.layer_z_size

        self.z_norm = RMSNorm(self.hidden_size, eps=config.norm_eps, affine=True)
        self.z_filter = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

        self.z_up = FusedLinear(
            self.hidden_size,
            [self.layer_z_size]*2,
            bias=False
        )
        self.z_down = nn.Linear(self.layer_z_size, self.hidden_size, bias=False)

        self.z_scale = np.sqrt(config.z_init_scale / config.num_layers)


    def forward(
        self,
        hidden_states,
        state_types,
        noise,
        position_ids: torch.LongTensor,
        attention_mask=None,
        past_key_value=None,
    ):
        hidden_states = super().forward(
            hidden_states, state_types, position_ids, attention_mask, past_key_value
        )

        mu, log_sigma = self.z_up(
            self.z_norm(hidden_states),
            scale=self.z_scale
        ).chunk(2, dim=-1)
        sigma = F.softplus(log_sigma + np.log(np.e - 1))

        z = mu + sigma * noise
        hidden_states = hidden_states + self.z_down(z) * self.z_filter

        return hidden_states, mu, sigma, z


    def update_stream(self, hidden_states, y):
        if self.disable_filter:
            return y + hidden_states

        return (
            self.stream_filter * y +
            (1 - self.stream_filter) * hidden_states
        )


class SwiftModel(XLAModel):

    config_class = SwiftConfig


    def _init_weights(self, module):

        if hasattr(module, 'special_inited') and module.special_inited:
            return
        
        if hasattr(module, 'special_init'):
            module.special_init(self.config)

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(0.0, 1/np.sqrt(module.weight.shape[1]))
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(0.0, 1.0)


    def __init__(self, config: SwiftConfig, fast_start=False):
        super().__init__(config, fast_start=fast_start)

        # info
        self.vocab_size = config.vocab_size
        self.max_sequence_length = config.max_sequence_length

        # embeddings
        self.vocab_embs = nn.Embedding(config.vocab_size+1, config.hidden_size)
        self.encoder_embs = nn.Embedding(2, config.hidden_size)
        self.decoder_embs = nn.Embedding(2, config.hidden_size)

        # positional embeddings
        assert config.use_rope is not None
        self.use_rope = config.use_rope
        if self.use_rope:
            self.pos_embs = None
        else:
            self.pos_embs = nn.Embedding(config.max_sequence_length, config.hidden_size)
        
        # layers
        self.num_layers = config.num_layers
        self.layers = nn.ModuleList(
            [SwiftLayer(config, layer_idx) for layer_idx in range(config.num_layers)]
        )

        # lm modeling
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def _get_position_ids(self, input_ids: torch.LongTensor):
        return torch.arange(input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device)[None]


    def get_encoder_states(self, input_ids, position_ids, mask):
        input_ids = input_ids + 1
        mask = mask.long()

        encoder_states = self.vocab_embs(input_ids)
        if self.use_rope:
            encoder_states = encoder_states/2 + self.encoder_embs(mask)/2
        else:
            encoder_states = encoder_states/3 + self.encoder_embs(mask)/3 + self.pos_embs(position_ids)/3

        return encoder_states


    def get_decoder_states(self, input_ids, position_ids, mask):
        input_ids = torch.where(mask, torch.zeros_like(input_ids), input_ids+1)
        mask = mask.long()

        decoder_states = self.vocab_embs(input_ids)
        if self.use_rope:
            decoder_states = decoder_states/2 + self.decoder_embs(mask)/2
        else:
            decoder_states = decoder_states/3 + self.decoder_embs(mask)/3 + self.pos_embs(position_ids)/3

        return decoder_states


    def get_hidden_states(self, input_ids, position_ids, mask):

        encoder_states = self.get_encoder_states(input_ids, position_ids, mask)
        decoder_states = self.get_decoder_states(input_ids, position_ids, mask)

        return torch.cat([encoder_states, decoder_states], dim=0)


    def forward(
        self,
        input_ids,
        mask
    ):
        bs, seq_len = input_ids.shape

        position_ids = self._get_position_ids(input_ids)
        hidden_states = self.get_hidden_states(input_ids, position_ids, mask)

        z = torch.randn(
            [self.num_layers, bs, seq_len, self.config.z_size],
            device=input_ids.device, dtype=hidden_states.dtype
        )

        mus = []
        sigmas = []
        for idx, layer in enumerate(self.layers):
            
            hidden_states, m, s = layer(
                hidden_states, z[idx], position_ids
            )
            mus.append(m)
            sigmas.append(s)
        
        mus = torch.stack(mus, dim=0)
        sigmas = torch.stack(sigmas, dim=0)

        decoder_states = hidden_states.chunk(2, dim=0)[1]
        lm_logits = self.lm_head(self.norm(decoder_states))
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        return lm_logits, mus, sigmas


    def sample(
        self,
        input_ids,
        mask,
        z=None
    ):
        bs, seq_len = input_ids.shape

        position_ids = self._get_position_ids(input_ids)
        decoder_states = self.get_decoder_states(input_ids, position_ids, mask)

        if z is None:
            z = torch.randn(
                [self.num_layers, bs, seq_len, self.config.z_size],
                device=input_ids.device, dtype=decoder_states.dtype
            )

        for idx, layer in enumerate(self.layers):
            decoder_states = layer.sample(
                decoder_states, z[idx], position_ids
            )

        lm_logits = self.lm_head(self.norm(decoder_states))

        return torch.argmax(lm_logits, dim=-1)
