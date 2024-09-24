import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.xla import XLAConfig, XLAModel
from utils.model_utils import (
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
            Number of hidden layers in the Transformers.
        num_decoder_layers (`int`):
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
        z_over_scale (`float`):
            The scale of the initial latent space.
    """

    model_type = 'swift'

    def __init__(
        self,
        hidden_size=None,
        mlp_size=None,
        attention_head_size=None,
        num_attention_heads=None,
        num_registers=None,
        num_layers=None,
        num_decoder_layers=None,
        hidden_act=None,
        norm_eps=None,
        rope_fraction=None,
        rope_base=None,
        z_size=None,
        z_over_scale=None,
        *args,
        **kwargs,
    ):

        self.hidden_size = hidden_size
        self.mlp_size = mlp_size

        self.attention_head_size = attention_head_size
        self.num_attention_heads = num_attention_heads
        self.num_registers = num_registers

        self.num_layers = num_layers
        self.num_decoder_layers = num_decoder_layers

        self.hidden_act = hidden_act
        self.norm_eps = norm_eps
        
        self.rope_fraction = rope_fraction
        self.rope_base = rope_base

        self.z_size = z_size
        self.z_over_scale = z_over_scale

        super().__init__(*args, **kwargs)


class SwiftLayer(nn.Module):

    def special_init(self, config: SwiftConfig):

        self.enc_scale.weight.data.zero_()
        self.dec_scale.weight.data.zero_()
        self.enc_bias.weight.data.zero_()
        self.dec_bias.weight.data.zero_()
        self.enc_filter.weight.data.zero_()
        self.dec_filter.weight.data.zero_()

        self.cross.linear.weight.data.zero_()
        if self.cross.linear.bias is not None:
            self.cross.linear.bias.data.zero_()

        self.enc_scale.special_inited = True
        self.dec_scale.special_inited = True
        self.enc_bias.special_inited = True
        self.dec_bias.special_inited = True
        self.enc_filter.special_inited = True
        self.dec_filter.special_inited = True

        self.cross.linear.special_inited = True


    def __init__(self, config: SwiftConfig, layer_idx: int):
        super().__init__()

        # basic shapes
        self.hidden_size = config.hidden_size
        self.z_size = config.z_size
        self.qkv_size = config.attention_head_size * config.num_attention_heads

        # norms
        self.enc_norm = nn.LayerNorm(config.hidden_size, config.norm_eps, elementwise_affine=False)
        self.dec_norm = nn.LayerNorm(config.hidden_size, config.norm_eps, elementwise_affine=False)
        
        # in parameters
        self.enc_scale = nn.Embedding(2, config.hidden_size)
        self.dec_scale = nn.Embedding(2, config.hidden_size)
        self.enc_bias = nn.Embedding(2, config.hidden_size)
        self.dec_bias = nn.Embedding(2, config.hidden_size)

        # cross parameters
        self.cross = FusedLinear(
            config.hidden_size,
            [config.hidden_size]*2,
            bias=False
        )

        # input projections
        self.enc_up = FusedLinear(
            config.hidden_size,
            [3*self.qkv_size] + [config.mlp_size]*2 + [config.z_size]*2,
            bias=False
        )
        self.dec_up = FusedLinear(
            config.hidden_size,
            [3*self.qkv_size] + [config.mlp_size]*2 + [config.z_size],
            bias=False
        )

        self.attention = RotaryAttention(
            config.hidden_size,
            config.attention_head_size,
            config.num_attention_heads,
            config.num_registers,
            True,
            config.rope_fraction,
            config.max_sequence_length,
            config.rope_base,
            layer_idx
        )
        self.mlp = GLU(
            config.hidden_act
        )

        self.enc_down = FusedLinear(
            [self.qkv_size, config.mlp_size, self.z_size],
            config.hidden_size,
            bias=False
        )
        self.dec_down = FusedLinear(
            [self.qkv_size, config.mlp_size, self.z_size],
            config.hidden_size,
            bias=False
        )

        # output filters
        self.enc_filter = nn.Embedding(2, config.hidden_size)
        self.dec_filter = nn.Embedding(2, config.hidden_size)

        # z scale
        self.z_scale = np.sqrt(config.z_over_scale / (config.z_size * config.num_layers))


    def forward(
        self,
        encoder_states,
        decoder_states,
        mask,
        noise,
        position_ids=None,
        attention_mask=None,
        past_key_value=None,
    ):
        float_mask = mask.to(encoder_states.dtype).unsqueeze(-1)

        # apply conditional cross scaling
        enc_normed = self.enc_norm(encoder_states)
        dec_normed = self.dec_norm(decoder_states)

        enc_x = self.enc_bias(mask) + (1+self.enc_scale(mask)) * enc_normed
        dec_x = self.dec_bias(mask) + (1+self.dec_scale(mask)) * dec_normed

        enc_x = enc_x + torch.where(
            mask.bool().unsqueeze(-1),
            *self.cross(dec_x)
        )

        # get outputs
        enc_qkv, enc_gate, enc_val, enc_mu, enc_log_sigma = self.enc_up(enc_x)
        dec_qkv, dec_gate, dec_val, dec_mu = self.dec_up(dec_x)

        qkv = torch.cat([enc_qkv, dec_qkv], dim=0)
        enc_attn_out, dec_attn_out = self.attention(
            *qkv.chunk(3, dim=-1),
            position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value
        ).chunk(2, dim=0)

        enc_mlp_out = self.mlp(enc_gate, enc_val)
        dec_mlp_out = self.mlp(dec_gate, dec_val)

        enc_mu = enc_mu * self.z_scale * float_mask
        enc_log_sigma = enc_log_sigma * self.z_scale * float_mask
        dec_mu = dec_mu * self.z_scale * float_mask

        enc_sigma = F.softplus(enc_log_sigma + np.log(np.e - 1))

        z = (enc_mu + enc_sigma * noise) * float_mask

        encoder_states = encoder_states + self.enc_filter(mask) * self.enc_down(enc_attn_out, enc_mlp_out, z)
        decoder_states = decoder_states + self.dec_filter(mask) * self.dec_down(dec_attn_out, dec_mlp_out, z)

        kl = (
            -torch.log(enc_sigma)
            + 0.5 * (enc_sigma**2 + (enc_mu - dec_mu)**2)
            - 0.5
        ).sum(-1).sum(-1)

        return encoder_states, decoder_states, kl
        

    def sample(
        self,
        decoder_states,
        mask,
        noise,
        position_ids=None,
        attention_mask=None,
        past_key_value=None,
    ):
        float_mask = mask.to(decoder_states.dtype).unsqueeze(-1)

        dec_normed = self.dec_norm(decoder_states)
        dec_x = self.dec_bias(mask) + (1+self.dec_scale(mask)) * dec_normed

        dec_qkv, dec_gate, dec_val, dec_mu = self.dec_up(dec_x)

        dec_attn_out = self.attention(
            *dec_qkv.chunk(3, dim=-1),
            position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value
        )
        dec_mlp_out = self.mlp(dec_gate, dec_val)

        z = float_mask * (noise + dec_mu)

        decoder_states = decoder_states + self.dec_filter(mask) * self.dec_down(dec_attn_out, dec_mlp_out, z)

        return decoder_states


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

        self.hidden_size = config.hidden_size
        self.z_size = config.z_size
        self.num_layers = config.num_layers

        self.enc_embs = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dec_embs = nn.Embedding(config.vocab_size+1, config.hidden_size)

        # layers
        self.layers = nn.ModuleList([
            SwiftLayer(config, i)
            for i in range(config.num_layers)
        ])

        # outputs
        self.norm = nn.LayerNorm(config.hidden_size, config.norm_eps, elementwise_affine=True)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids,
        mask
    ):
        bs, seq_len = input_ids.shape

        noise = torch.randn(
            [bs, seq_len, self.num_layers, self.z_size],
            device=input_ids.device, dtype=self.enc_embs.weight.dtype
        )
        long_mask = mask.long()

        encoder_states = self.enc_embs(input_ids)
        decoder_states = torch.where(
            mask.unsqueeze(-1),
            self.dec_embs(torch.zeros_like(input_ids)),
            self.dec_embs(input_ids+1)
        )

        kl = 0
        for i, layer in enumerate(self.layers):

            encoder_states, decoder_states, kl_out = layer(
                encoder_states,
                decoder_states,
                long_mask,
                noise[:, :, i],
            )

            kl = kl + kl_out

        lm_logits = self.lm_head(self.norm(decoder_states))
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        return lm_logits, kl


    def sample(
        self,
        input_ids,
        mask,
        noise=None
    ):
        bs, seq_len = input_ids.shape

        if noise is None:
            noise = torch.randn(
                [bs, seq_len, self.num_layers, self.z_size],
                device=input_ids.device, dtype=self.enc_embs.weight.dtype
            )
        long_mask = mask.long()

        decoder_states = torch.where(
            mask.unsqueeze(-1),
            self.dec_embs(torch.zeros_like(input_ids)),
            self.dec_embs(input_ids+1)
        )

        for i, layer in enumerate(self.layers):

            decoder_states = layer.sample(
                decoder_states,
                long_mask,
                noise[:, :, i],
            )

        lm_logits = self.lm_head(self.norm(decoder_states))
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        return lm_logits
    