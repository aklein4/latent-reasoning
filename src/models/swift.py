import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.xla import XLAConfig, XLAModel
from utils.model_utils import (
    RMSNorm,
    FusedLinear,
    FullRotaryAttention,
    FullGLU
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


class SwiftBaseLayer(nn.Module):

    def __init__(self, config: SwiftConfig, layer_idx: int):
        super().__init__()

        # basic shapes
        self.hidden_size = config.hidden_size
        
        # input norms
        self.attn_norm = RMSNorm(config.hidden_size, config.norm_eps, affine=True)
        self.mlp_norm = RMSNorm(config.hidden_size, config.norm_eps, affine=True)

        # output filters
        self.attn_filter = nn.Parameter(torch.zeros([1, 1, config.hidden_size]))
        self.mlp_filter = nn.Parameter(torch.zeros([1, 1, config.hidden_size]))

        # operations
        self.attn = FullRotaryAttention(
            self.hidden_size,
            config.attention_head_size,
            config.num_attention_heads,
            config.num_registers,
            True,
            config.rope_fraction,
            config.max_sequence_length,
            config.rope_base,
            layer_idx
        )
        self.mlp = FullGLU(
            self.hidden_size,
            config.mlp_size,
            config.hidden_act,
        )


    def forward(
        self,
        hidden_states,
        position_ids=None,
        attention_mask=None,
        past_key_value=None,
    ):

        # attention
        attn_out = self.attn(
            self.attn_norm(hidden_states),
            position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value
        )
        hidden_states = hidden_states + self.attn_filter * attn_out

        # mlp
        mlp_out = self.mlp(self.mlp_norm(hidden_states))
        hidden_states = hidden_states + self.mlp_filter * mlp_out

        return hidden_states
    

class SwiftLayer(nn.Module):

    def __init__(self, config: SwiftConfig, layer_idx: int):
        super().__init__()

        # basic shapes
        self.hidden_size = config.hidden_size
        self.z_size = config.z_size

        # base layers
        self.enc_base = SwiftBaseLayer(config, layer_idx)
        self.dec_base = SwiftBaseLayer(config, layer_idx)

        # norms
        self.enc_norm = RMSNorm(config.hidden_size, config.norm_eps, affine=True)
        self.dec_norm = RMSNorm(config.hidden_size, config.norm_eps, affine=True)

        # input projections
        self.enc_up = FusedLinear(
            config.hidden_size,
            [config.z_size, config.z_size],
            bias=False
        )
        self.dec_up = FusedLinear(
            config.hidden_size,
            [config.z_size, config.z_size],
            bias=False
        )

        # output projections
        self.enc_down = nn.Linear(config.z_size, config.hidden_size, bias=False)
        self.dec_down = nn.Linear(config.z_size, config.hidden_size, bias=False)

        # output filters
        self.enc_filter = nn.Parameter(torch.zeros([1, 1, config.hidden_size]))
        self.dec_filter = nn.Parameter(torch.zeros([1, 1, config.hidden_size]))

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
        encoder_states = self.enc_base(
            encoder_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
        )
        decoder_states = self.dec_base(
            decoder_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
        )

        enc_mu, enc_log_sigma = self.enc_up(self.enc_norm(encoder_states))
        dec_mu, dec_log_sigma = self.dec_up(self.dec_norm(decoder_states))

        enc_mu = enc_mu * self.z_scale * mask.unsqueeze(-1)
        enc_log_sigma = enc_log_sigma * self.z_scale * mask.unsqueeze(-1)

        dec_mu = dec_mu * self.z_scale * mask.unsqueeze(-1)
        dec_log_sigma = dec_log_sigma * self.z_scale * mask.unsqueeze(-1)

        enc_sigma = F.softplus(enc_log_sigma + np.log(np.e - 1))
        dec_sigma = F.softplus(dec_log_sigma + np.log(np.e - 1))

        z = (enc_mu + enc_sigma * noise) * mask.unsqueeze(-1)

        encoder_states = encoder_states + self.enc_filter * self.enc_down(z)
        decoder_states = decoder_states + self.dec_filter * self.dec_down(z)

        kl = (
            torch.log(dec_sigma) - torch.log(enc_sigma)
            + (enc_sigma**2 + (enc_mu - dec_mu)**2) / (2 * (dec_sigma**2))
            - 0.5
        ).sum(-1).sum(-1)

        return encoder_states, decoder_states, kl
        

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

        self.vocab_embs = nn.Embedding(config.vocab_size+1, config.hidden_size)

        # switches
        self.enc_x_switch = nn.Parameter(torch.zeros([1, 1, config.hidden_size]))
        self.enc_y_switch = nn.Parameter(torch.zeros([1, 1, config.hidden_size]))
        self.dec_x_switch = nn.Parameter(torch.zeros([1, 1, config.hidden_size]))

        # layers
        self.layers = nn.ModuleList([
            SwiftLayer(config, i)
            for i in range(config.num_layers)
        ])

        # outputs
        self.norm = RMSNorm(config.hidden_size, config.norm_eps, affine=True)
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
            device=input_ids.device, dtype=self.enc_x_switch.dtype
        )
        base_states = self.vocab_embs(input_ids + 1)
        float_mask = mask.float()

        encoder_states = torch.where(
            mask.unsqueeze(-1),
            base_states + self.enc_y_switch,
            base_states + self.enc_x_switch,
        )
        decoder_states = torch.where(
            mask.unsqueeze(-1),
            self.vocab_embs(torch.zeros_like(input_ids)),
            base_states + self.dec_x_switch
        )

        kl = 0
        for i, layer in enumerate(self.layers):

            encoder_states, decoder_states, kl_out = layer(
                encoder_states,
                decoder_states,
                float_mask,
                noise[:, :, i],
            )

            kl = kl + kl_out

        lm_logits = self.lm_head(self.norm(decoder_states))
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        return lm_logits, kl
