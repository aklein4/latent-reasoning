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
import utils.constants as constants


class HLmConfig(XLAConfig):
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
        z_mlp_mult (`int`):
            The multiplier for the size of the IAF MLPs.
    """

    model_type = 'hlm'

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
        z_mlp_mult=None,
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
        self.z_mlp_mult = z_mlp_mult

        super().__init__(*args, **kwargs)


class UnconditionalIO(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_size, eps=1e-5, elementwise_affine=True)
        self.filter = nn.Parameter(torch.zeros(1, 1, hidden_size))


    def enter(self, x):
        return self.norm(x)
    
    def exit(self, hidden_states, y):
        return hidden_states + self.filter * y


class ConditionalIO(nn.Module):

    def special_init(self, config: HLmConfig):
            
        self.scale.weight.data.zero_()
        self.bias.weight.data.zero_()
        self.filter.weight.data.zero_()

        self.scale.special_inited = True
        self.bias.special_inited = True
        self.filter.special_inited = True


    def __init__(self, hidden_size):

        super().__init__()

        self.norm = nn.LayerNorm(hidden_size, eps=1e-5, elementwise_affine=False)

        self.scale = nn.Embedding(2, hidden_size)
        self.bias = nn.Embedding(2, hidden_size)
        
        self.filter = nn.Embedding(2, hidden_size)
    

    def enter(self, x, mask):
        return (
            self.bias(mask) + 
            (1+self.scale(mask)) * self.norm(x)
        )
    

    def exit(self, hidden_states, y, mask):
        return (
            hidden_states +
            self.filter(mask) * y
        )


class HLmLayer(nn.Module):

    def __init__(self, config: HLmConfig, layer_idx: int):
        super().__init__()

        # basic shapes
        self.hidden_size = config.hidden_size
        self.mlp_size = config.mlp_size

        self.z_size = config.z_size
        self.z_mlp_mult = config.z_mlp_mult
        self.z_mlp_size = self.z_size * self.z_mlp_mult
        
        self.cat_size = self.hidden_size + self.z_size
        self.qkv_size = config.attention_head_size * config.num_attention_heads

        # norms
        self.enc_io = UnconditionalIO(config.hidden_size)
        self.dec_io = UnconditionalIO(config.hidden_size)
        self.gen_io = ConditionalIO(config.hidden_size)

        # input projections
        self.enc_up = FusedLinear(
            self.cat_size,
            [3*self.qkv_size] + [config.mlp_size]*2,
            bias=False,
        )
        self.enc_iaf_up = FusedLinear(
            self.cat_size,
            [2*config.z_size] + [self.z_mlp_size]*2,
            bias=False,
            mask=self._get_iaf_up_mask()
        )
        self.enc_iaf_down = FusedLinear(
            self.z_mlp_size,
            2*self.z_size,
            bias=False,
            mask=self._get_iaf_down_mask()
        )

        self.dec_up = FusedLinear(
            self.cat_size,
            [3*self.qkv_size] + [config.mlp_size]*2,
            bias=False
        )

        self.gen_up = FusedLinear(
            self.cat_size,
            [3*self.qkv_size] + [config.mlp_size]*2,
            bias=False
        )
        self.gen_z_up = FusedLinear(
            self.hidden_size,
            self.z_size,
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
        self.gen_down = FusedLinear(
            [self.qkv_size, config.mlp_size, self.z_size],
            config.hidden_size,
            bias=False
        )

        # z scale
        self.z_scale = np.sqrt(1 / (config.z_size * config.num_layers))


    def _get_iaf_up_mask(self):
        full_size = 2*self.z_size + 2*self.z_mlp_size

        # hidden states can apply to anything
        hidden_mask = torch.ones(full_size, self.hidden_size)

        # bias is auto-regressive (no diagonal)
        noise_bias_mask = torch.tril(torch.ones(self.z_size, self.z_size), diagonal=-1)
        noise_bias_mask = noise_bias_mask.repeat(2, 1)

        # mlp is auto-regressive (with diagonal)
        noise_mlp_mask = torch.tril(torch.ones(self.z_size, self.z_size), diagonal=0)
        noise_mlp_mask = noise_mlp_mask.repeat_interleave(self.z_mlp_mult, dim=0)
        noise_mlp_mask = noise_mlp_mask.repeat(2, 1)

        # combine noise masks
        noise_mask = torch.cat([noise_bias_mask, noise_mlp_mask], dim=0)

        # combine all masks
        return torch.cat([hidden_mask, noise_mask], dim=1)


    def _get_iaf_down_mask(self):
        mask = torch.tril(torch.ones(self.z_size, self.z_size), diagonal=-1)
        mask = mask.repeat_interleave(self.z_mlp_mult, dim=1)
        out = mask.repeat(2, 1)
        return out


    def forward(
        self,
        encoder_states,
        decoder_states,
        generator_states,
        mask,
        noise,
        encoder_attn_mask,
        decoder_attn_mask,
        generator_attn_mask,
    ):
        # noise is only applied to the output area
        float_mask = mask.to(encoder_states.dtype).unsqueeze(-1)
        noise = noise * float_mask

        # get encoder variables
        enc_x = torch.cat([self.enc_io.enter(encoder_states), noise], dim=-1)
        enc_qkv, enc_gate, enc_val = self.enc_up(enc_x)
        enc_z_bias, enc_z_gate, enc_z_val = self.enc_iaf_up(enc_x)

        # get z, set mu and sigma in input area to 0
        enc_mu, enc_log_sigma = (
            float_mask * self.z_scale * (
                enc_z_bias +
                self.enc_iaf_down(self.mlp(enc_z_gate, enc_z_val))
            )
        ).chunk(2, dim=-1)
        enc_sigma = F.softplus(enc_log_sigma + np.log(np.e - 1))

        z = enc_mu + enc_sigma * noise

        # get decoder variables
        dec_x = torch.cat([self.dec_io.enter(decoder_states), z], dim=-1)
        dec_qkv, dec_gate, dec_val = self.dec_up(dec_x)

        # get generator variables, set mu in input area to 0
        gen_x = self.gen_io.enter(generator_states, mask)
        gen_qkv, gen_gate, gen_val = self.gen_up(torch.cat([gen_x, z], dim=-1))
        gen_mu = self.gen_z_up(gen_x) * float_mask * self.z_scale

        # apply attention
        qkv = torch.cat([enc_qkv, dec_qkv, gen_qkv], dim=0)
        attn_mask = torch.cat(
            [
                encoder_attn_mask.expand(encoder_states.shape[0], -1, -1, -1),
                decoder_attn_mask.expand(decoder_states.shape[0], -1, -1, -1),
                generator_attn_mask.expand(generator_states.shape[0], -1, -1, -1),
            ],
            dim=0
        )
        enc_attn_out, dec_attn_out, gen_attn_out = self.attention(
            *qkv.chunk(3, dim=-1),
            attention_mask=attn_mask
        ).chunk(3, dim=0)

        # apply mlp
        enc_mlp_out = self.mlp(enc_gate, enc_val)
        dec_mlp_out = self.mlp(dec_gate, dec_val)
        gen_mlp_out = self.mlp(gen_gate, gen_val)

        encoder_states = self.enc_io.exit(encoder_states, self.enc_down(enc_attn_out, enc_mlp_out, z))
        decoder_states = self.dec_io.exit(decoder_states, self.dec_down(dec_attn_out, dec_mlp_out, z))
        generator_states = self.gen_io.exit(generator_states, self.gen_down(gen_attn_out, gen_mlp_out, z), mask)

        # get kl
        kl = (
            -torch.log(enc_sigma)
            + 0.5 * (enc_sigma**2 + (enc_mu - gen_mu)**2)
            - 0.5
        ).sum(-1).sum(-1)

        uncond_kl = (
            -torch.log(enc_sigma)
            + 0.5 * (enc_sigma**2 + enc_mu**2)
            - 0.5
        ).sum(-1).sum(-1)

        return encoder_states, decoder_states, generator_states, kl, uncond_kl
        

class HLmModel(XLAModel):

    config_class = HLmConfig


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


    def __init__(self, config: HLmConfig, fast_start=False):
        super().__init__(config, fast_start=fast_start)

        self.hidden_size = config.hidden_size
        self.z_size = config.z_size
        self.num_layers = config.num_layers

        self.enc_embs = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dec_embs = nn.Embedding(1, config.hidden_size)
        self.gen_embs = nn.Embedding(config.vocab_size+1, config.hidden_size)

        # layers
        self.layers = nn.ModuleList([
            HLmLayer(config, i)
            for i in range(config.num_layers)
        ])

        # outputs
        self.norm = nn.LayerNorm(config.hidden_size, config.norm_eps, elementwise_affine=True)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.gradient_checkpointing = False

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
        decoder_states = self.dec_embs(torch.zeros_like(input_ids))
        generator_states = torch.where(
            mask.unsqueeze(-1),
            self.gen_embs(torch.zeros_like(input_ids)),
            self.gen_embs(input_ids+1)
        )

        encoder_attn_mask = torch.zeros(1, 1, seq_len, seq_len, device=input_ids.device, dtype=encoder_states.dtype)
        encoder_attn_mask = torch.where(
            mask.unsqueeze(1).unsqueeze(1), # [bs, 1=head, 1=q, seq_len=k]
            torch.zeros_like(encoder_attn_mask),
            torch.full_like(encoder_attn_mask, float('-inf'))
        )
        decoder_attn_mask = encoder_attn_mask
        generator_attn_mask = torch.zeros_like(encoder_attn_mask)

        kl = 0
        uncond_kl = 0
        for i, layer in enumerate(self.layers):

            if self.gradient_checkpointing and constants.XLA_AVAILABLE:
                encoder_states, decoder_states, generator_states, kl_out, uncond_kl_out = self._gradient_checkpointing_func(
                    layer.__call__,
                    encoder_states,
                    decoder_states,
                    generator_states,
                    long_mask,
                    noise[:, :, i],
                    encoder_attn_mask,
                    decoder_attn_mask,
                    generator_attn_mask
                )
            else:
                encoder_states, decoder_states, generator_states, kl_out, uncond_kl_out = layer(
                    encoder_states,
                    decoder_states,
                    generator_states,
                    long_mask,
                    noise[:, :, i],
                    encoder_attn_mask,
                    decoder_attn_mask,
                    generator_attn_mask
                )

            kl = kl + kl_out
            uncond_kl = uncond_kl + uncond_kl_out

        lm_logits = self.lm_head(self.norm(decoder_states))
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        return lm_logits, kl, uncond_kl
