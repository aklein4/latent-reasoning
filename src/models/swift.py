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

    model_type = 'base'

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

        # derived
        assert self.z_size % self.num_layers == 0
        self.layer_z_size = self.z_size // self.num_layers

        super().__init__(*args, **kwargs)


class SwiftLayer(nn.Module):

    def special_init(self, config: SwiftConfig):
        if self.mode != 'generator':
            return

        self.z_up.linear.weight.data.zero_()
        if self.z_up.linear.bias is not None:
            self.z_up.linear.bias.data.zero_()

        self.z_up.linear.special_inited = True


    def __init__(self, config: SwiftConfig, layer_idx: int, mode):
        super().__init__()
        self.mode = mode
        assert mode in ["encoder", "generator", "decoder"]

        # basic shapes
        self.hidden_size = config.hidden_size
        self.layer_z_size = config.layer_z_size
        
        # input norms
        self.attn_norm = RMSNorm(config.hidden_size, config.norm_eps, affine=True)
        self.mlp_norm = RMSNorm(config.hidden_size, config.norm_eps, affine=True)
        self.z_norm = RMSNorm(config.hidden_size, config.norm_eps, affine=True)

        # output filters
        self.attn_filter = nn.Parameter(torch.zeros([1, 1, config.hidden_size]))
        self.mlp_filter = nn.Parameter(torch.zeros([1, 1, config.hidden_size]))
        self.z_filter = nn.Parameter(torch.zeros([1, 1, config.hidden_size]))

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

        self.z_up = FusedLinear(
            config.hidden_size,
            [config.layer_z_size]*2,
            bias=False
        )
        self.z_down = nn.Linear(
            config.layer_z_size,
            config.hidden_size,
            bias=False
        )

        # rescale z for better kl
        self.z_scale = np.sqrt(config.z_over_scale / config.z_size)


    def forward(
        self,
        hidden_states,
        mask,
        noise=None,
        z_in=None,
        position_ids=None,
        attention_mask=None,
        past_key_value=None,
    ):
        if self.mode == "encoder":
            assert z_in is None and noise is not None
        elif self.mode == "generator":
            assert noise is None and z_in is not None
        else:
            assert noise is None and z_in is None

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

        if self.mode == 'decoder':
            return hidden_states

        # z
        mu, log_sigma = self.z_up(self.z_norm(hidden_states))
        mu = mu * self.z_scale
        sigma = F.softplus(log_sigma * self.z_scale + np.log(np.e - 1))
        if self.mode == 'generator':
            sigma = torch.ones_like(sigma)

        # apply mask
        mu = torch.where(
            mask.unsqueeze(-1),
            mu,
            torch.zeros_like(mu)
        )
        sigma = torch.where(
            mask.unsqueeze(-1),
            sigma,
            torch.ones_like(sigma)
        )

        # if encoder, apply reparametrization for z, otherwise use z_in
        if self.mode == 'encoder':
            z = mu + sigma * noise
        else:
            z = z_in

        # apply down operator (out scale is reszero)
        z_out = self.z_down(z)
        hidden_states = hidden_states + self.z_filter * z_out

        # only return z_out if encoder
        if self.mode == 'encoder':
            return hidden_states, mu, sigma, z

        return hidden_states, mu, sigma


class SwiftEncoder(nn.Module):

    def __init__(self, config: SwiftConfig):
        super().__init__()

        # basic shapes
        self.hidden_size = config.hidden_size
        self.z_size = config.z_size
        self.layer_z_size = config.layer_z_size

        # vocab inputs
        self.vocab_embs = nn.Embedding(config.vocab_size, config.hidden_size)

        # switches
        self.vocab_switch = nn.Parameter(torch.randn([1, 1, config.hidden_size]))
        self.x_switch = nn.Parameter(torch.randn([1, 1, config.hidden_size]))

        # layers
        self.layers = nn.ModuleList(
            [SwiftLayer(config, layer_idx, "encoder") for layer_idx in range(config.num_layers)]
        )

        self.gradient_checkpointing = False


    def forward(
        self,
        input_ids,
        mask,
        noise,
    ):

        # basic vocab tokens
        vocab_tokens = (self.vocab_embs(input_ids))

        hidden_states = torch.where(
            mask.unsqueeze(-1),
            vocab_tokens + self.x_switch,
            vocab_tokens + self.vocab_switch,
        ) / np.sqrt(2)

        # insert zero noise for the enc vocab tokens
        padded_noise = torch.where(
            mask.unsqueeze(-1).unsqueeze(-1),
            noise,
            torch.zeros_like(noise)
        )

        # run layers
        z = []
        mus = []
        sigmas = []
        for ind, layer in enumerate(self.layers):
            if self.gradient_checkpointing:
                hidden_states, mu, sigma, z_curr = self._gradient_checkpointing_func(
                    layer,
                    hidden_states,
                    mask,
                    padded_noise[:, :, ind]
                )
            else:
                hidden_states, mu, sigma, z_curr = layer(
                    hidden_states,
                    mask,
                    padded_noise[:, :, ind]
                )

            z.append(z_curr)
            mus.append(mu)
            sigmas.append(sigma)

        # [bs, T, L, zD]
        z = torch.stack(z, dim=2)
        mus = torch.stack(mus, dim=2)
        sigmas = torch.stack(sigmas, dim=2)

        return z, mus, sigmas


class SwiftGenerator(nn.Module):

    def __init__(self, config: SwiftConfig):
        super().__init__()

        # basic shapes
        self.hidden_size = config.hidden_size
        self.z_size = config.z_size
        self.layers_z_size = config.layer_z_size

        # vocab inputs
        self.vocab_embs = nn.Embedding(2+config.vocab_size, config.hidden_size)
        self.prompt_switch = nn.Parameter(torch.randn([1, 1, config.hidden_size]))

        # layers
        self.layers = nn.ModuleList(
            [SwiftLayer(config, layer_idx, "generator") for layer_idx in range(config.num_layers)]
        )

        self.gradient_checkpointing = False


    def forward(
        self,
        input_ids,
        mask,
        z,
        uncond_mask
    ):

        hidden_states = torch.where(
            mask.unsqueeze(-1),
            self.vocab_embs(torch.ones_like(input_ids)),
            (self.vocab_embs(input_ids+2) + self.prompt_switch) / np.sqrt(2)
        )
        hidden_states = torch.where(
            uncond_mask.unsqueeze(-1).unsqueeze(-1) & ~mask.unsqueeze(-1),
            self.vocab_embs(torch.zeros_like(input_ids)),
            hidden_states
        )

        # insert zero z for the vocab tokens, and a zero for the first thought
        padded_z = torch.where(
            mask.unsqueeze(-1).unsqueeze(-1),
            z,
            torch.zeros_like(z)
        )

        # run layers
        mus = []
        sigmas = []
        for ind, layer in enumerate(self.layers):
            if self.gradient_checkpointing:
                hidden_states, mu, sigma = self._gradient_checkpointing_func(
                    layer,
                    hidden_states,
                    mask,
                    None,
                    padded_z[:, :, ind],
                )
            else:
                hidden_states, mu, sigma = layer(
                    hidden_states,
                    mask,
                    None,
                    padded_z[:, :, ind],
                )

            mus.append(mu)
            sigmas.append(sigma)

        mus = torch.stack(mus, dim=2)
        sigmas = torch.stack(sigmas, dim=2)

        # take only the LM states, starting with the start token
        return mus, sigmas


class SwiftDecoder(nn.Module):

    def __init__(self, config: SwiftConfig):
        super().__init__()

        # basic shapes
        self.hidden_size = config.hidden_size
        self.z_size = config.z_size

        # vocab inputs
        self.proj_in = nn.Linear(config.z_size, config.hidden_size, bias=False)

        # layers
        self.layers = nn.ModuleList(
            [SwiftLayer(config, layer_idx, "decoder") for layer_idx in range(config.num_decoder_layers)]
        )

        # lm modeling
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps, affine=True)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.gradient_checkpointing = False
    

    def forward(
        self,
        z
    ):
        
        hidden_states = self.proj_in(z.view(z.shape[0], -1, self.z_size))

        for layer in self.layers:
            if self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    layer,
                    hidden_states,
                    None
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    None
                )
        
        lm_logits = self.lm_head(self.norm(hidden_states))
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        return lm_logits


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

        self.num_layers = config.num_layers
        self.layer_z_size = config.layer_z_size

        self.encoder = SwiftEncoder(config)
        self.generator = SwiftGenerator(config)
        self.decoder = SwiftDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids,
        mask,
        uncond_mask
    ):
        bs, seq_len = input_ids.shape

        noise = torch.randn(
            [bs, seq_len, self.config.num_layers, self.config.layer_z_size],
            device=input_ids.device, dtype=self.encoder.vocab_switch.dtype
        )

        z, enc_mu, enc_sigma = self.encoder(input_ids, mask, noise)
        gen_mu, gen_sigma = self.generator(input_ids, mask, z, uncond_mask)
        lm_logits = self.decoder(z)

        return lm_logits, enc_mu, enc_sigma, gen_mu, gen_sigma
