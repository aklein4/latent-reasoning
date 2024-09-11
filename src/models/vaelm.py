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
from utils.logging_utils import log_print


def _get_position_ids(x: torch.LongTensor):
    # simple sequential position ids
    return torch.arange(x.shape[1], dtype=torch.long, device=x.device)[None]


def _get_mask(x: torch.LongTensor):
    # simple causal mask
    mask = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device)
    mask = torch.triu(mask, diagonal=1)
    return torch.masked_fill(
        torch.zeros_like(mask).float(),
        mask,
        float('-inf')
    )[None, None]


class VaeLmConfig(XLAConfig):
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
        layer_norm_eps (`float`):
            The epsilon used by the normalization layers.
        use_rope (`bool`):
            Whether or not to use the RoPE embeddings.
        rope_fraction (`int`):
            The fraction of the hidden size to use for the RoPE embeddings.
        rope_base (`float`):
            The base period of the RoPE embeddings.
        thought_length (`int`):
            The length of the thought vectors.
        z_size (`int`):
            The size of the latent space.
        z_over_scale (`float`):
            The scale of the z values.
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
        layer_norm_eps=None,
        use_rope=None,
        rope_fraction=None,
        rope_base=None,
        thought_length=None,
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

        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        
        self.use_rope = use_rope
        self.rope_fraction = rope_fraction
        self.rope_base = rope_base

        self.thought_length = thought_length
        self.z_size = z_size
        self.z_over_scale = z_over_scale

        super().__init__(*args, **kwargs)


class VaeLmLayer(nn.Module):

    def special_init(self, config: VaeLmConfig):
        if self.is_encoder:
            return

        self.up.linear.weight.data.normal_(0.0, 1/np.sqrt(self.hidden_size))
        self.up.linear.weight.data[-2*self.layer_z_size:].zero_()
        if self.up.linear.bias is not None:
            self.up.linear.bias.data.zero_()
        
        self.up.linear.special_inited = True


    def __init__(self, config: VaeLmConfig, layer_idx: int, is_encoder):
        super().__init__()
        self.is_encoder = is_encoder

        # basic shapes
        self.hidden_size = config.hidden_size
        self.mlp_size = config.mlp_size
        self.qkv_size = config.attention_head_size * config.num_attention_heads
        
        assert config.z_size % config.num_layers == 0
        self.layer_z_size = config.z_size // config.num_layers

        # input norm
        self.norm = RMSNorm(config.hidden_size, config.layer_norm_eps, affine=True)

        # linear projections
        self.up = FusedLinear(
            self.hidden_size,
            [self.qkv_size]*3 + [self.mlp_size]*2 + [self.layer_z_size]*2,
            bias=True
        )
        self.down = FusedLinear(
            [self.qkv_size, self.mlp_size, self.layer_z_size],
            self.hidden_size,
            bias=False,
        )

        # operations
        self.attn = RotaryAttention(
            self.hidden_size,
            config.attention_head_size,
            config.num_attention_heads,
            config.num_registers,
            config.use_rope,
            config.rope_fraction,
            config.max_sequence_length+config.thought_length+2,
            config.rope_base,
            layer_idx
        )
        self.mlp = GLU(config.hidden_act)

        # rescale z for better kl
        self.z_scale = np.sqrt(config.z_over_scale / config.z_size)


    def forward(
        self,
        hidden_states,
        position_ids,
        noise=None,
        z_in=None,
        attention_mask=None,
        past_key_value=None,
    ):
        if self.is_encoder:
            assert z_in is None and noise is not None
        else:
            assert noise is None and z_in is not None

        # get operators, using per-function affine (scales add one for better decay)
        q, k, v, gate, up, mu, log_sigma = self.up(self.norm(hidden_states))

        # apply operators
        attn_out = self.attn(
            q, k, v,
            position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value
        )
        mlp_out = self.mlp(gate, up)

        # fix latent params
        mu = mu * self.z_scale
        sigma = F.softplus(log_sigma * self.z_scale + np.log(np.e - 1))
        
        # if encoder, apply reparametrization for z, otherwise use z_in
        if self.is_encoder:
            z = mu + sigma * noise
        else:
            z = z_in

        # apply down operator (out scale is reszero)
        hidden_states = hidden_states + self.down(attn_out, mlp_out, z)

        # only return z_out if encoder
        if self.is_encoder:
            return hidden_states, mu, sigma, z

        return hidden_states, mu, sigma


class VaeLmEncoder(nn.Module):

    def __init__(self, config: VaeLmConfig):
        super().__init__()

        # basic shapes
        self.hidden_size = config.hidden_size
        self.z_size = config.z_size
        self.layer_z_size = config.z_size // config.num_layers
        self.thought_length = config.thought_length
        self.num_layers = config.num_layers

        # vocab inputs
        self.vocab_embs = nn.Embedding(config.vocab_size, config.hidden_size)
        self.vocab_switch = nn.Parameter(torch.randn([1, 1, config.hidden_size]))

        # encoder inputs
        self.in_proj = nn.Linear(config.z_size, config.hidden_size, bias=False)
        self.thought_embs = nn.Parameter(torch.randn([1, self.thought_length, config.hidden_size]))
        self.thought_switch = nn.Parameter(torch.randn([1, 1, config.hidden_size]))

        # layers
        self.layers = nn.ModuleList(
            [VaeLmLayer(config, layer_idx, is_encoder=True) for layer_idx in range(config.num_layers)]
        )

        self.gradient_checkpointing = False


    def forward(
        self,
        input_ids,
        noise,
        reparam_scale=None
    ):
        bs, seq_len = input_ids.shape

        # basic vocab tokens
        vocab_tokens = (
            self.vocab_embs(input_ids) +
            self.vocab_switch
        ) / np.sqrt(2)

        # encoder takes noise shifted to the right by one [bs, T, L, zD], removing last
        encoder_tokens = (
            self.thought_embs +
            self.thought_switch +
            self.in_proj(
                torch.cat(
                    [
                        torch.zeros_like(noise[:, :1].view(bs, -1, self.z_size)),
                        noise[:, :-1].view(bs, -1, self.z_size)
                    ],
                    dim=1
                )
            )
        ) / np.sqrt(3)

        # combine
        hidden_states = torch.cat([vocab_tokens, encoder_tokens], dim=1)
        position_ids = None
        mask = _get_mask(hidden_states)

        # insert zero noise for the enc vocab tokens
        padded_noise = torch.cat(
            [
                torch.zeros([bs, seq_len, self.num_layers, self.layer_z_size], device=noise.device, dtype=noise.dtype),
                noise,
            ],
            dim=1
        )

        # run layers
        z = []
        mus = []
        sigmas = []
        for ind, layer in enumerate(self.layers):
            hidden_states, mu, sigma, z_curr = layer(
                hidden_states,
                position_ids,
                noise=padded_noise[:, :, ind],
                attention_mask=mask
            )

            z.append(z_curr[:, -encoder_tokens.shape[1]:])
            mus.append(mu[:, -encoder_tokens.shape[1]:])
            sigmas.append(sigma[:, -encoder_tokens.shape[1]:])

        # [bs, T, L, zD]
        z = torch.stack(z, dim=2)
        mus = torch.stack(mus, dim=2)
        sigmas = torch.stack(sigmas, dim=2)

        # apply reparametrization
        if reparam_scale is not None:
            z = z * reparam_scale + ((1-reparam_scale) * z).detach()
            mus = mus * reparam_scale + ((1-reparam_scale) * mus).detach()
            sigmas = sigmas * reparam_scale + ((1-reparam_scale) * sigmas).detach()

        return z, mus, sigmas


class VaeLmDecoder(nn.Module):

    def __init__(self, config: VaeLmConfig):
        super().__init__()

        # basic shapes
        self.hidden_size = config.hidden_size
        self.z_size = config.z_size
        self.layer_z_size = config.z_size // config.num_layers
        self.thought_length = config.thought_length
        self.num_layers = config.num_layers

        # vocab inputs
        self.vocab_start = nn.Parameter(torch.randn([1, 1, config.hidden_size]))
        self.vocab_embs = nn.Embedding(config.vocab_size, config.hidden_size)
        self.vocab_switch = nn.Parameter(torch.randn([1, 1, config.hidden_size]))

        # decoder inputs (extra emb for dummy end token)
        self.in_proj = nn.Linear(config.z_size, config.hidden_size, bias=False)
        self.thought_embs = nn.Parameter(torch.randn([1, self.thought_length+1, config.hidden_size]))
        self.thought_switch = nn.Parameter(torch.randn([1, 1, config.hidden_size]))

        # layers
        self.layers = nn.ModuleList(
            [VaeLmLayer(config, layer_idx, is_encoder=False) for layer_idx in range(config.num_layers)]
        )

        # lm modeling
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.gradient_checkpointing = False


    def forward(
        self,
        input_ids,
        z,
    ):
        bs, seq_len = input_ids.shape

        # vocab tokens get start token added, and remove the last token
        vocab_tokens = (
            torch.cat(
                [
                    self.vocab_start.expand(bs, -1, -1),
                    self.vocab_embs(input_ids[:, :-1])
                ],
                dim=1
             ) +
            self.vocab_switch
        ) / np.sqrt(2)

        # decoder takes z with shift to the right by one [bs, T+1, L, zD], keeping last
        decoder_tokens = (
            self.thought_embs +
            self.thought_switch +
            self.in_proj(
                torch.cat(
                    [
                        torch.zeros_like(z[:, :1].view(bs, -1, self.z_size)),
                        z.view(bs, -1, self.z_size)
                    ],
                    dim=1
                )
            )
        ) / np.sqrt(3)

        # combine
        hidden_states = torch.cat([decoder_tokens, vocab_tokens], dim=1)
        position_ids = None
        mask = _get_mask(hidden_states)

        # insert zero z for the vocab tokens, and a zero for the first thought
        padded_z = torch.cat(
            [
                torch.zeros_like(z[:, :1]),
                z,
                torch.zeros_like(z[:, :1]).expand(-1, vocab_tokens.shape[1], -1, -1)
            ],
            dim=1
        )

        # run layers
        mus = []
        sigmas = []
        for ind, layer in enumerate(self.layers):
            hidden_states, mu, sigma = layer(
                hidden_states,
                position_ids,
                z_in=padded_z[:, :, ind],
                attention_mask=mask
            )

            mus.append(mu[:, :self.thought_length])
            sigmas.append(sigma[:, :self.thought_length])

        mus = torch.stack(mus, dim=2)
        sigmas = torch.stack(sigmas, dim=2)

        # take only the LM states, starting with the start token
        lm_states = self.norm(hidden_states[:, -vocab_tokens.shape[1]:])
        lm_logits = self.lm_head(lm_states)
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        return lm_logits, mus, sigmas


class VaeLmModel(XLAModel):

    config_class = VaeLmConfig


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


    def __init__(self, config: VaeLmConfig, fast_start=False):
        super().__init__(config, fast_start=fast_start)

        # info
        self.vocab_size = config.vocab_size
        self.max_sequence_length = config.max_sequence_length
        self.thought_length = config.thought_length
        self.z_size = config.z_size
        self.layer_z_size = config.z_size // config.num_layers
        self.num_layers = config.num_layers

        # models
        self.encoder = VaeLmEncoder(config)
        self.decoder = VaeLmDecoder(config)

        # positional embeddings
        assert config.use_rope is not None and config.use_rope, "RoPE embeddings are required for VaeLm!"

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids,
        noise=None,
        reparam_scale=None
    ):
        bs, seq_len = input_ids.shape

        if noise is None:
            noise = torch.randn(
                [bs, self.thought_length, self.num_layers, self.layer_z_size],
                device=input_ids.device, dtype=self.encoder.thought_embs.dtype
            )

        # run models
        z, enc_mus, enc_sigmas = self.encoder(input_ids, noise, reparam_scale=reparam_scale)
        lm_logits, dec_mus, dec_sigmas = self.decoder(input_ids, z)

        return lm_logits, enc_mus, enc_sigmas, dec_mus, dec_sigmas
