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

        super().__init__(*args, **kwargs)


class VaeLmLayer(nn.Module):

    def special_init(self, config: VaeLmConfig):

        self.up_in_scales.weight.data.zero_()
        self.up_out_scales.weight.data.zero_()
        self.up_out_biases.weight.data.zero_()
        self.down_in_scales.weight.data.zero_()
        self.down_out_scales.weight.data.zero_()

        self.up_in_scales.special_inited = True
        self.up_out_scales.special_inited = True
        self.up_out_biases.special_inited = True
        self.down_in_scales.special_inited = True
        self.down_out_scales.special_inited = True


    def __init__(self, config: VaeLmConfig, layer_idx: int):
        super().__init__()

        # basic shapes
        self.hidden_size = config.hidden_size
        self.mlp_size = config.mlp_size
        self.qkv_size = config.attention_head_size * config.num_attention_heads
        
        assert config.z_size % config.num_layers == 0
        self.layer_z_size = config.z_size // config.num_layers

        # linear projections
        self.up = FusedLinear(
            self.hidden_size,
            [self.qkv_size]*3 + [self.mlp_size]*2 + [self.layer_z_size]*4,
            bias=False
        )
        self.down = FusedLinear(
            [self.qkv_size, self.mlp_size, self.layer_z_size, self.layer_z_size],
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

        # input norm (no affine)
        self.norm = RMSNorm(config.hidden_size, config.layer_norm_eps, affine=False)

        # per-function scales and biases
        self.up_in_scales = nn.Embedding(4, self.hidden_size)
        self.up_out_scales = nn.Embedding(4, self.up.total_out)
        self.up_out_biases = nn.Embedding(4, self.up.total_out)

        self.down_in_scales = nn.Embedding(4, self.down.total_in)
        self.down_out_scales = nn.Embedding(4, self.hidden_size)

        # rescale z for better kl
        self.z_scale = 1 / np.sqrt(config.z_size)


    def forward(
        self,
        is_encoder,
        hidden_states,
        state_types,
        position_ids,
        noise=None,
        z_in=None,
        attention_mask=None,
        past_key_value=None,
    ):
        if self.attn.layer_idx == 0:
            log_print(torch.is_grad_enabled())

        # get operators, using per-function affine (scales add one for better decay)
        q, k, v, gate, up, enc_mu, enc_log_sigma, dec_mu, dec_log_sigma = self.up(
            self.norm(hidden_states),
            in_scale=(self.up_in_scales(state_types)+1),
            scale=(self.up_out_scales(state_types)+1),
            bias=self.up_out_biases(state_types)
        )

        # discard the unused latent params
        if is_encoder:
            mu = enc_mu
            log_sigma = enc_log_sigma
        else:
            mu = dec_mu
            log_sigma = dec_log_sigma

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
        
        # if encoder, apply reparametrization for z_out, ignore z_in
        if is_encoder:
            z_out = mu + sigma * noise
            z_in = torch.zeros_like(z_out)

        # if decoder, use z_in, ignore z_out
        else:
            z_out = torch.zeros_like(z_in)

        # apply down operator (out scale is reszero)
        hidden_states = hidden_states + self.down(
            attn_out, mlp_out, z_in, z_out,
            in_scale=(self.down_in_scales(state_types)+1),
            scale=self.down_out_scales(state_types)
        )

        # only return z_out if encoder
        if is_encoder:
            return hidden_states, mu, sigma, z_out

        return hidden_states, mu, sigma


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

        # embeddings
        self.vocab_embs = nn.Embedding(config.vocab_size, config.hidden_size)
        # encoder gets one token per thought
        self.encoder_embs = nn.Parameter(torch.randn([1, self.thought_length, config.hidden_size]))
        # decoder gets one token per thought + extra intake token
        self.decoder_embs = nn.Parameter(torch.randn([1, self.thought_length+1, config.hidden_size]))
        # start token to generate first output
        self.decoder_start = nn.Parameter(torch.randn([1, 1, config.hidden_size]))

        # input projections (project from z to hidden size)
        self.encoder_in_proj = nn.Linear(config.z_size, config.hidden_size, bias=False)
        self.decoder_in_proj = nn.Linear(config.z_size, config.hidden_size, bias=False)

        # positional embeddings
        assert config.use_rope is not None and config.use_rope, "RoPE embeddings are required for VaeLm!"

        # layers
        self.num_layers = config.num_layers
        self.layers = nn.ModuleList(
            [VaeLmLayer(config, layer_idx) for layer_idx in range(config.num_layers)]
        )

        # lm modeling
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # z info
        self.z_size = config.z_size
        self.layer_z_size = config.z_size // config.num_layers

        # gradient checkpointing
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


    def _get_position_ids(self, x: torch.LongTensor):
        # simple sequential position ids
        return torch.arange(x.shape[1], dtype=torch.long, device=x.device)[None]


    def _get_mask(self, x: torch.LongTensor):
        # simple causal mask
        mask = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device)
        mask = torch.triu(mask, diagonal=1)
        return torch.masked_fill(
            torch.zeros_like(mask).float(),
            mask,
            float('-inf')
        )[None, None]


    def forward(
        self,
        input_ids,
    ):
        bs, seq_len = input_ids.shape

        """ Easy things """
        noise = torch.randn(
            [bs, self.thought_length, self.num_layers, self.layer_z_size],
            device=input_ids.device, dtype=self.decoder_start.dtype
        )
        x_tokens = self.vocab_embs(input_ids)

        """ Encoder """

        # encoder takes noise shifted to the right by one [bs, T, L, zD], removing last
        encoder_tokens = (
            self.encoder_embs.expand(bs, -1, -1) +
            self.encoder_in_proj(
                torch.cat(
                    [
                        torch.zeros_like(noise[:, :1].view(bs, -1, self.z_size)),
                        noise[:, :-1].view(bs, -1, self.z_size)
                    ],
                    dim=1
                )
            )
        ) / np.sqrt(2)
        encoder_states = torch.cat([x_tokens, encoder_tokens], dim=1)
        position_ids = self._get_position_ids(encoder_states)
        mask = self._get_mask(encoder_states)

        # enc vocab = 0 [bs, S], enc compute = 1 [bs, T]
        encoder_types = torch.cat([torch.zeros_like(x_tokens[:, :, 0]).long(), torch.zeros_like(encoder_tokens[:, :, 0]).long()+1], dim=1)
        
        # insert zero noise for the enc vocab tokens
        enc_noise = torch.cat(
            [
                torch.zeros([bs, seq_len, self.num_layers, self.layer_z_size], device=noise.device, dtype=noise.dtype),
                noise,
            ],
            dim=1
        )

        # run encoder
        z = []
        enc_mus = []
        enc_sigmas = []
        for idx, layer in enumerate(self.layers):

            if self.gradient_checkpointing:
                encoder_states, m, s, z_out = self._gradient_checkpointing_func(
                    layer.__call__,
                    True,
                    encoder_states,
                    encoder_types,
                    position_ids,
                    enc_noise[:, :, idx],
                    None,
                    mask
                )
            else:
                encoder_states, m, s, z_out = layer(
                    is_encoder=True,
                    hidden_states=encoder_states,
                    state_types=encoder_types,
                    position_ids=position_ids,
                    noise=enc_noise[:, :, idx],
                    z_in=None,
                    attention_mask=mask
                )

            # take only the thought vectors
            z.append(z_out[:, -self.thought_length:])
            enc_mus.append(m[:, -self.thought_length:])
            enc_sigmas.append(s[:, -self.thought_length:])
        
        # [bs, T, L, zD]
        z = torch.stack(z, dim=2)
        enc_mus = torch.stack(enc_mus, dim=2)
        enc_sigmas = torch.stack(enc_sigmas, dim=2)

        """ Decoder """

        # decoder takes z shifted to the right by one [bs, T+1, L, zD], keeping last
        decoder_tokens = (
            self.decoder_embs.expand(bs, -1, -1) +
            self.decoder_in_proj(
                torch.cat(
                    [
                        torch.zeros_like(z[:, :1].view(bs, -1, self.z_size)),
                        z.view(bs, -1, self.z_size)
                    ],
                    dim=1
                )
            )
        ) / np.sqrt(2)

        # we add the start token before vocab, and remove last vocab token
        decoder_states = torch.cat([decoder_tokens, self.decoder_start.expand(bs, -1, -1), x_tokens[:, :-1]], dim=1)
        position_ids = self._get_position_ids(decoder_states)
        mask = self._get_mask(decoder_states)

        # dec thought = 2 [bs, T+1], dec out = 3 [bs, T]
        decoder_types = torch.cat([torch.zeros_like(decoder_tokens[:, :, 0]).long()+2, torch.zeros_like(x_tokens[:, :, 0]).long()+3], dim=1)

        # insert zero z for the dec vocab tokens AND the last thought token (it does not generate)
        dec_z = torch.cat(
            [
                z,
                torch.zeros([bs, 1+seq_len, self.num_layers, self.layer_z_size], device=z.device, dtype=z.dtype)
            ],
            dim=1
        )

        # run decoder
        dec_mus = []
        dec_sigmas = []
        for idx, layer in enumerate(self.layers):

            if self.gradient_checkpointing:
                decoder_states, m, s = self._gradient_checkpointing_func(
                    layer.__call__,
                    False,
                    decoder_states,
                    decoder_types,
                    position_ids,
                    None,
                    dec_z[:, :, idx],
                    mask
                )
            else:
                decoder_states, m, s = layer(
                    is_encoder=False,
                    hidden_states=decoder_states,
                    state_types=decoder_types,
                    position_ids=position_ids,
                    noise=None,
                    z_in=dec_z[:, :, idx],
                    attention_mask=mask
                )

            # take only the thought vectors (last thought token does not generate)
            dec_mus.append(m[:, :self.thought_length])
            dec_sigmas.append(s[:, :self.thought_length])

        dec_mus = torch.stack(dec_mus, dim=2)
        dec_sigmas = torch.stack(dec_sigmas, dim=2)

        """ LM """

        # take only the LM states, starting with the start token
        lm_states = self.norm(decoder_states[:, self.thought_length+1:])
        assert lm_states.shape[1] == seq_len
        
        lm_logits = self.lm_head(lm_states)
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        """ Return """

        return lm_logits, enc_mus, enc_sigmas, dec_mus, dec_sigmas
