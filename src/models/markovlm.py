import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.xla import XLAConfig, XLAModel
from utils.model_utils import (
    ReZeroIO,
    FusedLinear,
    RotaryAttention,
    GLU,
)


class MarkovLmConfig(XLAConfig):
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
    """

    model_type = 'markovlm'


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
        control=False,
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
        assert z_size % num_layers == 0, f"z_size ({z_size}) must be divisible by num_layers ({num_layers})"

        self.control = control

        super().__init__(*args, **kwargs)


class MarkovLmEncoderLayer(nn.Module):

    def __init__(self, config: MarkovLmConfig, layer_idx: int):
        super().__init__()
        qkv_size = config.attention_head_size * config.num_attention_heads

        # basic shapes
        self.hidden_size = config.hidden_size
        self.layer_z_size = config.z_size // config.num_layers

        # norms
        self.io = ReZeroIO(self.hidden_size, config.norm_eps)

        # z components
        self.z_proj = nn.Linear(self.hidden_size, 2*self.layer_z_size, bias=False)

        # transformer projections
        self.up = FusedLinear(
            [self.hidden_size, self.layer_z_size],
            [qkv_size]*3 + [config.mlp_size]*2,
            bias=False
        )
        self.down = FusedLinear(
            [self.layer_z_size, qkv_size, config.mlp_size],
            self.hidden_size,
            bias=False
        )

        self.attention = RotaryAttention(
            config.attention_head_size,
            config.num_attention_heads,
            config.num_registers,
            True,
            config.rope_fraction,
            config.max_sequence_length,
            config.rope_base,
            layer_idx,
        )
        self.mlp = GLU(config.hidden_act)

        # z scale
        self.z_scale = np.sqrt(1 / config.z_size)


    def forward(
        self,
        hidden_states,
        noise,
        attn_mask,
    ):

        x = self.io.enter(hidden_states)

        # get z
        mu, log_sigma = (
            self.z_scale *
            self.z_proj(x)
        ).chunk(2, dim=-1)
        sigma = F.softplus(log_sigma + np.log(np.e - 1))

        z = mu + sigma * noise

        # apply transformer
        q, k, v, mlp_gate, mlp_val = self.up(x, z)
        attn_out = self.attention(
            q, k, v,
            attention_mask=attn_mask
        )
        mlp_out = self.mlp(mlp_gate, mlp_val)

        y = self.down(z, attn_out, mlp_out)
        hidden_states = self.io.exit(hidden_states, y)

        return hidden_states, z, mu, sigma


class MarkovLmDecoderLayer(nn.Module):

    def __init__(self, config: MarkovLmConfig, layer_idx: int):
        super().__init__()

        # basic shapes
        self.hidden_size = config.hidden_size
        self.layer_z_size = config.z_size // config.num_layers

        # norms
        self.io = ReZeroIO(self.hidden_size, config.norm_eps)

        # z components
        self.z_proj = nn.Linear(self.hidden_size, 2*self.layer_z_size, bias=False)

        # transformer projections
        self.up = FusedLinear(
            [self.hidden_size, self.layer_z_size],
            [config.mlp_size]*2,
            bias=False
        )
        self.down = FusedLinear(
            [self.layer_z_size, config.mlp_size],
            self.hidden_size,
            bias=False
        )

        self.mlp = GLU(config.hidden_act)

        # z scale
        self.z_scale = np.sqrt(1 / config.z_size)


    def forward(
        self,
        hidden_states,
        z=None,
        noise=None,
    ):
        assert z is not None or noise is not None
        assert z is None or noise is None

        x = self.io.enter(hidden_states)

        # get z
        mu, log_sigma = (
            self.z_scale *
            self.z_proj(x)
        ).chunk(2, dim=-1)
        sigma = F.softplus(log_sigma + np.log(np.e - 1))

        if z is None:
            z = mu + sigma * noise

        # apply transformer
        mlp_gate, mlp_val = self.up(x, z)
        mlp_out = self.mlp(mlp_gate, mlp_val)

        y = self.down(z, mlp_out)
        hidden_states = self.io.exit(hidden_states, y)

        return hidden_states, z, mu, sigma


class MarkovLmEncoder(nn.Module):

    def __init__(self, config: MarkovLmConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.z_size = config.z_size
        self.layer_z_size = config.z_size // config.num_layers

        self.embs = nn.Embedding(config.vocab_size, config.hidden_size)
        self.noise_proj = nn.Linear(self.z_size, self.hidden_size, bias=False)

        self.layers = nn.ModuleList([
            MarkovLmEncoderLayer(config, i)
            for i in range(config.num_layers)
        ])

        self.control = config.control

    
    def forward(
        self,
        input_ids,
        noise
    ):
        bs, seq_len = input_ids.shape
        
        hidden_states = self.embs(input_ids)
        if not self.control:
            hidden_states[:, :-1] = (
                hidden_states[:, :-1] +
                self.noise_proj(noise.view(bs, seq_len, self.z_size)[:, 1:])
            ) / np.sqrt(2)

        attn_mask = torch.full(
            [1, 1, seq_len, seq_len],
            float("-inf"),
            device=input_ids.device, dtype=hidden_states.dtype
        )
        attn_mask = torch.tril(attn_mask, diagonal=-1).detach()
        if self.control:
            attn_mask = attn_mask.transpose(-1, -2) + attn_mask

        zs = []
        mus = []
        sigmas = []
        for i, layer in enumerate(self.layers):
            
            hidden_states, z, mu, sigma = layer(
                hidden_states,
                noise=noise[:, :, i],
                attn_mask=attn_mask
            )

            zs.append(z)
            mus.append(mu)
            sigmas.append(sigma)
        
        return (
            torch.stack(zs, dim=2),
            torch.stack(mus, dim=2),
            torch.stack(sigmas, dim=2)
        )
    

class MarkovLmDecoder(nn.Module):

    def __init__(self, config: MarkovLmConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.z_size = config.z_size
        self.layer_z_size = config.z_size // config.num_layers

        self.embs = nn.Embedding(config.vocab_size, config.hidden_size)
        self.z_proj = nn.Linear(config.z_size, config.hidden_size, bias=False)

        self.layers = nn.ModuleList([
            MarkovLmDecoderLayer(config, i)
            for i in range(config.num_layers)
        ])

        # outputs
        self.norm = nn.LayerNorm(config.hidden_size, config.norm_eps, elementwise_affine=True)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


    def forward(
        self,
        input_ids,
        z,
    ):
        bs, seq_len = input_ids.shape
        
        hidden_states = self.embs(input_ids)
        hidden_states[:, 1:] = (
            hidden_states[:, 1:] +
            self.z_proj(z.view(bs, seq_len, self.z_size)[:, 1:])
        ) / np.sqrt(2)

        z_shifted = torch.cat([z[:, 1:], torch.zeros_like(z[:, :1])], dim=1)

        mus = []
        sigmas = []
        for i, layer in enumerate(self.layers):
            
            hidden_states, _, mu, sigma = layer(
                hidden_states,
                z=z_shifted[:, :, i],
            )

            mus.append(mu)
            sigmas.append(sigma)

        lm_logits = self.lm_head(self.norm(hidden_states))
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        mus = torch.stack(mus, dim=2)
        sigmas = torch.stack(sigmas, dim=2)

        return lm_logits, mus, sigmas


class MarkovLmModel(XLAModel):

    config_class = MarkovLmConfig


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


    def __init__(self, config: MarkovLmConfig, fast_start=False):
        super().__init__(config, fast_start=fast_start)

        self.z_size = config.z_size
        self.num_layers = config.num_layers
        self.layer_z_size = config.z_size // config.num_layers

        self.encoder = MarkovLmEncoder(config)
        self.decoder = MarkovLmDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()


    def kl(self, enc_mu, enc_sigma, dec_mu, dec_sigma):
        return (
            torch.log(dec_sigma) - torch.log(enc_sigma)
            + 0.5 * (enc_sigma**2 + (enc_mu-dec_mu)**2) / (dec_sigma**2)
            - 0.5
        ).sum(-1).sum(-1)


    def forward(
        self,
        input_ids,
    ):
        bs, seq_len = input_ids.shape

        # sample noise for the encoder
        noise = torch.randn(
            [bs, seq_len, self.num_layers, self.layer_z_size],
            device=input_ids.device, dtype=self.encoder.embs.weight.dtype
        )

        z, enc_mu, enc_sigma = self.encoder(input_ids, noise)
        lm_logits, dec_mu, dec_sigma = self.decoder(input_ids, z)

        # we never use the first token
        enc_mu = enc_mu[:, 1:]
        enc_sigma = enc_sigma[:, 1:]

        # we never use the last token
        dec_mu = dec_mu[:, :-1]
        dec_sigma = dec_sigma[:, :-1]

        kl = self.kl(enc_mu, enc_sigma, dec_mu, dec_sigma)

        # add a zero to the end for padding
        kl = torch.cat([kl, torch.zeros_like(kl[:, :1])], dim=1)

        return lm_logits, kl
