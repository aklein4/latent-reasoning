import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.xla import XLAConfig, XLAModel
from utils.model_utils import (
    FusedLinear,
    RotaryAttention,
    GLU,
    GaussianIAF,
    FullRotaryAttention,
    FullGLU,
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
        num_iaf_attention_heads (`int`):
            Number of attention heads for the encoder IAF.
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
        num_iaf_attention_heads=None,
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
        self.num_iaf_attention_heads = num_iaf_attention_heads
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


class HLmEncoderLayer(nn.Module):
    def __init__(self, config: HLmConfig, layer_idx: int):
        super().__init__()

        # basic shapes
        self.hidden_size = config.hidden_size
        self.qkv_size = config.attention_head_size * config.num_attention_heads
        self.mlp_size = config.mlp_size
        self.z_size = config.z_size
        self.cat_size = self.hidden_size + self.z_size
        
        self.num_attn_heads = config.num_attention_heads
        self.num_iaf_heads = config.num_iaf_attention_heads
        self.num_bid_heads = self.num_attn_heads - self.num_iaf_heads

        # norms
        self.z_io = UnconditionalIO(config.hidden_size)
        self.attn_io = UnconditionalIO(config.hidden_size)
        self.mlp_io = UnconditionalIO(config.hidden_size)

        # z components
        self.z_up = GaussianIAF(
            self.hidden_size,
            self.z_size,
            config.z_mlp_mult,
            config.hidden_act
        )
        self.z_down = nn.Linear(self.z_size, self.hidden_size, bias=False)

        # transformer components
        self.attention = FullRotaryAttention(
            self.cat_size+self.z_size,
            config.attention_head_size,
            config.num_attention_heads,
            config.num_registers,
            True,
            config.rope_fraction,
            config.max_sequence_length,
            config.rope_base,
            layer_idx,
            matrix_mask=self._get_matrix_mask(config),
            out_size=self.hidden_size
        )
        self.mlp = FullGLU(
            self.cat_size, config.mlp_size, config.hidden_act
        )

        # z scale
        self.z_scale = np.sqrt(1 / (config.z_size * config.num_layers))


    def _get_matrix_mask(self, config: HLmConfig):
        
        # hidden states and can apply to anything
        hidden_mask = torch.ones(3*self.qkv_size, self.hidden_size)
        z_mask = torch.ones(3*self.qkv_size, self.z_size)

        # noise can ONLY apply to iaf heads
        noise_iaf_mask = torch.ones(self.num_iaf_heads*config.attention_head_size, self.z_size)
        noise_bid_mask = torch.zeros(self.num_bid_heads*config.attention_head_size, self.z_size)
        noise_mask = torch.cat([noise_iaf_mask, noise_bid_mask], dim=0)
        noise_mask = noise_mask.repeat(3, 1)

        return torch.cat([hidden_mask, z_mask, noise_mask], dim=1)


    @torch.no_grad()
    def _get_iaf_attn_mask(self, attn_mask):
        # expand the mask to number of heads
        attn_mask = attn_mask.expand(-1, self.num_attn_heads, -1, -1)

        # iaf heads can not attend to themselves
        iaf_mask = torch.full_like(attn_mask, float('-inf'))
        iaf_mask[:, :self.num_iaf_heads] = torch.tril(iaf_mask[:, :self.num_iaf_heads], diagonal=-1)

        return attn_mask + iaf_mask
        

    def forward(
        self,
        hidden_states,
        mask,
        noise,
        next_noise,
        attn_mask
    ):
        float_mask = mask.to(hidden_states.dtype).unsqueeze(-1)
        noise = noise * float_mask

        # get z
        mu, log_sigma = (
            float_mask * self.z_scale *
            self.z_up(
                self.z_io.enter(hidden_states),
                noise
            )
        ).chunk(2, dim=-1)
        sigma = F.softplus(log_sigma + np.log(np.e - 1))

        z = mu + sigma * noise

        hidden_states = self.z_io.exit(hidden_states, self.z_down(z))

        # get attn
        x = torch.cat([self.attn_io.enter(hidden_states), z, next_noise], dim=-1)
        attn_out = self.attention(
            x,
            attention_mask=self._get_iaf_attn_mask(attn_mask)
        )
        hidden_states = self.attn_io.exit(hidden_states, attn_out)

        # get mlp
        x = torch.cat([self.mlp_io.enter(hidden_states), z], dim=-1)
        mlp_out = self.mlp(x)
        hidden_states = self.mlp_io.exit(hidden_states, mlp_out)

        return hidden_states, z, mu, sigma
    

class HLmDecoderLayer(nn.Module):

    def __init__(self, config: HLmConfig, layer_idx: int):
        super().__init__()

        # basic shapes
        self.hidden_size = config.hidden_size
        self.qkv_size = config.attention_head_size * config.num_attention_heads
        self.mlp_size = config.mlp_size
        self.z_size = config.z_size
        self.cat_size = self.hidden_size + self.z_size
        
        # norm
        self.io = UnconditionalIO(config.hidden_size)

        # projections
        self.up = FusedLinear(
            self.cat_size,
            [3*self.qkv_size] + [config.mlp_size]*2,
            bias=False
        )
        self.down = FusedLinear(
            [self.qkv_size, config.mlp_size, self.z_size],
            config.hidden_size,
            bias=False
        )

        # transformer components
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


    def forward(
        self,
        hidden_states,
        mask,
        z,
        attn_mask
    ):
        float_mask = mask.to(hidden_states.dtype).unsqueeze(-1)
        z = z * float_mask

        # get values
        x = torch.cat([self.io.enter(hidden_states), z], dim=-1)
        qkv, gate, val = self.up(x)

        # apply transformer
        attn_out = self.attention(
            qkv,
            attention_mask=attn_mask
        )
        mlp_out = self.mlp(gate, val)

        return self.io.exit(hidden_states, self.down(attn_out, mlp_out, z))


class HLmGeneratorLayer(nn.Module):

    def __init__(self, config: HLmConfig, layer_idx: int):
        super().__init__()

        # basic shapes
        self.hidden_size = config.hidden_size
        self.qkv_size = config.attention_head_size * config.num_attention_heads
        self.mlp_size = config.mlp_size
        self.z_size = config.z_size
        self.cat_size = self.hidden_size + self.z_size
        
        # norm
        self.io = ConditionalIO(config.hidden_size)

        # projections
        self.z_proj = nn.Linear(self.hidden_size, self.z_size, bias=False)
        self.up = FusedLinear(
            self.cat_size,
            [3*self.qkv_size] + [config.mlp_size]*2,
            bias=False
        )
        self.down = FusedLinear(
            [self.qkv_size, config.mlp_size, self.z_size],
            config.hidden_size,
            bias=False
        )

        # transformer components
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

        self.z_scale = np.sqrt(1 / (config.z_size * config.num_layers))


    def forward(
        self,
        hidden_states,
        mask,
        attn_mask,
        z=None,
        noise=None
    ):
        assert z is not None or noise is not None
        assert z is None or noise is None

        float_mask = mask.to(hidden_states.dtype).unsqueeze(-1)

        # get z
        x = self.io.enter(hidden_states, mask)
        mu = float_mask * self.z_scale * self.z_proj(x)
        
        if z is None:
            z = mu + noise
        z = z * float_mask
        
        # apply transformer
        qkv, gate, val = self.up(torch.cat([x, z], dim=-1))

        attn_out = self.attention(
            qkv,
            attention_mask=attn_mask
        )
        mlp_out = self.mlp(gate, val)

        hidden_states = self.io.exit(hidden_states, self.down(attn_out, mlp_out, z), mask)

        return hidden_states, mu


class HLmEncoder(nn.Module):

    def __init__(self, config: HLmConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.z_size = config.z_size
        self.num_layers = config.num_layers

        self.embs = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([
            HLmEncoderLayer(config, i)
            for i in range(config.num_layers)
        ])

    
    def forward(
        self,
        input_ids,
        mask,
        noise
    ):
        bs, seq_len = input_ids.shape
        long_mask = mask.long()
        
        hidden_states = self.embs(input_ids)
        
        attn_mask = torch.zeros(1, 1, seq_len, seq_len, device=input_ids.device, dtype=hidden_states.dtype)
        attn_mask = torch.where(
            mask.unsqueeze(1).unsqueeze(1), # [bs, 1=head, 1=q, seq_len=k]
            torch.zeros_like(attn_mask),
            torch.full_like(attn_mask, float('-inf'))
        )

        padded_noise = torch.cat([noise, torch.zeros_like(noise[:, :, -1:])], dim=2)

        zs = []
        mus = []
        sigmas = []
        for i, layer in enumerate(self.layers):
            
            hidden_states, z, mu, sigma = layer(
                hidden_states,
                long_mask,
                noise[:, :, i],
                padded_noise[:, :, i+1],
                attn_mask
            )

            zs.append(z)
            mus.append(mu)
            sigmas.append(sigma)
        
        return (
            torch.cat(zs, dim=2),
            torch.cat(mus, dim=2),
            torch.cat(sigmas, dim=2)
        )
    

class HLmDecoder(nn.Module):

    def __init__(self, config: HLmConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.z_size = config.z_size
        self.num_layers = config.num_layers

        self.embs = nn.Embedding(1, config.hidden_size)

        self.layers = nn.ModuleList([
            HLmDecoderLayer(config, i)
            for i in range(config.num_layers)
        ])

        # outputs
        self.norm = nn.LayerNorm(config.hidden_size, config.norm_eps, elementwise_affine=True)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    
    def forward(
        self,
        input_ids,
        mask,
        z,
    ):
        bs, seq_len = input_ids.shape
        long_mask = mask.long()
        
        hidden_states = torch.where(
            mask.unsqueeze(-1),
            self.embs(torch.zeros_like(input_ids)),
            self.embs(input_ids+1)
        )

        attn_mask = None

        mus = []
        for i, layer in enumerate(self.layers):
            
            hidden_states, mu = layer(
                hidden_states,
                long_mask,
                attn_mask,
                z=z[:, :, i],
            )

            mus.append(mu)

        return torch.cat(mus, dim=2)


class HLmGenerator(nn.Module):

    def __init__(self, config: HLmConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.z_size = config.z_size
        self.num_layers = config.num_layers

        self.embs = nn.Embedding(1+config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([
            HLmDecoderLayer(config, i)
            for i in range(config.num_layers)
        ])

    
    def forward(
        self,
        input_ids,
        mask,
        z,
    ):
        bs, seq_len = input_ids.shape
        long_mask = mask.long()
        
        hidden_states = self.embs(torch.zeros_like(input_ids))
        
        attn_mask = torch.zeros(1, 1, seq_len, seq_len, device=input_ids.device, dtype=hidden_states.dtype)
        attn_mask = torch.where(
            mask.unsqueeze(1).unsqueeze(1), # [bs, 1=head, 1=q, seq_len=k]
            torch.zeros_like(attn_mask),
            torch.full_like(attn_mask, float('-inf'))
        )

        for i, layer in enumerate(self.layers):
            
            hidden_states = layer(
                hidden_states,
                long_mask,
                z[:, :, i],
                attn_mask
            )

        lm_logits = self.lm_head(self.norm(hidden_states))
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        return lm_logits


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

        self.z_size = config.z_size
        self.num_layers = config.num_layers

        self.encoder = HLmEncoder(config)
        self.decoder = HLmDecoder(config)
        self.generator = HLmGenerator(config)

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
            device=input_ids.device, dtype=self.encoder.embs.weight.dtype
        )

        z, enc_mu, enc_sigma = self.encoder(input_ids, mask, noise)
        lm_logits = self.decoder(input_ids, mask, z)
        gen_mu = self.generator(input_ids, mask, z)

        kl = (
            -torch.log(enc_sigma)
            + 0.5 * (enc_sigma**2 + (enc_mu-gen_mu)**2)
            - 0.5
        ).sum(-1).sum(-1).sum(-1)

        uncond_kl = (
            -torch.log(enc_sigma)
            + 0.5 * (enc_sigma**2 + enc_mu**2)
            - 0.5
        ).sum(-1).sum(-1).sum(-1)

        return lm_logits, kl, uncond_kl
