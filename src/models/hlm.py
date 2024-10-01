import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.xla import XLAConfig, XLAModel
from utils.model_utils import (
    FusedLinear,
    RotaryAttention,
    GLU,
    ReZeroIO,
)
from utils.prob_utils import GaussianIAF
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
        num_decoder_layers (`int`):
            Number of hidden layers in the Transformer decoder
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
        z_output_layers (`int`):
            The number of layers to keep of z for use by the decoder.
        reverse_z (`bool`):
            Whether to reverse the order of the z outputs.
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
        z_output_layers=None,
        reverse_z=None,
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

        self.z_output_layers = z_output_layers
        self.reverse_z = reverse_z

        super().__init__(*args, **kwargs)


class ConditionalIO(nn.Module):

    def special_init(self, config: HLmConfig): 
        self.scale.weight.data.zero_()
        self.bias.weight.data.zero_()
        self.filter.weight.data.zero_()
        self.scale.special_inited = True
        self.bias.special_inited = True
        self.filter.special_inited = True


    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=False)
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


class LmHead(nn.Module):

    def __init__(self, config: HLmConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, elementwise_affine=True)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states):
        lm_logits = self.lm_head(self.norm(hidden_states))
        return F.log_softmax(
            lm_logits,
            dim=-1,
            dtype=(torch.bfloat16 if constants.XLA_AVAILABLE else None)
        )


class HLmEncoderLayer(nn.Module):
    def __init__(self, config: HLmConfig, layer_idx: int):
        super().__init__()

        # basic shapes
        self.hidden_size = config.hidden_size
        self.qkv_size = config.attention_head_size * config.num_attention_heads
        self.mlp_size = config.mlp_size
        self.z_size = config.z_size
        
        # attention shapes
        self.num_attn_heads = config.num_attention_heads
        self.num_iaf_heads = config.num_iaf_attention_heads
        self.num_bid_heads = self.num_attn_heads - self.num_iaf_heads
        self.iaf_output_size = self.num_iaf_heads * config.attention_head_size

        # norms
        self.io = ReZeroIO(config.hidden_size, config.norm_eps)

        # z components
        self.z_proj = GaussianIAF(
            self.hidden_size,
            self.z_size,
            config.z_mlp_mult,
            config.hidden_act
        )

        # transformer projections
        self.up = FusedLinear(
            [self.hidden_size, self.z_size],
            [self.qkv_size, 2*self.qkv_size] + [config.mlp_size]*2,
            bias=False
        )
        self.iaf_up = FusedLinear(
            self.z_size,
            2*self.qkv_size,
            bias=False,
            mask=self._get_iaf_up_mask(config)
        )
        self.down = FusedLinear(
            [self.z_size, self.qkv_size, config.mlp_size],
            config.hidden_size,
            bias=False
        )

        self.attention = RotaryAttention(
            config.attention_head_size,
            config.num_attention_heads,
            8 if (config.num_registers == 0) else config.num_registers,
            True,
            config.rope_fraction,
            config.max_sequence_length,
            config.rope_base,
            layer_idx,
            position_scale=(config.patch_size if hasattr(config, 'patch_size') else 1.0)
        )
        self.mlp = GLU(config.hidden_act)

        # z scale
        self.z_scale = np.sqrt(
            (config.patch_size if hasattr(config, 'patch_size') else 1.0) /
            (config.z_size * config.num_layers)
        )


    @torch.no_grad()
    def _get_iaf_up_mask(self, config: HLmConfig):
  
        # only iaf heads can see the next noise
        noise_iaf_mask = torch.ones(self.num_iaf_heads*config.attention_head_size, self.z_size)
        noise_bid_mask = torch.zeros(self.num_bid_heads*config.attention_head_size, self.z_size)
        
        # expand to cover k and v
        noise_kv_mask = torch.cat([noise_iaf_mask, noise_bid_mask], dim=0)
        noise_kv_mask = noise_kv_mask.repeat(2, 1)

        return noise_kv_mask


    @torch.no_grad()
    def get_iaf_attn_mask(self, attn_mask):
        # expand the mask to number of heads
        attn_mask = attn_mask.expand(-1, self.num_attn_heads, -1, -1).clone()

        # iaf heads can not attend to themselves
        iaf_mask = torch.full_like(attn_mask, float('-inf'))
        attn_mask[:, :self.num_iaf_heads] += torch.triu(iaf_mask[:, :self.num_iaf_heads], diagonal=0)

        return attn_mask
        

    def forward(
        self,
        hidden_states,
        mask,
        noise,
        next_noise,
        attn_mask
    ):
        float_mask = mask.to(hidden_states.dtype).unsqueeze(-1)
        noise = noise * float_mask # noise becomes zero where not used

        x = self.io.enter(hidden_states)

        # get z, params become zero where not used
        mu, log_sigma = (
            float_mask * self.z_scale *
            self.z_proj(
                x,
                noise
            )
        ).chunk(2, dim=-1)
        sigma = F.softplus(log_sigma + np.log(np.e - 1))

        # z becomes zero when noise and params are zeroed
        z = mu + sigma * noise

        # apply transformer
        q, kv, mlp_gate, mlp_val = self.up(x, z)
        kv = kv + self.iaf_up(next_noise)

        # get attn
        attn_out = self.attention(
            q, *kv.chunk(2, dim=-1),
            attention_mask=attn_mask
        )
        mlp_out = self.mlp(mlp_gate, mlp_val)

        hidden_states = self.io.exit(
            hidden_states,
            self.down(z, attn_out, mlp_out),
        )

        return hidden_states, z, mu, sigma
    

class HLmGeneratorLayer(nn.Module):

    def __init__(self, config: HLmConfig, layer_idx: int):
        super().__init__()

        # basic shapes
        self.hidden_size = config.hidden_size
        self.qkv_size = config.attention_head_size * config.num_attention_heads
        self.mlp_size = config.mlp_size
        self.z_size = config.z_size
        
        # norm
        self.io = ConditionalIO(config.hidden_size, config.norm_eps)

        # z components
        self.z_proj = nn.Linear(self.hidden_size, self.z_size, bias=False)

        # transformer projections
        self.up = FusedLinear(
            [self.hidden_size, self.z_size],
            [self.qkv_size]*3 + [config.mlp_size]*2,
            bias=False
        )
        self.down = FusedLinear(
            [self.z_size, self.qkv_size, config.mlp_size],
            config.hidden_size,
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
            position_scale=(config.patch_size if hasattr(config, 'patch_size') else 1.0)
        )
        self.mlp = GLU(config.hidden_act)

        # z scale
        self.z_scale = np.sqrt(
            (config.patch_size if hasattr(config, 'patch_size') else 1.0) /
            (config.z_size * config.num_layers)
        )


    def forward(
        self,
        hidden_states,
        mask,
        z=None,
        noise=None,
        attn_mask=None,
    ):
        assert z is not None or noise is not None
        assert z is None or noise is None
        float_mask = mask.to(hidden_states.dtype).unsqueeze(-1)

        x = self.io.enter(hidden_states, mask)

        # get z
        mu = (
            float_mask *
            self.z_scale *
            self.z_proj(x)
        )

        if z is None:
            z = mu + noise
        z = z * float_mask

        # apply transformer
        q, k, v, mlp_gate, mlp_val = self.up(x, z)
        attn_out = self.attention(
            q, k, v,
            attention_mask=attn_mask
        )
        mlp_out = self.mlp(mlp_gate, mlp_val)

        hidden_states = self.io.exit(
            hidden_states,
            self.down(z, attn_out, mlp_out),
            mask
        )

        return hidden_states, z, mu
    

class HLmDecoderLayer(nn.Module):

    def __init__(self, config: HLmConfig, layer_idx: int):
        super().__init__()

        # basic shapes
        self.hidden_size = config.hidden_size
        self.qkv_size = config.attention_head_size * config.num_attention_heads
        self.mlp_size = config.mlp_size
        
        # norm
        self.io = ReZeroIO(config.hidden_size, config.norm_eps)

        # transformer projections
        self.up = FusedLinear(
            self.hidden_size,
            [self.qkv_size]*3 + [config.mlp_size]*2,
            bias=False
        )
        self.down = FusedLinear(
            [self.qkv_size, config.mlp_size],
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
            position_scale=(config.patch_size if hasattr(config, 'patch_size') else 1.0)
        )
        self.mlp = GLU(config.hidden_act)


    def forward(
        self,
        hidden_states,
        attn_mask=None,
    ):

        x = self.io.enter(hidden_states)

        # apply transformer
        q, k, v, mlp_gate, mlp_val = self.up(x)
        attn_out = self.attention(
            q, k, v,
            attention_mask=attn_mask
        )
        mlp_out = self.mlp(mlp_gate, mlp_val)

        hidden_states = self.io.exit(
            hidden_states,
            self.down(attn_out, mlp_out),
        )

        return hidden_states


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
        
        # mask out conditionals
        attn_mask = torch.zeros(1, 1, seq_len, seq_len, device=input_ids.device, dtype=hidden_states.dtype)
        attn_mask = torch.where(
            mask.unsqueeze(1).unsqueeze(1), # [bs, 1=head, 1=q, seq_len=k]
            torch.zeros_like(attn_mask),
            torch.full_like(attn_mask, float('-inf'))
        )
        attn_mask = self.layers[0].get_iaf_attn_mask(attn_mask)
        attn_mask = attn_mask.detach()

        # pad noise for last layer iaf heads
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
            torch.stack(zs, dim=2),
            torch.stack(mus, dim=2),
            torch.stack(sigmas, dim=2)
        )


class HLmGenerator(nn.Module):

    def __init__(self, config: HLmConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.z_size = config.z_size
        self.num_layers = config.num_layers

        self.embs = nn.Embedding(1+config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([
            HLmGeneratorLayer(config, i)
            for i in range(config.num_layers)
        ])


    def forward(
        self,
        input_ids,
        mask,
        z,
        num_uncond=None
    ):
        bs, seq_len = input_ids.shape
        long_mask = mask.long()
        
        # mask is zero, otherwise keep conditionals
        hidden_states = torch.where(
            mask.unsqueeze(-1),
            self.embs(torch.zeros_like(input_ids)),
            self.embs(input_ids+1)
        )

        # mask out conditionals for uncond sequences
        attn_mask = torch.zeros(bs, 1, seq_len, seq_len, device=input_ids.device, dtype=hidden_states.dtype)
        if num_uncond is not None:
            attn_mask[:num_uncond] = torch.where(
                mask[:num_uncond].unsqueeze(1).unsqueeze(1), # [bs, 1=head, 1=q, seq_len=k]
                torch.zeros_like(attn_mask[:num_uncond]),
                torch.full_like(attn_mask[:num_uncond], float('-inf'))
            )
        attn_mask = attn_mask.detach()

        mus = []
        for i, layer in enumerate(self.layers):
            
            hidden_states, _, mu = layer(
                hidden_states,
                long_mask,
                z=z[:, :, i],
                attn_mask=attn_mask
            )

            mus.append(mu)

        return torch.stack(mus, dim=2)
    

class HLmDecoder(nn.Module):

    lm_head_type = LmHead


    def __init__(self, config: HLmConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.z_output_size = config.z_output_layers * config.z_size
        self.num_layers = config.num_decoder_layers

        self.z_proj = nn.Linear(self.z_output_size, config.hidden_size, bias=False)

        self.layers = nn.ModuleList([
            HLmDecoderLayer(config, i)
            for i in range(self.num_layers)
        ])

        self.lm_head = self.lm_head_type(config)

        self.gradient_checkpointing = False


    def forward(
        self,
        mask,
        z_out
    ):
        bs, seq_len = mask.shape[:2]
        
        # get hidden states based on z
        hidden_states = self.z_proj(z_out)

        # get attention mask (remove conditionals)
        attn_mask = torch.zeros(1, 1, seq_len, seq_len, device=mask.device, dtype=hidden_states.dtype)
        attn_mask = torch.where(
            mask.unsqueeze(1).unsqueeze(1), # [bs, 1=head, 1=q, seq_len=k]
            torch.zeros_like(attn_mask),
            torch.full_like(attn_mask, float('-inf'))
        ).detach()

        for i, layer in enumerate(self.layers):
            
            if self.gradient_checkpointing and constants.XLA_AVAILABLE:
                hidden_states = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attn_mask
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attn_mask
                )
        
        if self.gradient_checkpointing and constants.XLA_AVAILABLE:
            lm_logits = self._gradient_checkpointing_func(
                self.lm_head.__call__,
                hidden_states
            )
        else:
            lm_logits = self.lm_head(hidden_states)

        return lm_logits


class HLmModel(XLAModel):

    config_class = HLmConfig


    def _init_weights(self, module):

        if hasattr(module, 'special_inited') and module.special_inited:
            return
        
        if hasattr(module, 'special_init'):
            module.special_init(self.config)

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(0.0, 1/np.sqrt(module.weight.shape[1]))
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(0.0, 1.0)


    def __init__(self, config: HLmConfig, fast_start=False):
        super().__init__(config, fast_start=fast_start)

        self.z_size = config.z_size
        self.num_layers = config.num_layers
        self.z_output_layers = config.z_output_layers
        self.z_output_size = config.z_output_layers * config.z_size

        self.reverse_z = config.reverse_z
        assert self.reverse_z is not None

        self.encoder = HLmEncoder(config)
        self.generator = HLmGenerator(config)
        self.decoder = HLmDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids,
        mask,
        num_uncond=None
    ):
        if num_uncond is not None:
            assert num_uncond > 0
            assert num_uncond <= input_ids.shape[0]

        bs, seq_len = input_ids.shape

        # sample noise for the encoder
        noise = torch.randn(
            [bs, seq_len, self.num_layers, self.z_size],
            device=input_ids.device, dtype=self.encoder.embs.weight.dtype
        )

        # pass through the encoder
        z, enc_mu, enc_sigma = self.encoder(input_ids, mask, noise)
        
        # flip z if needed
        if self.reverse_z:
            z = z.flip(2)
            enc_mu = enc_mu.flip(2)
            enc_sigma = enc_sigma.flip(2)

        # pass through the generator
        gen_mu = self.generator(input_ids, mask, z, num_uncond=num_uncond)

        # get z for the decoder
        z_out = z[:, :, -self.z_output_layers:].view(bs, seq_len, self.z_output_size)

        # pass through the decoder
        lm_logits = self.decoder(mask, z_out)

        kl = (
            -torch.log(enc_sigma)
            + 0.5 * (enc_sigma**2 + (enc_mu-gen_mu)**2)
            - 0.5
        ).sum(-1).sum(-1)

        if num_uncond is not None:
            uncond_kl = kl[:num_uncond]
            kl = kl[num_uncond:]
            return lm_logits, kl, uncond_kl

        return lm_logits, kl
