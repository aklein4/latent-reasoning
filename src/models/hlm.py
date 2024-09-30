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
        logits = self.lm_head(self.norm(hidden_states))
        return F.log_softmax(logits, dim=-1)


class HLmEncGenLayer(nn.Module):

    def __init__(self, config: HLmConfig, layer_idx: int):
        super().__init__()

        # basic shapes
        self.hidden_size = config.hidden_size
        self.qkv_size = config.attention_head_size * config.num_attention_heads
        self.mlp_size = config.mlp_size
        self.z_size = config.z_size
        
        # attention shapes
        self.attention_head_size = config.attention_head_size
        self.num_registers = config.num_registers
        self.num_attn_heads = config.num_attention_heads
        self.num_iaf_heads = config.num_iaf_attention_heads
        self.num_bid_heads = self.num_attn_heads - self.num_iaf_heads

        # attention components
        self.enc_io_attn = ReZeroIO(config.hidden_size, config.norm_eps)
        self.gen_io_attn = ConditionalIO(config.hidden_size, config.norm_eps)

        self.enc_attn_up = FusedLinear(
            [self.hidden_size, self.z_size],
            [self.qkv_size, 2*self.qkv_size],
            bias=True,
            mask=self._get_qkv_matrix_mask()
        )
        self.gen_attn_up = FusedLinear(
            self.hidden_size,
            [self.qkv_size, 2*self.qkv_size],
            bias=True
        )

        self.enc_registers = nn.Parameter(torch.randn(1, config.num_registers, 2*self.qkv_size))
        self.gen_registers = nn.Parameter(torch.randn(1, config.num_registers, 2*self.qkv_size))

        self.attention = RotaryAttention(
            config.attention_head_size,
            config.num_attention_heads,
            0,
            True,
            config.rope_fraction,
            config.max_sequence_length,
            config.rope_base,
            layer_idx,
            position_scale=(config.patch_size if hasattr(config, 'patch_size') else 1.0)
        )

        self.enc_attn_down = nn.Linear(self.qkv_size, self.hidden_size, bias=False)
        self.gen_attn_down = nn.Linear(self.qkv_size, self.hidden_size, bias=False)

        # mlp components
        self.enc_io_mlp = ReZeroIO(config.hidden_size, config.norm_eps)
        self.gen_io_mlp = ConditionalIO(config.hidden_size, config.norm_eps)

        self.enc_mlp_up = FusedLinear(
            self.hidden_size,
            [self.mlp_size]*2,
            bias=False
        )
        self.gen_mlp_up = FusedLinear(
            self.hidden_size,
            [self.mlp_size]*2,
            bias=False
        )

        self.mlp = GLU(config.hidden_act)

        self.enc_mlp_down = nn.Linear(self.mlp_size, self.hidden_size, bias=False)
        self.gen_mlp_down = nn.Linear(self.mlp_size, self.hidden_size, bias=False)

        # z components
        self.enc_io_z = ReZeroIO(config.hidden_size, config.norm_eps)
        self.gen_io_z = ConditionalIO(config.hidden_size, config.norm_eps)

        self.enc_z_proj = GaussianIAF(
            self.hidden_size,
            self.z_size,
            config.z_mlp_mult,
            config.hidden_act
        )
        self.gen_z_proj = nn.Linear(
            self.hidden_size,
            self.z_size, 
            bias=False
        )

        self.enc_z_down = nn.Linear(
            self.z_size,
            self.hidden_size,
            bias=False
        )
        self.gen_z_down = nn.Linear(
            self.z_size,
            self.hidden_size,
            bias=False
        )

        # z scale
        self.z_scale = np.sqrt(
            (config.patch_size if hasattr(config, 'patch_size') else 1.0) /
            (config.z_size * config.num_layers)
        )


    @torch.no_grad()
    def _get_qkv_matrix_mask(self):
        
        # hidden states can apply to anything
        hidden_mask = torch.ones(3*self.qkv_size, self.hidden_size)

        # noise can ONLY apply to iaf heads k and v
        noise_q_mask = torch.zeros(self.qkv_size, self.z_size)
        
        noise_iaf_mask = torch.ones(self.num_iaf_heads*self.attention_head_size, self.z_size)
        noise_bid_mask = torch.zeros(self.num_bid_heads*self.attention_head_size, self.z_size)
        noise_kv_mask = torch.cat([noise_iaf_mask, noise_bid_mask], dim=0)
        noise_kv_mask = noise_kv_mask.repeat(2, 1)

        noise_mask = torch.cat([noise_q_mask, noise_kv_mask], dim=0)

        return torch.cat([hidden_mask, noise_mask], dim=1)


    @torch.no_grad()
    def get_encoder_attn_mask(self, attn_mask):
        # expand the mask to number of heads
        attn_mask = attn_mask.expand(-1, self.num_attn_heads, -1, -1).clone()

        # iaf heads can not attend to themselves
        iaf_mask = torch.full_like(attn_mask, float('-inf'))
        attn_mask[:, :self.num_iaf_heads] += torch.triu(iaf_mask[:, :self.num_iaf_heads], diagonal=0)

        # everything can attend to the registers
        return torch.cat(
            [
                attn_mask,
                torch.zeros_like(attn_mask[:, :, :, :1]).expand(-1, -1, -1, self.num_registers)
            ],
            dim=-1
        )


    @torch.no_grad()
    def get_generator_attn_mask(self, attn_mask):
        # expand the mask to number of heads
        attn_mask = attn_mask.expand(-1, self.num_attn_heads, -1, -1).clone()

        # everything can attend to the registers
        return torch.cat(
            [
                attn_mask,
                torch.zeros_like(attn_mask[:, :, :, :1]).expand(-1, -1, -1, self.num_registers)
            ],
            dim=-1
        )
        

    def forward(
        self,
        encoder_states,
        generator_states,
        mask,
        noise,
        encoder_attn_mask,
        generator_attn_mask
    ):
        float_mask = mask.to(encoder_states.dtype).unsqueeze(-1)
        noise = noise * float_mask # noise becomes zero where not used

        # perform attention
        enc_q, enc_kv = self.enc_attn_up(self.enc_io_attn.enter(encoder_states), noise)
        gen_q, gen_kv = self.gen_attn_up(self.gen_io_attn.enter(generator_states, mask))
        
        # add registers
        enc_kv = torch.cat([enc_kv, self.enc_registers.expand(enc_kv.shape[0], -1, -1)], dim=1)
        gen_kv = torch.cat([gen_kv, self.gen_registers.expand(gen_kv.shape[0], -1, -1)], dim=1)

        # combine
        q = torch.cat([enc_q, gen_q], dim=0)
        k, v = torch.cat([enc_kv, gen_kv], dim=0).chunk(2, dim=-1)
        attn_mask = torch.cat([encoder_attn_mask, generator_attn_mask], dim=0)

        # get custom position ids for the registers
        k_pos_ids = torch.arange(encoder_states.shape[1], device=k.device, dtype=torch.long).long()[None]
        k_pos_ids = torch.cat([k_pos_ids, k_pos_ids[:, :1].expand(-1, self.num_registers)], dim=1)

        # mechanism
        enc_attn_out, gen_attn_out = self.attention(
            q, k, v,
            attention_mask=attn_mask,
            k_position_ids=k_pos_ids
        ).chunk(2, dim=0)

        # add to resiuals
        encoder_states = self.enc_io_attn.exit(
            encoder_states,
            self.enc_attn_down(enc_attn_out)
        )
        generator_states = self.gen_io_attn.exit(
            generator_states,
            self.gen_attn_down(gen_attn_out),
            mask
        )

        # perform mlp
        enc_mlp_gate, enc_mlp_val = self.enc_mlp_up(self.enc_io_mlp.enter(encoder_states))
        gen_mlp_gate, gen_mlp_val = self.gen_mlp_up(self.gen_io_mlp.enter(generator_states, mask))

        enc_mlp_out = self.mlp(enc_mlp_gate, enc_mlp_val)
        gen_mlp_out = self.mlp(gen_mlp_gate, gen_mlp_val)

        encoder_states = self.enc_io_mlp.exit(
            encoder_states,
            self.enc_mlp_down(enc_mlp_out)
        )
        generator_states = self.gen_io_mlp.exit(
            generator_states,
            self.gen_mlp_down(gen_mlp_out),
            mask
        )

        # perform z
        enc_mu, enc_log_sigma = (
            float_mask * self.z_scale *
            self.enc_z_proj(
                self.enc_io_z.enter(encoder_states),
                noise
            )
        ).chunk(2, dim=-1)
        enc_sigma = F.softplus(enc_log_sigma + np.log(np.e - 1))

        gen_mu = (
            float_mask * self.z_scale *
            self.gen_z_proj(
                self.gen_io_z.enter(generator_states, mask)
            )
        )

        z = enc_mu + enc_sigma * noise

        encoder_states = self.enc_io_z.exit(
            encoder_states,
            self.enc_z_down(z)
        )
        generator_states = self.gen_io_z.exit(
            generator_states,
            self.gen_z_down(gen_mu),
            mask
        )

        return z, encoder_states, generator_states, enc_mu, enc_sigma, gen_mu
    

class HLmDecoderLayer(nn.Module):

    def __init__(self, config: HLmConfig, layer_idx: int):
        super().__init__()

        # basic shapes
        self.hidden_size = config.hidden_size
        qkv_size = config.attention_head_size * config.num_attention_heads
        
        # attention components
        self.attn_io = ReZeroIO(config.hidden_size, config.norm_eps)

        self.attn_up = FusedLinear(
            self.hidden_size,
            [qkv_size]*3,
            bias=True
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

        self.attn_down = nn.Linear(qkv_size, self.hidden_size, bias=False)

        # mlp components
        self.mlp_io = ReZeroIO(config.hidden_size, config.norm_eps)

        self.mlp_up = FusedLinear(
            self.hidden_size,
            [config.mlp_size, config.mlp_size],
            bias=False
        )

        self.mlp = GLU(config.hidden_act)

        self.mlp_down = nn.Linear(config.mlp_size, self.hidden_size, bias=False)


    def forward(
        self,
        hidden_states,
        attn_mask=None,
    ):
        
        # perform attention
        q, k, v = self.attn_up(self.attn_io.enter(hidden_states))

        attn_out = self.attention(
            q, k, v,
            attention_mask=attn_mask
        )
        
        hidden_states = self.attn_io.exit(
            hidden_states,
            self.attn_down(attn_out)
        )

        # perform mlp
        mlp_gate, mlp_val = self.mlp_up(self.mlp_io.enter(hidden_states))

        mlp_out = self.mlp(mlp_gate, mlp_val)

        hidden_states = self.mlp_io.exit(
            hidden_states,
            self.mlp_down(mlp_out)
        )

        return hidden_states


class HLmEncGen(nn.Module):

    def init_input_params(self, config: HLmConfig):
        self.enc_embs = nn.Embedding(config.vocab_size, config.hidden_size)
        self.gen_embs = nn.Embedding(1+config.vocab_size, config.hidden_size)


    def __init__(self, config: HLmConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.z_size = config.z_size
        self.num_layers = config.num_layers

        self.init_input_params(config)

        self.layers = nn.ModuleList([
            HLmEncGenLayer(config, i)
            for i in range(config.num_layers)
        ])

        self.gradient_checkpointing = False


    def get_states(self, input_ids, mask):
        encoder_states = self.enc_embs(input_ids)
        generator_states = torch.where(
            mask.unsqueeze(-1),
            self.gen_embs(torch.zeros_like(input_ids)),
            self.gen_embs(input_ids+1)
        )

        return encoder_states, generator_states

    
    def forward(
        self,
        input_ids,
        mask,
        noise,
        num_uncond=None
    ):
        bs, seq_len = input_ids.shape[:2]
        long_mask = mask.long()
        
        # get hidden states
        encoder_states, generator_states = self.get_states(input_ids, mask)

        # get encoder attention mask, (remove conditionals)
        enc_attn_mask = torch.zeros(bs, 1, seq_len, seq_len, device=input_ids.device, dtype=encoder_states.dtype)
        enc_attn_mask = torch.where(
            mask.unsqueeze(1).unsqueeze(1), # [bs, 1=head, 1=q, seq_len=k]
            torch.zeros_like(enc_attn_mask),
            torch.full_like(enc_attn_mask, float('-inf'))
        )
        enc_attn_mask = self.layers[0].get_encoder_attn_mask(enc_attn_mask)
        enc_attn_mask = enc_attn_mask.detach()

        # get generator attention mask, (remove conditionals for uncond sequences)
        gen_attn_mask = torch.zeros(bs, 1, seq_len, seq_len, device=input_ids.device, dtype=generator_states.dtype)
        if num_uncond is not None:
            gen_attn_mask[:num_uncond] = torch.where(
                mask[:num_uncond].unsqueeze(1).unsqueeze(1), # [bs, 1=head, 1=q, seq_len=k]
                torch.zeros_like(gen_attn_mask[:num_uncond]),
                torch.full_like(gen_attn_mask[:num_uncond], float('-inf'))
            )
        gen_attn_mask = self.layers[0].get_generator_attn_mask(gen_attn_mask)
        gen_attn_mask = gen_attn_mask.detach()

        zs = []
        enc_mus = []
        enc_sigmas = []
        gen_mus = []
        for i, layer in enumerate(self.layers):
            
            if self.gradient_checkpointing and constants.XLA_AVAILABLE:
                z, encoder_states, generator_states, enc_mu, enc_sigma, gen_mu = self._gradient_checkpointing_func(
                    layer.__call__,
                    encoder_states,
                    generator_states,
                    long_mask,
                    noise[:, :, i],
                    enc_attn_mask,
                    gen_attn_mask
                )
            else:
                z, encoder_states, generator_states, enc_mu, enc_sigma, gen_mu = layer(
                    encoder_states,
                    generator_states,
                    long_mask,
                    noise[:, :, i],
                    enc_attn_mask,
                    gen_attn_mask
                )

            zs.append(z)
            enc_mus.append(enc_mu)
            enc_sigmas.append(enc_sigma)
            gen_mus.append(gen_mu)
        
        return (
            torch.stack(zs, dim=2),
            torch.stack(enc_mus, dim=2),
            torch.stack(enc_sigmas, dim=2),
            torch.stack(gen_mus, dim=2)
        )
    

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

        self.enc_gen = HLmEncGen(config)
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
            device=input_ids.device, dtype=self.enc_gen.enc_embs.weight.dtype
        )

        # pass through the encoder
        z, enc_mu, enc_sigma, gen_mu = self.enc_gen(input_ids, mask, noise, num_uncond=num_uncond)

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
