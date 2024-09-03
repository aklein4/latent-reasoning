import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.xla import XLAConfig, XLAModel
from utils.model_utils import (
    SeperatedLayerNorm,
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
        num_layers (`int`):
            Number of hidden layers in the Transformer decoder.
        use_bias (`bool`):
            Whether or not the model should use bias for internal layers.
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
        z_size (`int`):
            The size of the latent space.
        z_init_scale (`float`):
            The scale of the initial z weights.
        base_gate (`float`):
            The base gate value for the highway residual.
        gate_init_scale (`float`):
            The scale of the initial gate weights.
    """

    model_type = 'base'

    def __init__(
        self,
        hidden_size,
        mlp_size,
        attention_head_size,
        num_attention_heads,
        num_layers,
        use_bias,
        hidden_act,
        layer_norm_eps,
        use_rope,
        rope_fraction,
        rope_base,
        z_size,
        # z_init_scale,
        base_gate,
        gate_init_scale,   
        *args,
        **kwargs,
    ):

        self.hidden_size = hidden_size
        self.mlp_size = mlp_size
        self.attention_head_size = attention_head_size
        self.num_attention_heads = num_attention_heads

        self.num_layers = num_layers

        self.use_bias = use_bias
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        
        self.use_rope = use_rope
        self.rope_fraction = rope_fraction
        self.rope_base = rope_base

        self.z_size = z_size
        # self.z_init_scale = z_init_scale

        self.base_gate = base_gate
        self.gate_init_scale = gate_init_scale

        super().__init__(*args, **kwargs)


class SwiftLayer(nn.Module):
  

    def special_init(self, config):
        
        self.proj_in.linear.weight.data.normal_(0.0, 1/np.sqrt(self.proj_in.total_in))
        if self.proj_in.linear.bias is not None:
            self.proj_in.linear.bias.data.zero_()
        
        self.proj_out.linear.weight.data.normal_(0.0, 1/np.sqrt(self.proj_out.total_in))
        if self.proj_out.linear.bias is not None:
            self.proj_out.linear.bias.data.zero_()

        # self.proj_in.linear.weight.data[-2*config.z_size:] *= config.z_init_scale
        self.proj_out.linear.weight.data[-self.hidden_size//2:] *= config.gate_init_scale

        self.proj_in.special_inited = True
        self.proj_out.special_inited = True
        self.proj_in.linear.special_inited = True
        self.proj_out.linear.special_inited = True


    def __init__(self, config: SwiftConfig, layer_idx: int):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.mlp_size = config.mlp_size
        self.qkv_size = config.attention_head_size * config.num_attention_heads
        self.z_size = config.z_size

        assert self.hidden_size % 2 == 0, "Hidden size must be divisible by 2!"

        self.proj_in = FusedLinear(
            self.hidden_size,
            [self.qkv_size]*3 + [self.mlp_size]*2 + [self.z_size]*3,
            bias=config.use_bias
        )
        self.proj_out = FusedLinear(
            [self.qkv_size, self.mlp_size, self.z_size],
            [self.hidden_size//2]*3,
            bias=False
        )

        self.attn = RotaryAttention(
            self.hidden_size,
            config.attention_head_size,
            config.num_attention_heads,
            config.use_rope,
            config.rope_fraction,
            config.max_sequence_length,
            config.rope_base,
            layer_idx
        )
        self.mlp = GLU(config.hidden_act)

        self.enc_norm = SeperatedLayerNorm(2, config.hidden_size, eps=config.layer_norm_eps)
        self.dec_norm = SeperatedLayerNorm(2, config.hidden_size, eps=config.layer_norm_eps)

        base_gate = max(1 / (layer_idx+2), config.base_gate)
        self.log_base_gate = np.log(base_gate / (1 - base_gate))

        self.z_scale = 1 / np.sqrt(config.num_layers * self.z_size)


    def forward(
        self,
        encoder_states: torch.Tensor,
        decoder_states: torch.Tensor,
        z: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask=None,
        past_key_value=None,
    ):

        enc_attn, enc_mlp, enc_shift, enc_mu, enc_sigma = self.stage_1(
            self.enc_norm(torch.cat(encoder_states, dim=-1)), position_ids, attention_mask, past_key_value
        )
        dec_attn, dec_mlp, dec_shift, dec_mu, dec_sigma = self.stage_1(
            self.dec_norm(torch.cat(decoder_states, dim=-1)), position_ids, attention_mask, past_key_value
        )

        z_out = ((dec_shift + enc_sigma) + enc_sigma * z)

        encoder_states = self.state_2(encoder_states, enc_attn, enc_mlp, z_out)
        decoder_states = self.state_2(decoder_states, dec_attn, dec_mlp, z_out)

        return encoder_states, decoder_states, enc_mu, enc_sigma


    def stage_1(self, x, position_ids, attention_mask=None, past_key_value=None):
        
        q, k, v, gate, val, shift, mu, log_sigma = self.proj_in(x)
        
        mu = self.z_scale * mu
        sigma = F.softplus(self.z_scale * log_sigma + np.log(np.e - 1))

        attn_out = self.attn(
            q, k, v,
            position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value
        )
        mlp_out = self.mlp(gate, val)

        return attn_out, mlp_out, shift, mu, sigma


    def state_2(self, x, attn, mlp, z):

        y_gate, y_res, log_gate = self.proj_out(attn, mlp, z)
        gate = torch.sigmoid(log_gate + self.log_base_gate)

        gated, res = x
        gated = gate * y_gate + (1 - gate) * gated
        res = y_res + res

        return gated, res


class SwiftModel(XLAModel):


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

        # info
        self.vocab_size = config.vocab_size
        self.max_sequence_length = config.max_sequence_length

        # embeddings
        self.vocab_embs = nn.Embedding(config.vocab_size+1, config.hidden_size)
        self.encoder_embs = nn.Embedding(2, config.hidden_size)
        self.decoder_embs = nn.Embedding(2, config.hidden_size)

        # positional embeddings
        self.use_rope = config.use_rope
        if self.use_rope:
            self.pos_embs = None
        else:
            self.pos_embs = nn.Embedding(config.max_sequence_length, config.hidden_size)
        
        # layers
        self.num_layers = config.num_layers
        self.layers = nn.ModuleList(
            [SwiftLayer(config, layer_idx) for layer_idx in range(config.num_layers)]
        )

        # lm modeling
        self.norm = SeperatedLayerNorm(2, config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def _get_position_ids(self, input_ids: torch.LongTensor):
        return torch.arange(input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device)[None]


    def get_hidden_states(self, input_ids, position_ids, mask):
        mask = mask.long()
        input_ids = input_ids + 1

        hidden_states = self.vocab_embs(input_ids)
        if self.use_rope:
            encoder_states = hidden_states/2 + self.encoder_embs(mask)/2
        else:
            encoder_states = hidden_states/3 + self.encoder_embs(mask)/3 + self.pos_embs(position_ids)/3

        input_ids = torch.where(mask.bool(), torch.zeros_like(input_ids), input_ids)
        hidden_states = self.vocab_embs(input_ids)
        if self.use_rope:
            decoder_states = hidden_states/2 + self.decoder_embs(mask)/2
        else:
            decoder_states = hidden_states/3 + self.decoder_embs(mask)/3 + self.pos_embs(position_ids)/3

        encoder_states = tuple(torch.chunk(encoder_states, 2, dim=-1))
        decoder_states = tuple(torch.chunk(decoder_states, 2, dim=-1))

        return encoder_states, decoder_states


    def forward(
        self,
        input_ids,
        mask
    ):
        bs, seq_len = input_ids.shape

        position_ids = self._get_position_ids(input_ids)
        encoder_states, decoder_states = self.get_hidden_states(input_ids, position_ids, mask)

        z = torch.randn(
            [self.num_layers, bs, seq_len, self.config.z_size],
            device=input_ids.device, dtype=encoder_states[0].dtype
        )

        mus = []
        sigmas = []
        for idx, layer in enumerate(self.layers):
            
            encoder_states, decoder_states, m, s = layer(
                encoder_states, decoder_states, z[idx], position_ids
            )
            mus.append(m)
            sigmas.append(s)
        
        mus = torch.stack(mus, dim=0)
        sigmas = torch.stack(sigmas, dim=0)

        decoder_states = self.norm(torch.cat(decoder_states, dim=-1))
        lm_logits = self.lm_head(decoder_states)
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        return lm_logits, mus, sigmas
