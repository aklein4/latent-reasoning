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
        z_size (`int`):
            The size of the latent space.
        disable_fiter (`bool`):
            Whether or not to disable the stream filter.
        debug (`bool`):
            Whether or not to run in debug mode, to test if the decoder can see encoder information.
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
        z_size=None,
        disable_filter=False,
        debug=False,
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

        self.z_size = z_size

        self.disable_filter = disable_filter
        self.debug = debug

        super().__init__(*args, **kwargs)


class SwiftLayer(nn.Module):

    def special_init(self, config: SwiftConfig):
        if config.debug:
            return

        self.cross_proj.weight.data.zero_()
        if self.cross_proj.bias is not None:
            self.cross_proj.bias.data.zero_()
        
        self.cross_proj.special_inited = True


    def __init__(self, config: SwiftConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.disable_filter = config.disable_filter
        self.debug = config.debug

        self.hidden_size = config.hidden_size
        self.mlp_size = config.mlp_size
        self.qkv_size = config.attention_head_size * config.num_attention_heads
        self.z_size = config.z_size

        self.up = FusedLinear(
            self.hidden_size,
            [self.qkv_size]*3 + [self.mlp_size]*2 + [self.z_size]*3,
            bias=False
        )
        self.down = FusedLinear(
            [self.qkv_size, self.mlp_size, self.z_size],
            self.hidden_size,
            bias=False,
        )

        self.attn = RotaryAttention(
            self.hidden_size,
            config.attention_head_size,
            config.num_attention_heads,
            config.num_registers,
            config.use_rope,
            config.rope_fraction,
            config.max_sequence_length,
            config.rope_base,
            self.layer_idx
        )
        self.mlp = GLU(config.hidden_act)

        self.norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False)
        self.norm_scale = nn.Parameter(torch.zeros(2, 1, 1, self.hidden_size))
        self.norm_bias = nn.Parameter(torch.zeros(2, 1, 1, self.hidden_size))
        self.cross_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.z_scale = 1 / np.sqrt(config.num_layers * self.z_size)

        k = torch.linspace(0, 1, self.hidden_size)
        stream_filter = torch.maximum(
            torch.exp(-2 * np.log(config.num_layers+1) * k),
            torch.full_like(k, 1 / (layer_idx + 2))
        )
        stream_filter = stream_filter[None, None, :]
        self.register_buffer('stream_filter', stream_filter, persistent=True)


    def forward(
        self,
        hidden_states,
        z: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask=None,
        past_key_value=None,
    ):
        bs = hidden_states.shape[0]//2

        normed = self.norm(hidden_states.view(2, bs, *hidden_states.shape[-2:]))
        x = normed * (self.norm_scale + 1) + self.norm_bias
        x[0] = x[0] + self.cross_proj(normed[1]) # encoder gets info from decoder
        x = x.view(*hidden_states.shape)

        q, k, v, gate, up, shift, mu, log_sigma = self.up(x)

        shift = shift[bs:] # decoder is second half
        mu = mu[:bs] * self.z_scale # encoder is first half
        log_sigma = log_sigma[:bs] * self.z_scale
        sigma = F.softplus(log_sigma + np.log(np.e - 1))

        attn_out = self.attn(
            q, k, v,
            position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value
        )
        mlp_out = self.mlp(gate, up)
        z_out = (shift + mu + sigma * z).repeat(2, 1, 1)

        if self.debug:
            z_out = torch.zeros_like(z_out)

        y = self.down(attn_out, mlp_out, z_out)

        return self.update_stream(hidden_states, y), mu, sigma


    def update_stream(self, hidden_states, y):
        if self.disable_filter:
            return y + hidden_states

        return (
            self.stream_filter * y +
            (1 - self.stream_filter) * hidden_states
        )


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

        # info
        self.vocab_size = config.vocab_size
        self.max_sequence_length = config.max_sequence_length

        # embeddings
        self.vocab_embs = nn.Embedding(config.vocab_size+1, config.hidden_size)
        self.encoder_embs = nn.Embedding(2, config.hidden_size)
        self.decoder_embs = nn.Embedding(2, config.hidden_size)

        # positional embeddings
        assert config.use_rope is not None
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
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def _get_position_ids(self, input_ids: torch.LongTensor):
        return torch.arange(input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device)[None]


    def get_hidden_states(self, input_ids, position_ids, mask):
        mask = mask.long()
        input_ids = input_ids + 1

        encoder_states = self.vocab_embs(input_ids)
        if self.use_rope:
            encoder_states = encoder_states/2 + self.encoder_embs(mask)/2
        else:
            encoder_states = encoder_states/3 + self.encoder_embs(mask)/3 + self.pos_embs(position_ids)/3

        input_ids = torch.where(mask.bool(), torch.zeros_like(input_ids), input_ids)
        decoder_states = self.vocab_embs(input_ids)
        if self.use_rope:
            decoder_states = decoder_states/2 + self.decoder_embs(mask)/2
        else:
            decoder_states = decoder_states/3 + self.decoder_embs(mask)/3 + self.pos_embs(position_ids)/3

        return torch.cat([encoder_states, decoder_states], dim=0)


    def forward(
        self,
        input_ids,
        mask
    ):
        bs, seq_len = input_ids.shape

        position_ids = self._get_position_ids(input_ids)
        hidden_states = self.get_hidden_states(input_ids, position_ids, mask)

        z = torch.randn(
            [self.num_layers, bs, seq_len, self.config.z_size],
            device=input_ids.device, dtype=hidden_states.dtype
        )

        mus = []
        sigmas = []
        for idx, layer in enumerate(self.layers):
            
            hidden_states, m, s = layer(
                hidden_states, z[idx], position_ids
            )
            mus.append(m)
            sigmas.append(s)
        
        mus = torch.stack(mus, dim=0)
        sigmas = torch.stack(sigmas, dim=0)

        decoder_states = hidden_states.chunk(2, dim=0)[1]
        lm_logits = self.lm_head(self.norm(decoder_states))
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        return lm_logits, mus, sigmas
