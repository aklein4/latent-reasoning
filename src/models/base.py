from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.xla import XLAConfig, XLAModel
from utils.model_utils import (
    RotaryAttention,
    GLU
)


class BaseConfig(XLAConfig):
    """
    Base configuration class for experiments.

    Args:
        vocab_size (`int`):
            Vocabulary size of the model. Defines the number of different tokens that
            can be represented by the `inputs_ids`.
        max_sequence_length (`int`):
            The maximum sequence length that this model might ever be used with.
        hidden_size (`int`):
            Number of hidden layers in the Transformer decoder.
        mlp_size (`int`):
            Dimension of the MLP representations.
        num_layers (`int`):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`):
            Number of attention heads for each attention layer in the Transformer encoder.
        use_bias (`bool`, *optional*, defaults to `True`):
            Whether or not the model should use bias for internal layers.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string).
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the normalization layers.
        use_rope (`bool`, *optional*, defaults to `False`):
            Whether or not to use the RoPE embeddings.
        rope_fraction (`int`, *optional*, defaults to 1):
            The fraction of the hidden size to use for the RoPE embeddings.
        rope_base (`float`, *optional*, defaults to `10000.0`):
            The base period of the RoPE embeddings.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing
             all weight matrices.
        identity_init(`bool`, *optional*, defaults to `False`):
            Whether or not to initialize the model with identity blocks.
        bos_token_id (int, *optional*, defaults to 0):
            The id of the `BOS` token in the vocabulary.
        eos_token_id (int, *optional*, defaults to 0):
            The id of the `EOS` token in the vocabulary.
        pad_token_id (int, *optional*, defaults to 0):
            The id of the `PAD` token in the vocabulary.
        ignore_segment_ids (`bool`, *optional*, defaults to `False`):
            Whether or not to ignore the segment ids in transformer.
        gradient_checkpointing_layers (`int`, *optional*, defaults to 1000000):
            The number of layers to checkpoint in the model.
    """

    model_type = 'base'

    def __init__(
        self,
        vocab_size,
        max_sequence_length,
        hidden_size,
        mlp_size,
        num_layers,
        num_attention_heads,
        use_bias=True,
        hidden_act="silu",
        layer_norm_eps=1e-5,
        use_rope=False,
        rope_fraction=1,
        rope_base=10000.0,
        initializer_range=0.02,
        identity_init=False,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        ignore_segment_ids=False,
        gradient_checkpointing_layers=1000000,
        *args,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length

        self.hidden_size = hidden_size
        self.mlp_size = mlp_size

        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads

        self.use_bias = use_bias
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        
        self.use_rope = use_rope
        self.rope_fraction = rope_fraction
        self.rope_base = rope_base

        self.initializer_range = initializer_range
        self.identity_init = identity_init

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        self.ignore_segment_ids = ignore_segment_ids

        self.gradient_checkpointing_layers = gradient_checkpointing_layers

        super().__init__(*args, **kwargs)


class BaseLayer(nn.Module):
  
    def __init__(self, config: BaseConfig, layer_idx: int):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.mlp_size = config.mlp_size

        self.proj_in = nn.Linear(
            self.hidden_size,
            3 * self.hidden_size + 2 * self.mlp_size,
            bias=config.use_bias
        )
        self.proj_out = nn.Linear(
            self.hidden_size + self.mlp_size,
            self.hidden_size,
            bias=False
        )

        self.attn = RotaryAttention(
            self.hidden_size,
            config.num_attention_heads,
            config.use_rope,
            config.rope_fraction,
            config.max_sequence_length,
            config.rope_base,
            layer_idx
        )
        self.mlp = GLU(config.hidden_act)

        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask=None,
        past_key_value=None,
    ):

        x = self.norm(hidden_states)
        q, k, v, gate, val = self.proj_in(x).split(
            [self.hidden_size] * 3 + [self.mlp_size] * 2,
            dim=-1
        )

        attn_out = self.attn(
            q, k, v,
            position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value
        )
        mlp_out = self.mlp(gate, val)

        y = self.proj_out(torch.cat([attn_out, mlp_out], dim=-1))

        return hidden_states + y


class BaseTransformer(nn.Module):

    layer_type = BaseLayer


    def __init__(self, config: BaseConfig):
        super().__init__()

        # info
        self.vocab_size = config.vocab_size
        self.max_sequence_length = config.max_sequence_length

        # weights
        self.vocab_embs = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.use_rope = config.use_rope
        if self.use_rope:
            self.pos_embs = None
        else:
            self.pos_embs = nn.Embedding(config.max_sequence_length, config.hidden_size)
        
        self.layers = nn.ModuleList(
            [self.layer_type(config, layer_idx) for layer_idx in range(config.num_layers)]
        )
        
        self.get_extras(config)

        # Compute configuration
        self.gradient_checkpointing = config.gradient_checkpointing
        self.gradient_checkpointing_layers = config.gradient_checkpointing_layers


    def get_extras(self, config):
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


    def _get_mask(
        self,
        input_ids: torch.LongTensor,
        segment_ids: Optional[torch.LongTensor]=None,
        mask: Optional[torch.BoolTensor]=None,
    ) -> torch.BoolTensor:
        batch_size, seq_length = input_ids.shape

        # default eager causal mask
        if mask is None:
            mask = torch.ones(seq_length, seq_length, dtype=torch.bool, device=input_ids.device)
            mask = torch.triu(mask, diagonal=1)

        # must have batch dimension
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        # apply segment ids
        if segment_ids is not None:
            segment_mask = segment_ids[:, None, :] != segment_ids[:, :, None]
            mask = mask | segment_mask            

        # fill with -infs
        mask = torch.masked_fill(
            torch.zeros_like(mask).float(),
            mask,
            float('-inf')
        )

        # head dim
        mask = mask.unsqueeze(1)

        return mask.detach()


    def _get_position_ids(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor]=None,
    ) -> torch.LongTensor:
        batch_size, seq_length = input_ids.shape
        
        # default
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=input_ids.dtype, device=input_ids.device)

        # must have batch dimension
        if len(position_ids.shape) == 1:
            position_ids = position_ids.unsqueeze(0)

        return position_ids.detach()


    def get_hidden_states(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor
    ) -> torch.Tensor:
        
        hidden_states = self.vocab_embs(input_ids)
        
        if not self.use_rope:
            hidden_states = hidden_states + self.pos_embs(position_ids)

        return hidden_states


    def get_output(
        self,
        hidden_states
    ): 
        return self.norm(hidden_states)


    def forward(
        self,
        input_ids: torch.LongTensor,
        segment_ids: Optional[torch.LongTensor]=None,
        position_ids: Optional[torch.LongTensor]=None,
        attention_mask: Optional[torch.BoolTensor]=None,
        kv=None,
    ):

        # get inputs
        position_ids = self._get_position_ids(input_ids, position_ids)
        attention_mask = self._get_mask(input_ids, segment_ids, attention_mask)
        hidden_states = self.get_hidden_states(input_ids, position_ids)

        # run transformer
        for idx, layer in enumerate(self.layers):

            if self.gradient_checkpointing and self.training and idx < self.gradient_checkpointing_layers:
                if kv is not None:
                    raise ValueError("Gradient checkpointing is not compatible with cache!")

                raise NotImplementedError("Gradient checkpointing not implemented for this model!")
                hidden_states = fast_checkpoint(
                    layer.__call__,
                    hidden_states,
                    position_ids,
                    attention_mask,
                    None
                )

            else:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_value=kv,
                )

        return self.get_output(hidden_states)


class BaseLmModel(XLAModel):

    transformer_type = BaseTransformer


    # from StableLM
    def _init_weights(self, module):
        std = self.config.initializer_range

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


    def __init__(self, config: BaseConfig, fast_start=False):
        super().__init__(config, fast_start=fast_start)

        # transformer
        self.model = self.transformer_type(config)

        # lm modeling
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # for training
        self.ignore_segment_ids = config.ignore_segment_ids

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: torch.LongTensor,
        segment_ids: Optional[torch.LongTensor]=None,
        position_ids: Optional[torch.LongTensor]=None,
        attention_mask: Optional[torch.BoolTensor]=None,
        kv=None,
    ):
        if self.ignore_segment_ids:
            segment_ids = None

        # get lm predictions
        out = self.model(
            input_ids=input_ids,
            segment_ids=segment_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            kv=kv
        )

        lm_logits = self.lm_head(out)
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        return lm_logits
