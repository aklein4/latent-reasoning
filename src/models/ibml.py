from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from functools import partial
from dataclasses import dataclass

from transformers import PreTrainedModel, PretrainedConfig
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaAttention,
    LlamaMLP,
    LlamaPreTrainedModel,
    LlamaModel,
    LlamaForCausalLM,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPast

import utils.constants as constants
from utils.model_utils import expand_to_batch, print_gradient
from utils.dot_dict import DotDict


logger = logging.get_logger(__name__)


class IBMLConfig(PretrainedConfig):

    model_type = "ibml"
    supports_gradient_checkpointing = True


    def __init__(
        self,
        base_url: str = "meta-llama/Llama-2-7b-chat-hf",
        *args,
        **kwargs
    ):
        
        self.base_url = base_url

        super().__init__(*args, **kwargs)


class IBMLMechanism(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        
        self.Q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.K = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size, bias=False)
        self.O = nn.Linear(hidden_size, hidden_size, bias=False)

        self.w = nn.Parameter(torch.ones(hidden_size, hidden_size) / math.sqrt(hidden_size))

        self.norm = LlamaRMSNorm(hidden_size, eps=eps)

    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        memory_states: torch.FloatTensor,
        memory_mask: Optional[torch.FloatTensor] = None,
        prev_mat: Optional[torch.FloatTensor] = None,
        mat_beta: Optional[float] = 0.0,
    ):
        memory_states = memory_states.view(-1, memory_states.shape[-1])
        memory_mask = memory_mask.view(-1, 1) if memory_mask is not None else 1.0

        q = self.Q(hidden_states)
        k = self.K(memory_states) * memory_mask / math.sqrt(self.hidden_size)
        v = self.V(memory_states) * memory_mask

        mat = k.T @ v
        if prev_mat is not None:
            mat = mat_beta * prev_mat + (1 - mat_beta) * mat

        result = F.linear(q, self.w * mat)

        return self.O(self.norm(result)), mat


class IBMLDecoderLayer(nn.Module):

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()

        self.hidden_size = config.hidden_size

        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ibml_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.ibml_mech = IBMLMechanism(self.hidden_size, config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

        self.scale_inited = False
        self.ibml_scale = nn.Parameter(torch.zeros(self.hidden_size))


    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_states: torch.FloatTensor,
        memory_mask: Optional[torch.FloatTensor] = None,
        prev_mat: Optional[torch.FloatTensor] = None,
        mat_beta: Optional[float] = 0.0,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        attn_states = hidden_states.clone()
        hidden_states = residual + hidden_states

        # init the scales
        if not self.scale_inited:
            self.ibml_scale.data = attn_states.view(-1, attn_states.shape[-1]).std(dim=0).detach()
            self.scale_inited = True

        # IBML Mechanism
        residual = hidden_states
        hidden_states = self.ibml_layernorm(hidden_states)
        hidden_states, new_mat = self.ibml_mech(
            hidden_states=hidden_states,
            memory_states=memory_states,
            memory_mask=memory_mask,
            prev_mat=prev_mat,
            mat_beta=mat_beta,
        )
        hidden_states = residual + hidden_states * expand_to_batch(self.ibml_scale, hidden_states).detach()

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # Prepare outputs
        outputs = (hidden_states, new_mat)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


@dataclass
class BaseModelOutputWithMats(BaseModelOutputWithPast):

    mats: Optional[torch.FloatTensor] = None


class IBMLDecoder(LlamaModel):

    def __init__(self, config: LlamaConfig):
        LlamaPreTrainedModel.__init__(self, config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [IBMLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        memory_states: Optional[torch.FloatTensor] = None,
        memory_mask: Optional[torch.FloatTensor] = None,
        prev_mats: Optional[torch.FloatTensor] = None,
        mat_beta: Optional[float] = 0.0,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_mats = ()

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            prev_mat = prev_mats[i] if prev_mats is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    memory_states,
                    memory_mask,
                    prev_mat,
                    mat_beta,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    memory_states=memory_states,
                    memory_mask=memory_mask,
                    prev_mat=prev_mat,
                    mat_beta=mat_beta,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[2],)
            all_mats += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        all_mats = torch.stack(all_mats, dim=0)

        return BaseModelOutputWithMats(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            mats=all_mats,
        )


class IBMLModel(PreTrainedModel):

    config_class = IBMLConfig
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    

    def __init__(self, config: LlamaConfig, cpu=False):
        super().__init__(config)

        # get base model
        base_model = LlamaForCausalLM.from_pretrained(
            config.base_url,
            torch_dtype=torch.float16,
            device_map="cpu",
        ).to(torch.float32)

        # enable flash attention
        if str(constants.DEVICE) == "cuda" and not cpu:
            base_model.config._attn_implementation = "flash_attention_2"
        else:
            base_model.config._attn_implementation = "eager"

        # copy the LM parameters
        self.lm_head = base_model.lm_head
        self.lm_head.weight = nn.Parameter(self.lm_head.weight.data.clone().detach())

        # create the encoder and decoder
        self.encoder = LlamaModel(base_model.model.config)
        self.decoder = IBMLDecoder(base_model.model.config)

        # prepare the encoder
        self.encoder.load_state_dict(
            {k: v.clone().detach() for k, v in base_model.model.state_dict().items()},
            strict=True
        )
        self.encoder.norm.weight.data = torch.ones_like(self.encoder.norm.weight.data)

        # prepare the decoder
        self.decoder.load_state_dict(
            {k: v.clone().detach() for k, v in base_model.model.state_dict().items()},
            strict=False
        )

        # fix the embedding weights?
        # For some reason TinyLlama seems to have a broken start token embedding
        self.encoder.embed_tokens.weight.data[1] = self.encoder.embed_tokens.weight.data[2].clone().detach()
        self.decoder.embed_tokens.weight.data[1] = self.decoder.embed_tokens.weight.data[2].clone().detach()

        # Initialize weights and gradient checkpointing
        self.post_init()
    

    # overwrite to prevent overwriting the base model weights
    def init_weights(self):
        return

    
    def forward(
        self,
        input_ids: torch.LongTensor,
        memory_mask: Optional[torch.FloatTensor] = None,
        prev_mats: Optional[torch.FloatTensor] = None,
        mat_beta: Optional[float] = 0.0,
    ):

        memory_states = self.encoder(input_ids=input_ids).last_hidden_state
        memory_states = memory_states.reshape(-1, memory_states.shape[-1])

        outputs = self.decoder(
            input_ids=input_ids,
            memory_states=memory_states,
            memory_mask=memory_mask,
            prev_mats=prev_mats,
            mat_beta=mat_beta,
        )

        lm_logits = self.lm_head(outputs.last_hidden_state)
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        return DotDict(
            lm_logits=lm_logits,
            mats=outputs.mats,
        )
