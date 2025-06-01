from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    PreTrainedModel, PretrainedConfig,
    LlamaModel, LlamaForCausalLM
)
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm, LlamaAttention, LlamaMLP, Cache
)

import math

from utils.dot_dict import DotDict
from utils.model_utils import unsqueeze_to_batch, expand_to_batch, momentum_scan
import utils.constants as constants


class ZLmConfig(PretrainedConfig):
    """
    Configuration class for ZLM model.
    This is a subclass of LlamaConfig with additional parameters specific to ZLM.
    """

    model_type = "zlm"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        base_url: str = "meta-llama/Llama-2-7b-chat-hf",
        input_length: int = 128,
        output_length: int = 128,
        z_length: int = 512,
        latent_size: int = 128,
        fix_embeddings: bool = False,
        *args,
        **kwargs
    ):
        
        self.base_url = base_url

        self.input_length = input_length
        self.output_length = output_length
        self.z_length = z_length

        self.latent_size = latent_size

        self.fix_embeddings = fix_embeddings

        super().__init__(*args, **kwargs)


class Padder:

    def __init__(self, prefix_length, suffix_length):
        self.prefix_length = prefix_length
        self.suffix_length = suffix_length


    def pad(self, x):
        return torch.cat(
            [
                torch.zeros_like(x[..., :1, :]).expand(
                    *[-1 for _ in range(x.ndim - 2)],
                    self.prefix_length,
                    -1
                ),
                x,
                torch.zeros_like(x[..., :1, :]).expand(
                    *[-1 for _ in range(x.ndim - 2)],
                    self.suffix_length,
                    -1
                ),
            ],
            dim=-2
        )


    def unpad(self, x):
        if self.suffix_length > 0:
            return x[..., self.prefix_length:-self.suffix_length, :]
        else:
            return x[..., self.prefix_length:, :]


class ModulatingRMSNorm(nn.Module):

    def __init__(self, old_norm, strides):
        super().__init__()

        self.normalized_shape = old_norm.weight.shape
        self.num_d = len(self.normalized_shape)

        self.num_strides = len(strides)
        self.strides = strides

        self.norm = nn.RMSNorm(
            normalized_shape=self.normalized_shape,
            eps=old_norm.variance_epsilon,
            elementwise_affine=False
        )

        self.scales = nn.ParameterList(
            [
                nn.Parameter(old_norm.weight.data.clone().detach()) for _ in range(self.num_strides)
            ]
        )
        self.biases = nn.ParameterList(
            [
                nn.Parameter(torch.zeros_like(old_norm.weight.data)) for _ in range(self.num_strides)
            ]
        )

        self.index = None
        self.dropout = None
    

    def _get_scales_and_biases(self):

        out = []
        out_bias = []
        for i in range(self.num_strides):
            scale = self.scales[i]
            bias = self.biases[i]
            stride = self.strides[i]

            expanded_scale = scale[None].expand(stride, *([-1] * self.num_d))
            out.append(expanded_scale)
        
            expanded_bias = bias[None].expand(stride, *([-1] * self.num_d))
            out_bias.append(expanded_bias)

        return torch.cat(out, dim=0), torch.cat(out_bias, dim=0)


    def forward(self, x):

        x = self.norm(x)

        if self.index is None:
            scales, biases = self._get_scales_and_biases()
            scales = unsqueeze_to_batch(scales, x)
            biases = unsqueeze_to_batch(biases, x)

        else:
            scales = self.scales[self.index]
            biases = self.biases[self.index]

            scales = unsqueeze_to_batch(scales, x)
            biases = unsqueeze_to_batch(biases, x)

        if self.dropout is not None:
            x = F.dropout(x, p=self.dropout)

        return (x * scales) + biases


class ZLmPrefixLayer(nn.Module):

    def __init__(self, config, zlm_config: ZLmConfig, layer_idx: int, bi_length: int):
        super().__init__()

        self.hidden_size = config.hidden_size

        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)

        self.bi_length = bi_length
        self.bi_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.bi_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.bi_attn.is_causal = False

        self.scale_inited = False
        self.bi_scale = nn.Parameter(torch.zeros(self.hidden_size))
        

    def forward(
        self,
        hidden_states: torch.Tensor,
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
            self.bi_scale.data = attn_states.view(-1, attn_states.shape[-1]).std(dim=0).detach()
            self.scale_inited = True

        # Bidirectional Attention
        residual = hidden_states
        hidden_states = self.bi_layernorm(hidden_states[..., :self.bi_length, :])
        hidden_states, bi_attn_weights = self.bi_attn(
            hidden_states=hidden_states,
            attention_mask=None, # attention_mask=attention_mask,
            position_ids=(position_ids[..., :self.bi_length] if position_ids is not None else None),
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=(tuple(p[:, :self.bi_length, :] for p in position_embeddings) if position_embeddings is not None else None),
            **kwargs,
        )
        hidden_states = hidden_states * expand_to_batch(self.bi_scale, hidden_states).detach()
        hidden_states = torch.cat(
            [
                residual[..., :self.bi_length, :] + hidden_states,
                residual[..., self.bi_length:, :],
            ],
            dim=-2
        )

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # Prepare outputs
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class ZLmModel(PreTrainedModel):

    config_class = ZLmConfig
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    
    
    def __init__(self, config: ZLmConfig, cpu=False):
        super().__init__(config)

        # save config
        self.input_length = config.input_length
        self.output_length = config.output_length
        self.z_length = config.z_length
        self.latent_size = config.latent_size

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

        # save extra config
        self.hidden_size = base_model.config.hidden_size

        # copy token embeddings (clone to untie)
        self.embed_tokens = base_model.model.embed_tokens
        self.embed_tokens.weight = nn.Parameter(self.embed_tokens.weight.data.clone().detach())
        if config.fix_embeddings:
            # For some reason TinyLlama seems to have a broken start token embedding (?)
            self.embed_tokens.weight.data[1] = self.embed_tokens.weight.data[2].clone().detach()

        self.lm_head = base_model.lm_head
        self.lm_head.weight = nn.Parameter(self.lm_head.weight.data.clone().detach())

        # calculate embedding stats
        embed_std = self.embed_tokens.weight.data.std(0, keepdim=True).detach()
        embed_mean = self.embed_tokens.weight.data.mean(0, keepdim=True).detach()

        # create encoder special tokens
        self.encoder_sep_token = nn.Parameter(
            torch.randn(1, self.hidden_size) * embed_std + embed_mean
        )
        self.encoder_z_tokens = nn.Parameter(
            torch.randn(self.z_length, self.hidden_size) * embed_std + embed_mean
        )

        # create generator special tokens
        self.generator_z_tokens = nn.Parameter(
            torch.randn(self.z_length, self.hidden_size) * embed_std + embed_mean
        )

        # create decoder special tokens
        self.decoder_z_tokens = nn.Parameter(
            torch.randn(self.z_length, self.hidden_size) * embed_std + embed_mean
        )
        self.decoder_start_output_token = nn.Parameter(
            torch.randn(1, self.hidden_size) * embed_std + embed_mean
        )

        # create the encoder
        self.encoder = LlamaModel(base_model.model.config)
        for i in range(len(self.encoder.layers)):
            self.encoder.layers[i] = ZLmPrefixLayer(
                base_model.config,
                config,
                layer_idx=i,
                bi_length=(self.input_length + 1 + self.output_length) # +1 for the sep token
            )
        
        self.encoder.load_state_dict(
            {k: v.clone().detach() for k, v in base_model.model.state_dict().items()},
            strict=False
        )

        for layer in self.encoder.layers:
            layer.input_layernorm = ModulatingRMSNorm(
                layer.input_layernorm,
                [self.input_length, 1, self.output_length, self.z_length]
            )
            layer.post_attention_layernorm = ModulatingRMSNorm(
                layer.post_attention_layernorm,
                [self.input_length, 1, self.output_length, self.z_length]
            )
            layer.bi_layernorm = ModulatingRMSNorm(
                layer.bi_layernorm,
                [self.input_length, 1, self.output_length]
            )
        
        self.encoder.norm.weight.data = torch.ones_like(self.encoder.norm.weight.data)

        # create the generator
        self.generator = LlamaModel(base_model.model.config)

        self.generator.load_state_dict(
            {k: v.clone().detach() for k, v in base_model.model.state_dict().items()},
            strict=True
        )

        for layer in self.generator.layers:
            layer.input_layernorm = ModulatingRMSNorm(
                layer.input_layernorm,
                [self.input_length, self.z_length]
            )
            layer.post_attention_layernorm = ModulatingRMSNorm(
                layer.post_attention_layernorm,
                [self.input_length, self.z_length]
            )

        self.generator.norm.weight.data = torch.ones_like(self.generator.norm.weight.data)

        # create the decoder
        self.decoder = LlamaModel(base_model.model.config)
        for i in range(len(self.decoder.layers)):
            self.decoder.layers[i] = ZLmPrefixLayer(
                base_model.config,
                config,
                layer_idx=i,
                bi_length=(self.input_length + self.z_length) # +1 for the sep token
            )

        self.decoder.load_state_dict(
            {k: v.clone().detach() for k, v in base_model.model.state_dict().items()},
            strict=False
        )

        for layer in self.decoder.layers:
            layer.input_layernorm = ModulatingRMSNorm(
                layer.input_layernorm,
                [self.input_length, self.z_length, self.output_length]
            )
            layer.post_attention_layernorm = ModulatingRMSNorm(
                layer.post_attention_layernorm,
                [self.input_length, self.z_length, self.output_length]
            )
            layer.bi_layernorm = ModulatingRMSNorm(
                layer.bi_layernorm,
                [self.input_length, self.z_length]
            )

        # create the input linears
        self.encoder_noise_proj_in = nn.Linear(self.latent_size, self.hidden_size, bias=False)
        self.generator_z_proj_in = nn.Linear(self.latent_size, self.hidden_size, bias=False)
        self.decoder_z_proj_in = nn.Linear(self.latent_size, self.hidden_size, bias=False)

        # scale input layers by embedding stats
        self.encoder_noise_proj_in.weight.data *= embed_std[0][..., None]
        self.generator_z_proj_in.weight.data *= embed_std[0][..., None]
        self.decoder_z_proj_in.weight.data *= embed_std[0][..., None]

        # create the output linears
        self.encoder_mu_proj_out = nn.Linear(self.hidden_size, self.latent_size, bias=False)
        self.encoder_mu_extra_proj_out = nn.Linear(self.hidden_size, self.latent_size, bias=False)
        
        self.generator_mu_proj_out = nn.Linear(self.hidden_size, self.latent_size, bias=False)
        self.generator_mu_extra_proj_out = nn.Linear(self.hidden_size, self.latent_size, bias=False)

        # create the padders
        self.encoder_padder = Padder(
            (self.input_length + 1 + self.output_length), 0 # +1 for the sep token
        )
        self.generator_padder = Padder(
            prefix_length=self.input_length,
            suffix_length=0,
        )

        # Initialize weights and gradient checkpointing
        self.post_init()
    

    # overwrite to prevent overwriting the base model weights
    def init_weights(self):
        return


    def forward(
        self,
        input_ids: torch.LongTensor,
        output_ids: torch.LongTensor,
        disable_generator: bool = False,
        alpha: float = 0.0,
    ):
        
        # get the input and output tokens
        input_tokens = self.embed_tokens(input_ids)
        output_tokens = self.embed_tokens(output_ids)
        if disable_generator:
            input_tokens = input_tokens.detach()
            output_tokens = output_tokens.detach()

        # generate the noise
        noise = torch.randn(
            *input_ids.shape[:-1],
            self.z_length,
            self.latent_size,
        ).to(input_tokens)

        # get the encoder input
        encoder_hidden_states = torch.cat(
            [
                input_tokens,
                expand_to_batch(self.encoder_sep_token, input_tokens),
                output_tokens,
                expand_to_batch(self.encoder_z_tokens[:1], output_tokens),
                expand_to_batch(self.encoder_z_tokens[1:], output_tokens) + self.encoder_noise_proj_in(noise[..., :-1, :]),
            ],
            dim=-2
        )

        # pass through the encoder
        encoder_hidden_states = self.encoder(
            inputs_embeds=encoder_hidden_states
        ).last_hidden_state
        encoder_hidden_states = self.encoder_padder.unpad(encoder_hidden_states)

        # get the encoder outputs
        encoder_mus_base = self.encoder_mu_proj_out(encoder_hidden_states)
        encoder_mus_extra = F.rms_norm(
            self.encoder_mu_extra_proj_out(encoder_hidden_states),
            normalized_shape=[self.latent_size,],
            eps=self.encoder.config.rms_norm_eps
        )
        encoder_mus = encoder_mus_base + alpha * encoder_mus_extra
        z = encoder_mus + noise

        # use the generator
        if disable_generator:
            generator_mus_base = torch.zeros_like(encoder_mus)
            generator_mus_extra = torch.zeros_like(encoder_mus)

        else:

            # get the generator inputs
            generator_hidden_states = torch.cat(
                [
                    input_tokens,
                    expand_to_batch(self.generator_z_tokens[:1], input_tokens),
                    expand_to_batch(self.generator_z_tokens[1:], input_tokens) + self.generator_z_proj_in(z[..., :-1, :]),
                ],
                dim=-2
            )

            # pass through the generator
            generator_hidden_states = self.generator(
                inputs_embeds=generator_hidden_states,
            ).last_hidden_state
            generator_hidden_states = self.generator_padder.unpad(generator_hidden_states)

            # get the generator outputs
            generator_mus_base = self.generator_mu_proj_out(generator_hidden_states)
            generator_mus_extra = self.generator_mu_extra_proj_out(generator_hidden_states)

        generator_mus = generator_mus_base + alpha * generator_mus_extra

        # get the decoder input
        decoder_hidden_states = torch.cat(
            [
                input_tokens,
                expand_to_batch(self.decoder_z_tokens, output_tokens) + self.decoder_z_proj_in(z),
                expand_to_batch(self.decoder_start_output_token, output_tokens),
                output_tokens[..., :-1, :],
            ],
            dim=-2
        )

        decoder_hidden_states = self.decoder(
            inputs_embeds=decoder_hidden_states,
        ).last_hidden_state

        # get the output logits
        lm_logits = self.lm_head(
            decoder_hidden_states[..., -self.output_length:, :]
        )
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        return DotDict(
            encoder_mus=encoder_mus,
            encoder_mus_base=encoder_mus_base,
            encoder_mus_extra=encoder_mus_extra,
            generator_mus=generator_mus,
            generator_mus_base=generator_mus_base,
            generator_mus_extra=generator_mus_extra,
            lm_logits=lm_logits,
            z=z,
        )


    def _set_norm_index(self, index):
        for m in self.modules():
            if isinstance(m, ModulatingRMSNorm):
                m.index = index

    def _set_norm_dropout(self, dropout):
        for m in self.modules():
            if isinstance(m, ModulatingRMSNorm):
                m.dropout = dropout

    
    def _set_sample_mode(self, mode):
        for m in self.modules():
            if isinstance(m, ZLmLayer):
                m.sample_mode = mode
    
    def _set_down_mask(self, mask):
        # TODO: seperate this so it doesn' affect the encoder
        for m in self.modules():
            if isinstance(m, ZLmLayer):
                m.down_mask = mask


    def sample_noise(
        self, 
        input_ids: torch.LongTensor
    ):

        input_tokens = self.embed_tokens(input_ids)
        
        # generate the noise
        return torch.randn(
            *input_ids.shape[:-1],
            self.z_length,
            self.total_latent_size,
        ).to(input_tokens)


    @torch.no_grad()
    def sample(
        self,
        input_ids: torch.LongTensor,
        temperature: float = 1.0,
        noise=None,
        boost_scale: float = 1.0,
    ):
        from transformers.cache_utils import DynamicCache
        from tqdm import tqdm

        # get the input tokens
        input_tokens = self.embed_tokens(input_ids)

        # generate the noise
        if noise is None:
            noise = self.sample_noise(input_ids)
        noise = noise * temperature

        """ Generator """

        # initialize the cache
        cache = DynamicCache()

        # set sample mode
        self._set_sample_mode(True)

        # pass the input tokens through the generator
        self._set_norm_index(0)
        self._set_down_mask(0.0)

        input_embeds = torch.cat(
            [
                input_tokens,
                expand_to_batch(torch.zeros_like(noise[0, 0, :]), input_tokens),
            ],
            dim=-1
        )
        cache = self.generator(
            inputs_embeds=input_embeds,
            use_cache=True,
            past_key_values=cache,
        ).past_key_values        

        # pass the noise through the generator
        zs = []
        z_prev = torch.zeros_like(noise[..., :1, :])
        z_tokens = expand_to_batch(self.generator_z_tokens, input_tokens)

        self._set_norm_index(1)
        self._set_down_mask(1.0)

        for i in tqdm(range(self.z_length), desc="Sampling z tokens"):

            input_embeds = torch.cat(
                [
                    z_tokens[..., i:i+1, :] + self.generator_z_proj_in(z_prev),
                    noise[..., i:i+1, :],
                ],
                dim=-1
            )

            outputs = self.generator(
                inputs_embeds=input_embeds,
                use_cache=True,
                past_key_values=cache,
            )

            z_prev = outputs.last_hidden_state[..., self.hidden_size:] + noise[..., i:i+1, :]
            zs.append(z_prev)

            cache = outputs.past_key_values

        z = torch.cat(zs, dim=-2)
        mu = z - noise

        """ Decoder """

        cache = DynamicCache()

        # pass the input tokens through the decoder
        self._set_norm_index(0)
        cache = self.decoder(
            inputs_embeds=input_tokens,
            use_cache=True,
            past_key_values=cache,
        ).past_key_values

        # pass z through the decoder
        self._set_norm_index(1)
        cache = self.decoder(
            inputs_embeds=expand_to_batch(self.decoder_z_tokens, input_tokens) + self.decoder_z_proj_in(z),
            use_cache=True,
            past_key_values=cache,
        ).past_key_values
            
        self._set_norm_index(2)
        prev_token = expand_to_batch(self.decoder_start_output_token, input_tokens)
        output_tokens = []
        for i in tqdm(range(self.output_length), desc="Sampling output tokens"):

            outputs = self.decoder(
                inputs_embeds=prev_token,
                use_cache=True,
                past_key_values=cache,
            )

            logits = self.lm_head(outputs.last_hidden_state)
            ind = logits.argmax(-1)

            output_tokens.append(ind)

            prev_token = self.embed_tokens(ind)
            cache = outputs.past_key_values

        output_tokens = torch.cat(output_tokens, dim=-1)

        return DotDict(
            tokens=output_tokens,
            z=z,
            mu=mu
        )


    @torch.no_grad()
    def guided_sample(
        self,
        input_ids: torch.LongTensor,
        temperature: float = 1.0,
        guidance_scale: float = 2.0,
        dropout_level: float = 0.1,
    ):
        from transformers.cache_utils import DynamicCache
        from tqdm import tqdm

        # get the input tokens
        input_tokens = self.embed_tokens(input_ids)
        z_tokens = expand_to_batch(self.decoder_z_tokens, input_tokens)

        # generate the noise
        noise = torch.randn(
            *input_ids.shape[:-1],
            self.z_length,
            self.latent_size,
        ).to(input_tokens) * temperature

        # initialize the cache
        cache = DynamicCache()
        bad_cache = DynamicCache()

        # pass the input tokens through the decoder
        self._set_norm_index(0)
        cache = self.decoder(
            inputs_embeds=input_tokens,
            use_cache=True,
            past_key_values=cache,
        ).past_key_values        

        self._set_norm_dropout(dropout_level)
        bad_cache = self.decoder(
            inputs_embeds=input_tokens,
            use_cache=True,
            past_key_values=bad_cache,
        ).past_key_values
        self._set_norm_dropout(None)

        # pass the noise through the decoder
        z_prev = torch.zeros_like(noise[..., :1, :])
        self._set_norm_index(1)
        for i in tqdm(range(self.z_length), desc="Sampling z tokens"):

            outputs = self.decoder(
                inputs_embeds=(z_tokens[..., i:i+1, :] + self.decoder_z_proj_in(z_prev)),
                use_cache=True,
                past_key_values=cache,
            )

            self._set_norm_dropout(dropout_level)
            bad_outputs = self.decoder(
                inputs_embeds=(z_tokens[..., i:i+1, :] + self.decoder_z_proj_in(z_prev)),
                use_cache=True,
                past_key_values=bad_cache,
            )
            self._set_norm_dropout(None)

            mu_good = self.decoder_mu_proj_out(
                self.z_norm(outputs.last_hidden_state)
            )
            mu_bad = self.decoder_mu_proj_out(
                self.z_norm(bad_outputs.last_hidden_state)
            )
            mu_guided = mu_bad + guidance_scale * (mu_good - mu_bad)

            z_prev = noise[..., i:i+1, :] + mu_guided

            cache = outputs.past_key_values
            bad_cache = bad_outputs.past_key_values

        # sample the output tokens
        self._set_norm_index(2)

        prev_token = expand_to_batch(self.decoder_start_output_token, input_tokens)
        output_tokens = []
        for i in tqdm(range(self.output_length), desc="Sampling output tokens"):

            outputs = self.decoder(
                inputs_embeds=prev_token,
                use_cache=True,
                past_key_values=cache,
            )

            logits = self.lm_head(self.lm_norm(outputs.last_hidden_state))
            ind = logits.argmax(-1)

            output_tokens.append(ind)

            prev_token = self.embed_tokens(ind)
            cache = outputs.past_key_values

        output_tokens = torch.cat(output_tokens, dim=-1)

        return output_tokens
    