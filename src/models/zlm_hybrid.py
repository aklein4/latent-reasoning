import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    PreTrainedModel, PretrainedConfig,
    LlamaModel, LlamaForCausalLM
)

import numpy as np

from utils.dot_dict import DotDict
from utils.model_utils import unsqueeze_to_batch, expand_to_batch, momentum_scan
import utils.constants as constants


class ZLmHybridConfig(PretrainedConfig):
    """
    Configuration class for ZLM model.
    This is a subclass of LlamaConfig with additional parameters specific to ZLM.
    """

    model_type = "zlm_hybrid"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        base_url: str = "meta-llama/Llama-2-7b-chat-hf",
        input_length: int = 128,
        output_length: int = 128,
        z_length: int = 512,
        latent_size_per_layer: int = 8,
        num_latent_layers: int = 10,
        *args,
        **kwargs
    ):
        
        self.base_url = base_url

        self.input_length = input_length
        self.output_length = output_length

        self.z_length = z_length
        self.latent_size_per_layer = latent_size_per_layer
        self.num_latent_layers = num_latent_layers

        super().__init__(*args, **kwargs)


class LatentShaper:

    def __init__(self, latent_size_per_layer, num_latent_layers):
        self.latent_size_per_layer = latent_size_per_layer
        self.num_latent_layers = num_latent_layers
        
        self.total_latent_size = latent_size_per_layer * num_latent_layers


    def layerfy(self, x):
        assert x.shape[-1] == self.total_latent_size, f"Expected last dimension to be {self.total_latent_size}, but got {x.shape[-1]}"
        return x.view(
            *x.shape[:-1],
            self.latent_size_per_layer,
            self.num_latent_layers,
        )
    

    def unlayerfy(self, x):
        assert x.shape[-2:] == (self.latent_size_per_layer, self.num_latent_layers), f"Expected last two dimensions to be ({self.latent_size_per_layer}, {self.num_latent_layers}), but got {x.shape[-2:]}"
        return x.view(
            *x.shape[:-2],
            self.total_latent_size,
        )


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


class PassZLmLayer(nn.Module):

    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer
        self.hidden_size = base_layer.hidden_size


    def forward(
        self,
        hidden_states,
        *args,
        **kwargs,
    ):
        extra = hidden_states[..., self.hidden_size:]
        out = self.base_layer(
            hidden_states[..., :self.hidden_size],
            *args,
            **kwargs,
        )

        return (
            torch.cat([out[0], extra], dim=-1),
        ) + out[1:] # extra things in tuple


class ZLmLayer(nn.Module):

    def __init__(
        self,
        config: ZLmHybridConfig,
        is_encoder,
        base_layer,
        prefix_length,
        suffix_length,
        layer_idx,
        latent_layer_idx,
        latent_size_per_layer=None,
    ):
        super().__init__()
        
        self.is_encoder = is_encoder

        self.base_layer = base_layer
        self.hidden_size = base_layer.hidden_size

        self.latent_size_per_layer = (
            latent_size_per_layer if latent_size_per_layer is not None else config.latent_size_per_layer
        )
        self.num_latent_layers = config.num_latent_layers
        self.total_latent_size = self.latent_size_per_layer * self.num_latent_layers

        self.prefix_length = prefix_length
        self.suffix_length = suffix_length
        self.layer_idx = layer_idx
        self.latent_layer_idx = latent_layer_idx

        self.mu_norm = nn.RMSNorm(
            normalized_shape=self.hidden_size,
            eps=self.base_layer.input_layernorm.norm.eps,
            elementwise_affine=True
        )
        self.mu_up = nn.Linear(
            self.hidden_size,
            self.latent_size_per_layer,
            bias=False
        )
        self.z_down = nn.Linear(
            self.latent_size_per_layer,
            self.hidden_size,
            bias=False
        )

        self.shaper = LatentShaper(
            self.latent_size_per_layer,
            self.num_latent_layers
        )

        self.sample_mode = False
        self.down_mask = None


    def forward(
        self,
        hidden_states,
        *args,
        **kwargs,
    ):
        
        # hack to get arguments through existing LlamaModel
        total_noise_or_z = self.shaper.layerfy(hidden_states[..., self.hidden_size:])
        noise_or_z = total_noise_or_z[..., self.latent_layer_idx]
        
        # pass through the base layer
        hidden_states = hidden_states[..., :self.hidden_size]
        base_out = self.base_layer(
            hidden_states,
            *args,
            **kwargs,
        )
        hidden_states = base_out[0]

        # get mu
        mu = self.mu_up(
            self.mu_norm(hidden_states)
        )
        
        if self.sample_mode:
            noise_or_z = noise_or_z + mu

        # add z to the residual stream
        y = self.z_down(noise_or_z)
        
        if self.down_mask is not None:
            hidden_states = hidden_states + y * self.down_mask

        else:
            if self.suffix_length > 0:
                hidden_states = torch.cat(
                    [
                        hidden_states[..., :self.prefix_length, :],
                        hidden_states[..., self.prefix_length:-self.suffix_length, :] + y[..., self.prefix_length:-self.suffix_length, :],
                        hidden_states[..., -self.suffix_length:, :],
                    ],
                    dim=-2
                )
            else:
                hidden_states = torch.cat(
                    [
                        hidden_states[..., :self.prefix_length, :],
                        hidden_states[..., self.prefix_length:, :] + y[..., self.prefix_length:, :],
                    ],
                    dim=-2
                )

        total_noise_or_z_with_mu = total_noise_or_z.clone()
        total_noise_or_z_with_mu[..., self.latent_layer_idx] = mu

        hidden_states = torch.cat(
            [
                hidden_states,
                self.shaper.unlayerfy(total_noise_or_z_with_mu),
            ],
            dim=-1
        )

        return (hidden_states,) + base_out[1:]


class ZLmHybridModel(PreTrainedModel):

    config_class = ZLmHybridConfig
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    
    
    def __init__(self, config: ZLmHybridConfig, cpu=False):
        super().__init__(config)

        # save config
        self.input_length = config.input_length
        self.output_length = config.output_length

        self.z_length = config.z_length
        self.num_latent_layers = config.num_latent_layers
        self.latent_size_per_layer = config.latent_size_per_layer
        self.total_latent_size = config.latent_size_per_layer * config.num_latent_layers

        self.shaper = LatentShaper(
            self.latent_size_per_layer,
            self.num_latent_layers
        )

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

        # create the encoder and decoder
        self.encoder = LlamaModel(base_model.model.config)
        self.generator = LlamaModel(base_model.model.config)
        self.decoder = LlamaModel(base_model.model.config)

        # copy the encoder and decoder params from the base model
        transformers_with_strides = [
            (
                self.encoder,
                [self.input_length, 1, self.output_length, self.z_length], # 1 for sep token
                self.input_length + 1 + self.output_length, # +1 for sep token
                0,
                True,
                2 * self.latent_size_per_layer
            ),
            (
                self.generator,
                [self.input_length, self.z_length],
                self.input_length,
                0,
                False,
                self.latent_size_per_layer
            ),
            (
                self.decoder,
                [self.input_length, self.z_length, self.output_length],
                0,
                0,
                None,
                None
            )
        ]
        for transformer, strides, prefix, suffix, is_encoder, latent_size in transformers_with_strides:
            transformer.load_state_dict({k: v.clone().detach() for k, v in base_model.model.state_dict().items()})

            # replace the layernorms with modulating RMSNorms
            for layer in transformer.layers:
                layer.input_layernorm = ModulatingRMSNorm(
                    layer.input_layernorm,
                    strides
                )
                layer.post_attention_layernorm = ModulatingRMSNorm(
                    layer.post_attention_layernorm,
                    strides
                )
            
            # replace the existing layers with ZLmLayers
            if is_encoder is not None:
                latent_i = 0
                for i, layer in enumerate(transformer.layers):

                    # latent layers
                    if (len(transformer.layers) - i) <= self.num_latent_layers:
                        transformer.layers[i] = ZLmLayer(
                            config,
                            is_encoder,
                            layer,
                            prefix,
                            suffix,
                            i,
                            latent_i,
                            latent_size_per_layer=latent_size,
                        )
                        latent_i += 1

                    # pass through these layers
                    else:
                        transformer.layers[i] = PassZLmLayer(layer)
                
                # store the latent layers
                transformer.latent_layers = transformer.layers[-self.num_latent_layers:]

                # create the padder for the transformer
                transformer.padder = Padder(prefix, suffix)

        # create the input linears
        self.encoder_noise_proj_in = nn.Linear(self.total_latent_size, self.hidden_size, bias=False)
        self.generator_z_proj_in = nn.Linear(self.total_latent_size, self.hidden_size, bias=False)
        self.decoder_z_proj_in = nn.Linear(self.total_latent_size, self.hidden_size, bias=False)

        # scale input layers by embedding stats
        self.encoder_noise_proj_in.weight.data *= embed_std[0][..., None]
        self.generator_z_proj_in.weight.data *= embed_std[0][..., None]
        self.decoder_z_proj_in.weight.data *= embed_std[0][..., None]

        # fix the output norms
        self.encoder.norm = nn.Identity()
        self.generator.norm = nn.Identity()
        # decoder norm is fine

        # Initialize weights and gradient checkpointing
        self.post_init()
    

    # overwrite to prevent overwriting the base model weights
    def init_weights(self):
        return


    def forward(
        self,
        input_ids: torch.LongTensor,
        output_ids: torch.LongTensor,
        noise_scale: float = 1.0,
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
            self.total_latent_size,
        ).to(input_tokens) * noise_scale

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

        encoder_noise = self.shaper.layerfy(noise)
        encoder_noise = torch.cat([encoder_noise, torch.zeros_like(encoder_noise)], dim=-2)
        encoder_noise = self.encoder.latent_layers[0].shaper.unlayerfy(encoder_noise)

        encoder_hidden_states = torch.cat(
            [
                encoder_hidden_states,
                self.encoder.padder.pad(encoder_noise)
            ],
            dim=-1
        )

        # pass through the encoder
        encoder_hidden_states = self.encoder(
            inputs_embeds=encoder_hidden_states
        ).last_hidden_state
        
        # get flattened mus and z
        encoder_mus = self.encoder.padder.unpad(encoder_hidden_states[..., self.hidden_size:])
        encoder_mus = self.encoder.latent_layers[0].shaper.layerfy(encoder_mus)
        encoder_mus_base, encoder_mus_extra = torch.chunk(encoder_mus, 2, dim=-2)

        encoder_mus_extra = alpha * F.rms_norm(
            encoder_mus_extra.swapaxes(-2, -1),
            normalized_shape=[self.latent_size_per_layer],
            eps=self.encoder.latent_layers[0].mu_norm.eps
        ).swapaxes(-2, -1)

        encoder_mus = self.shaper.unlayerfy(encoder_mus_base + encoder_mus_extra)    
        z = noise + encoder_mus

        # get the generator input
        if disable_generator:
            generator_mus = torch.zeros_like(encoder_mus)

        else:
            generator_hidden_states = torch.cat(
                [
                    input_tokens,
                    expand_to_batch(self.generator_z_tokens[:1], input_tokens),
                    expand_to_batch(self.generator_z_tokens[1:], input_tokens) + self.generator_z_proj_in(z[..., :-1, :]),
                ],
                dim=-2
            )
            generator_hidden_states = torch.cat(
                [
                    generator_hidden_states,
                    self.generator.padder.pad(z)
                ],
                dim=-1
            )

            # pass through the generator
            generator_hidden_states = self.generator(
                inputs_embeds=generator_hidden_states,
            ).last_hidden_state

            # get the flattened generator mus
            generator_mus = self.generator.padder.unpad(generator_hidden_states[..., self.hidden_size:])

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
        lm_logits = self.lm_head(decoder_hidden_states[..., -self.output_length:, :])
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        return DotDict(
            encoder_mus=self.shaper.layerfy(encoder_mus),
            generator_mus=self.shaper.layerfy(generator_mus),
            encoder_mus_base=encoder_mus_base,
            encoder_mus_extra=encoder_mus_extra,
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
    