import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    PreTrainedModel, PretrainedConfig,
    LlamaModel, LlamaForCausalLM
)

from utils.dot_dict import DotDict
from utils.model_utils import unsqueeze_to_batch, expand_to_batch
import utils.constants as constants


class UncondZLmConfig(PretrainedConfig):
    """
    Configuration class for ZLM model.
    This is a subclass of LlamaConfig with additional parameters specific to ZLM.
    """

    model_type = "uncond_zlm"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        base_url: str = "meta-llama/Llama-2-7b-chat-hf",
        input_length: int = 128,
        output_length: int = 128,
        z_length: int = 512,
        latent_size_per_layer: int = 8,
        num_latent_layers: int = 10,
        mu_init_scale: float = 0.1,
        small_init_scale: float = 0.01,
        *args,
        **kwargs
    ):
        
        self.base_url = base_url

        self.input_length = input_length
        self.output_length = output_length

        self.z_length = z_length
        self.latent_size_per_layer = latent_size_per_layer
        self.num_latent_layers = num_latent_layers
        
        self.mu_init_scale = mu_init_scale
        self.small_init_scale = small_init_scale

        super().__init__(*args, **kwargs)


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


class UncondZLmLayer(nn.Module):

    def __init__(
        self,
        config: UncondZLmConfig,
        base_layer,
        prefix_length,
        layer_idx,
    ):
        super().__init__()
        
        self.hidden_size = base_layer.hidden_size
        self.latent_size = config.latent_size_per_layer

        self.base_layer = base_layer

        self.prefix_length = prefix_length
        self.layer_idx = layer_idx

        self.mu_norm = nn.RMSNorm(
            normalized_shape=self.hidden_size,
            eps=self.base_layer.input_layernorm.norm.eps,
            elementwise_affine=True
        )
        self.mu_up = nn.Linear(
            self.hidden_size,
            self.latent_size,
            bias=False
        )
        self.z_down = nn.Linear(
            self.latent_size,
            self.hidden_size,
            bias=False
        )

        self.mu_up.weight.data *= config.mu_init_scale
        self.z_down.weight.data *= config.small_init_scale

        self.noise = None
        self.z = None
        self.mu = None


    def pad(self, x):
        if x is None:
            return None
        return torch.cat(
            [
                torch.zeros_like(x[..., :1, :]).expand(
                    *[-1 for _ in range(x.ndim - 2)],
                    self.prefix_length,
                    -1
                ),
                x,
            ],
            dim=-2
        )

    def unpad(self, x):
        if x is None:
            return None
        return x[..., self.prefix_length:, :]
        

    def fill(
        self,
        noise=None,
        z=None,
        do_pad=True,
    ):
        assert (noise is not None) ^ (z is not None), "Either noise or z must be provided, but not both."
        self.noise = self.pad(noise) if do_pad else noise
        self.z = self.pad(z) if do_pad else z
    
    def get_noise(self, do_unpad=True):
        return self.unpad(self.noise) if do_unpad else self.noise
    
    def get_z(self, do_unpad=True):
        return self.unpad(self.z) if do_unpad else self.z

    def get_mu(self, do_unpad=True):
        return self.unpad(self.mu) if do_unpad else self.mu

    def clear(self):
        self.noise = None
        self.z = None
        self.mu = None


    def forward(self, *args, **kwargs):
        base_out = self.base_layer(*args, **kwargs)

        hidden_states = base_out[0]

        self.mu = self.mu_up(
            self.mu_norm(hidden_states)
        )

        if self.noise is not None:
            assert self.z is None, "z must be None if noise is provided."
            self.z = self.noise + self.mu
        else:
            assert self.z is not None, "z must be provided if noise is not."

        y = self.z_down(self.z)
        hidden_states = torch.cat(
            [
                hidden_states[..., :self.prefix_length, :],
                hidden_states[..., self.prefix_length:, :] + y[..., self.prefix_length:, :],
            ],
            dim=-2
        )

        return (hidden_states, *base_out[1:])


class UncondZLmModel(PreTrainedModel):

    config_class = UncondZLmConfig
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    
    
    def __init__(self, config: UncondZLmConfig, cpu=False):
        super().__init__(config)

        # save config
        self.input_length = config.input_length
        self.output_length = config.output_length

        self.z_length = config.z_length
        self.num_latent_layers = config.num_latent_layers
        self.latent_size_per_layer = config.latent_size_per_layer
        self.latent_size = config.latent_size_per_layer * config.num_latent_layers

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
        self.decoder_output_tokens = nn.Parameter(
            torch.randn(self.output_length, self.hidden_size) * embed_std + embed_mean
        )

        # create the encoder and decoder
        self.encoder = LlamaModel(base_model.model.config)
        self.generator = LlamaModel(base_model.model.config)
        self.decoder = LlamaModel(base_model.model.config)

        # copy the encoder and decoder params from the base model
        transformers_with_strides = [
            (
                self.encoder,
                [self.output_length + 1, self.z_length], # +1 for bos token
                self.output_length + 1
            ),
            (
                self.generator,
                [self.input_length, self.z_length],
                self.input_length
            ),
            (
                self.decoder,
                [1, self.z_length, self.output_length], # 1 for bos
                None
            )
        ]
        for transformer, strides, prefix in transformers_with_strides:
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
            
            if prefix is not None:
                for i, layer in enumerate(transformer.layers):

                    if (len(transformer.layers) - i) <= self.num_latent_layers:
                        transformer.layers[i] = UncondZLmLayer(
                            config,
                            layer,
                            prefix,
                            i
                    )
                
                transformer.latent_layers = transformer.layers[-self.num_latent_layers:]

        # create the input linears
        self.encoder_noise_proj_in = nn.Linear(self.latent_size, self.hidden_size, bias=False)
        self.generator_z_proj_in = nn.Linear(self.latent_size, self.hidden_size, bias=False)
        self.decoder_z_proj_in = nn.Linear(self.latent_size, self.hidden_size, bias=False)

        self.encoder_noise_proj_in.weight.data *= embed_std[0][..., None]
        self.generator_z_proj_in.weight.data *= embed_std[0][..., None]
        self.decoder_z_proj_in.weight.data *= embed_std[0][..., None]

        # Initialize weights and gradient checkpointing
        self.post_init()
    

    # overwrite to prevent overwriting the base model weights
    def init_weights(self):
        return


    def layerfy(self, x):
        assert x.shape[-1] == self.latent_size, f"Expected last dimension to be {self.latent_size}, but got {x.shape[-1]}"
        return x.view(
            *x.shape[:-1],
            self.latent_size_per_layer,
            self.num_latent_layers,
        )

    def unlayerfy(self, x):
        assert x.shape[-2:] == (self.latent_size_per_layer, self.num_latent_layers), f"Expected last two dimensions to be ({self.latent_size_per_layer}, {self.num_latent_layers}), but got {x.shape[-2:]}"
        return x.view(
            *x.shape[:-2],
            self.latent_size,
        )


    def pre_transformer(self, transformer, z=None, noise=None):
        if noise is not None:
            noise = self.layerfy(noise)
        if z is not None:
            z = self.layerfy(z)

        for i, layer in enumerate(transformer.latent_layers):
            layer.fill(
                noise=(noise[..., i] if noise is not None else None),
                z=(z[..., i] if z is not None else None)
            )


    def post_transformer(self, transformer):
        mu = self.unlayerfy(torch.stack(
            [layer.get_mu() for layer in transformer.latent_layers],
            dim=-1
        ))
        z = self.unlayerfy(torch.stack(
            [layer.get_z() for layer in transformer.latent_layers],
            dim=-1
        ))
        for layer in transformer.latent_layers:
            layer.clear()
        return mu, z
    

    def forward(
        self,
        input_ids: torch.LongTensor,
        output_ids: torch.LongTensor,
    ):
        
        # get the input and output tokens
        input_tokens = self.embed_tokens(input_ids)
        output_tokens = self.embed_tokens(output_ids)
        bos_token = self.embed_tokens(torch.full_like(input_ids[..., :1], self.encoder.config.bos_token_id))

        # generate the noise
        noise = torch.randn(
            *input_ids.shape[:-1],
            self.z_length,
            self.latent_size,
        ).to(input_tokens)

        # get the encoder input
        encoder_hidden_states = torch.cat(
            [
                bos_token,
                output_tokens,
                expand_to_batch(self.encoder_z_tokens[:1], output_tokens),
                expand_to_batch(self.encoder_z_tokens[1:], output_tokens) + self.encoder_noise_proj_in(noise[..., :-1, :]),
            ],
            dim=-2
        )

        # pass through the encoder
        self.pre_transformer(self.encoder, noise=noise)
        self.encoder(inputs_embeds=encoder_hidden_states)
        encoder_mus, z = self.post_transformer(self.encoder)

        # get the generator input
        generator_hidden_states = torch.cat(
            [
                input_tokens,
                expand_to_batch(self.generator_z_tokens[:1], input_tokens),
                expand_to_batch(self.generator_z_tokens[1:], input_tokens) + self.generator_z_proj_in(z[..., :-1, :]),
            ],
            dim=-2
        )

        # pass through the generator
        self.pre_transformer(self.generator, z=z)
        self.generator(inputs_embeds=generator_hidden_states)
        generator_mus, _ = self.post_transformer(self.generator)

        # get the decoder input
        decoder_hidden_states = torch.cat(
            [
                bos_token,
                expand_to_batch(self.decoder_z_tokens, output_tokens) + self.decoder_z_proj_in(z),
                expand_to_batch(self.decoder_output_tokens, output_tokens),
            ],
            dim=-2
        )

        # get the decoder output
        decoder_lm_logits = self.lm_head(
                decoder_hidden_states[..., -self.output_length:, :]
        )
        decoder_lm_logits = F.log_softmax(decoder_lm_logits, dim=-1)

        return DotDict(
            encoder_mus=self.layerfy(encoder_mus),
            generator_mus=self.layerfy(generator_mus),
            lm_logits=decoder_lm_logits,
        )


    def _set_norm_index(self, index):
        for m in self.modules():
            if isinstance(m, ModulatingRMSNorm):
                m.index = index

    def _set_norm_dropout(self, dropout):
        for m in self.modules():
            if isinstance(m, ModulatingRMSNorm):
                m.dropout = dropout


    def sample_noise(
        self, 
        input_ids: torch.LongTensor
    ):

        input_tokens = self.embed_tokens(input_ids)
        
        # generate the noise
        return torch.randn(
            *input_ids.shape[:-1],
            self.z_length,
            self.latent_size,
        ).to(input_tokens)


    @torch.no_grad()
    def sample(
        self,
        input_ids: torch.LongTensor,
        temperature: float = 1.0,
        noise=None
    ):
        from transformers.cache_utils import DynamicCache
        from tqdm import tqdm

        # get the input tokens
        input_tokens = self.embed_tokens(input_ids)
        z_tokens = expand_to_batch(self.decoder_z_tokens, input_tokens)

        # generate the noise
        if noise is None:
            noise = torch.randn(
                *input_ids.shape[:-1],
                self.z_length,
                self.latent_size,
            ).to(input_tokens) * temperature

        # initialize the cache
        cache = DynamicCache()

        # pass the input tokens through the decoder
        self._set_norm_index(0)
        cache = self.decoder(
            inputs_embeds=input_tokens,
            use_cache=True,
            past_key_values=cache,
        ).past_key_values        

        # pass the noise through the decoder
        mus = []
        z_prev = torch.zeros_like(noise[..., :1, :])
        self._set_norm_index(1)
        for i in tqdm(range(self.z_length), desc="Sampling z tokens"):

            outputs = self.decoder(
                inputs_embeds=(z_tokens[..., i:i+1, :] + self.decoder_z_proj_in(z_prev)),
                use_cache=True,
                past_key_values=cache,
            )

            mu = self.decoder_mu_proj_out(
                self.z_norm(outputs.last_hidden_state)
            )

            mus.append(mu)

            z_prev = noise[..., i:i+1, :] + mu
            cache = outputs.past_key_values

        mus = torch.cat(mus[:-1], dim=-2)

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

        return DotDict(
            tokens=output_tokens,
            mus=mus,
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
    