import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    PreTrainedModel, PretrainedConfig,
    LlamaModel, LlamaForCausalLM, LlamaConfig
)

from utils.dot_dict import DotDict
from utils.model_utils import unsqueeze_to_batch, expand_to_batch
import utils.constants as constants


class ZAEConfig(PretrainedConfig):
    """
    Configuration class for ZLM model.
    This is a subclass of LlamaConfig with additional parameters specific to ZLM.
    """

    model_type = "zae"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        base_url: str = "meta-llama/Llama-2-7b-chat-hf",
        input_length: int = 128,
        output_length: int = 128,
        latent_length: int = 16,
        latent_size: int = 128,
        num_decoder_layers: int = 12,
        *args,
        **kwargs
    ):
        
        self.base_url = base_url

        self.input_length = input_length
        self.output_length = output_length

        self.latent_length = latent_length
        self.latent_size = latent_size

        self.num_decoder_layers = num_decoder_layers

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


class ZAEModel(PreTrainedModel):

    config_class = ZAEConfig
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    
    
    def __init__(self, config: ZAEConfig, cpu=False):
        super().__init__(config)

        # save config
        self.input_length = config.input_length
        self.output_length = config.output_length

        self.latent_length = config.latent_length
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

        self.lm_head = base_model.lm_head
        self.lm_head.weight = nn.Parameter(self.lm_head.weight.data.clone().detach())

        # calculate embedding stats
        embed_std = self.embed_tokens.weight.data.std(0, keepdim=True).detach()
        embed_mean = self.embed_tokens.weight.data.mean(0, keepdim=True).detach()

        # create encoder special tokens
        self.encoder_bos_token = nn.Parameter(
            self.embed_tokens.weight.data[base_model.config.bos_token_id].clone().detach()[None]
        )
        self.encoder_z_tokens = nn.Parameter(
            torch.randn(self.latent_length, self.hidden_size) * embed_std + embed_mean
        )

        # create generator special tokens
        self.generator_z_tokens = nn.Parameter(
            torch.randn(self.latent_length, self.hidden_size) * embed_std + embed_mean
        )

        # create decoder special tokens
        self.decoder_bos_token = nn.Parameter(
            self.embed_tokens.weight.data[base_model.config.bos_token_id].clone().detach()[None]
        )
        self.decoder_z_tokens = nn.Parameter(
            torch.randn(self.latent_length, self.hidden_size) * embed_std + embed_mean
        )
        self.decoder_sep_token = nn.Parameter(
            torch.randn(1, self.hidden_size) * embed_std + embed_mean
        )

        # modify the vae config
        tmp_config = base_model.model.config.to_dict()
        tmp_config["num_hidden_layers"] = config.num_decoder_layers
        short_config = LlamaConfig(**tmp_config)

        # create the encoder and decoder
        self.encoder = LlamaModel(base_model.model.config)
        self.generator = LlamaModel(base_model.model.config)
        self.decoder = LlamaModel(short_config)

        # copy the encoder and decoder params from the base model
        transformers_with_strides = [
            (
                self.encoder,
                [self.output_length + 1, self.latent_length], # +1 for bos token
            ),
            (
                self.generator,
                [self.input_length, self.latent_length],
            ),
            (
                self.decoder,
                [1, self.latent_length, self.output_length], # 1 for bos
            )
        ]
        for transformer, strides in transformers_with_strides:
            transformer.load_state_dict(
                {k: v.clone().detach() for k, v in base_model.model.state_dict().items()},
                strict=False
            )

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

        # reset the scales of the encoder and generator norms to 1
        self.encoder.norm.weight.data = torch.ones_like(self.encoder.norm.weight.data)
        self.generator.norm.weight.data = torch.ones_like(self.generator.norm.weight.data)
        
        # create the encoder io linears
        self.encoder_mu_proj_out = nn.Linear(self.hidden_size, self.latent_size, bias=False)

        # create the generator io linears
        self.generator_mu_proj_out = nn.Linear(self.hidden_size, self.latent_size, bias=False)

        # create the decoder io linears
        self.decoder_z_proj_in = nn.Linear(self.latent_size, self.hidden_size, bias=False)
        self.decoder_z_proj_in.weight.data *= embed_std[0][..., None]

        # Initialize weights and gradient checkpointing
        self.post_init()
    

    # overwrite to prevent overwriting the base model weights
    def init_weights(self):
        return

    
    def forward(
        self,
        input_ids: torch.LongTensor,
        output_ids: torch.LongTensor,
    ):
        
        # get the input and output tokens
        input_tokens = self.embed_tokens(input_ids)
        output_tokens = self.embed_tokens(output_ids)

        # generate the noise
        noise = torch.randn(
            *input_ids.shape[:-1],
            self.latent_length,
            self.latent_size,
        ).to(input_tokens)

        # get the encoder input
        encoder_hidden_states = torch.cat(
            [
                expand_to_batch(self.encoder_bos_token, output_tokens),
                output_tokens,
                expand_to_batch(self.encoder_z_tokens, output_tokens),
            ],
            dim=-2
        )

        # pass through the encoder
        encoder_hidden_states = self.encoder(
            inputs_embeds=encoder_hidden_states
        ).last_hidden_state
        encoder_mus = self.encoder_mu_proj_out(
            encoder_hidden_states[..., -self.latent_length:, :]
        )

        # get z
        z = noise + encoder_mus

        # get the generator input
        generator_hidden_states = torch.cat(
            [
                input_tokens,
                expand_to_batch(self.generator_z_tokens, input_tokens),
            ],
            dim=-2
        )

        # pass through the generator
        generator_hidden_states = self.generator(
            inputs_embeds=generator_hidden_states,
        ).last_hidden_state
        generator_mus = self.generator_mu_proj_out(
            generator_hidden_states[..., -self.latent_length:, :]
        )

        # get the decoder input
        decoder_hidden_states = torch.cat(
            [
                expand_to_batch(self.decoder_bos_token, output_tokens),
                expand_to_batch(self.decoder_z_tokens, output_tokens) + self.decoder_z_proj_in(z),
                expand_to_batch(self.decoder_sep_token, output_tokens),
                output_tokens[..., :-1, :],
            ],
            dim=-2
        )

        # pass through the decoder
        decoder_hidden_states = self.decoder(
            inputs_embeds=decoder_hidden_states,
        ).last_hidden_state

        # get the lm logits
        decoder_lm_logits = self.lm_head(
                decoder_hidden_states[..., -self.output_length:, :]
        )
        decoder_lm_logits = F.log_softmax(decoder_lm_logits, dim=-1)

        return DotDict(
            encoder_mus=encoder_mus,
            generator_mus=generator_mus,
            lm_logits=decoder_lm_logits,
        )


    def encode(
        self,
        output_ids: torch.LongTensor,
    ):
        
        # get the output tokens
        output_tokens = self.embed_tokens(output_ids)

        # get the encoder input
        encoder_hidden_states = torch.cat(
            [
                expand_to_batch(self.encoder_bos_token, output_tokens),
                output_tokens,
                expand_to_batch(self.encoder_z_tokens, output_tokens),
            ],
            dim=-2
        )

        # pass through the encoder
        encoder_hidden_states = self.encoder(
            inputs_embeds=encoder_hidden_states
        ).last_hidden_state
        encoder_mus = self.encoder_mu_proj_out(
            encoder_hidden_states[..., -self.latent_length:, :]
        )

        return encoder_mus
