import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    PreTrainedModel, PretrainedConfig,
    GPT2Model, GPT2LMHeadModel
)

from utils.dot_dict import DotDict
from utils.model_utils import unsqueeze_to_batch, expand_to_batch
import utils.constants as constants


class GPTZLmConfig(PretrainedConfig):
    """
    Configuration class for ZLM model.
    This is a subclass of LlamaConfig with additional parameters specific to ZLM.
    """

    model_type = "gpt_zlm"
    supports_gradient_checkpointing = True


    def __init__(
        self,
        base_url: str = "openai-community/gpt2",
        input_length: int = 128,
        output_length: int = 128,
        z_length: int = 512,
        latent_size: int = 128,
        z_init_scale: float = 0.1,
        *args,
        **kwargs
    ):
        
        self.base_url = base_url

        self.input_length = input_length
        self.output_length = output_length

        self.z_length = z_length
        self.latent_size = latent_size
        
        self.z_init_scale = z_init_scale

        super().__init__(*args, **kwargs)


class ModulatingLayerNorm(nn.Module):

    def __init__(self, old_norm: nn.LayerNorm, strides):
        super().__init__()

        self.normalized_shape = old_norm.weight.shape
        self.num_d = len(self.normalized_shape)

        self.num_strides = len(strides)
        self.strides = strides

        self.norm = nn.LayerNorm(
            normalized_shape=self.normalized_shape,
            eps=old_norm.eps,
            elementwise_affine=False,
            bias=False
        )

        self.scales = nn.ParameterList(
            [
                old_norm.weight.data.clone().detach() for _ in range(self.num_strides)
            ]
        )
        self.biases = nn.ParameterList(
            [
                (old_norm.bias.data.clone().detach() if old_norm.bias is not None else torch.zeros_like(old_norm.weight.data))
                for _ in range(self.num_strides)
            ]
        )
    

    def _get_scales_and_biases(self):

        out = []
        out_biases = []
        for i in range(self.num_strides):
            scale = self.scales[i]
            bias = self.biases[i]
            stride = self.strides[i]

            expanded_scale = scale[None].expand(stride, *([-1] * self.num_d))
            out.append(expanded_scale)

            expanded_bias = bias[None].expand(stride, *([-1] * self.num_d))
            out_biases.append(expanded_bias)
        
        return torch.cat(out, dim=0), torch.cat(out_biases, dim=0)


    def forward(self, x):

        x = self.norm(x)

        scales, biases = self._get_scales_and_biases()
        scales = unsqueeze_to_batch(scales, x)
        biases = unsqueeze_to_batch(biases, x)

        return (x * scales) + biases


class GPTZLmModel(PreTrainedModel):

    config_class = GPTZLmConfig
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    
    
    def __init__(self, config: GPTZLmConfig):
        super().__init__(config)

        # save config
        self.input_length = config.input_length
        self.output_length = config.output_length

        self.z_length = config.z_length
        self.num_z = self.z_length - 1

        self.latent_size = config.latent_size

        # get base model
        base_model = GPT2LMHeadModel.from_pretrained(
            config.base_url
        ).to(torch.float32)

        # enable flash attention
        if str(constants.DEVICE) == "cuda":
            base_model.config._attn_implementation = "sdpa"

        # remove dropout
        base_model.config.resid_pdrop = 0.0
        base_model.config.embd_pdrop = 0.0
        base_model.config.attn_pdrop = 0.0

        self.hidden_size = base_model.config.n_embd

        # copy token embeddings (clone to untie)
        self.embed_tokens = base_model.transformer.wte
        self.embed_tokens.weight = nn.Parameter(self.embed_tokens.weight.data.clone().detach())

        self.lm_head = base_model.lm_head
        self.lm_head.weight = nn.Parameter(self.lm_head.weight.data.clone().detach())

        self.embed_positions = base_model.transformer.wpe

        # calculate embedding stats
        embed_std = self.embed_tokens.weight.data.std(0).detach()
        embed_mean = self.embed_tokens.weight.data.mean(0).detach()

        # create encoder special tokens
        self.encoder_sep_token = nn.Parameter(
            (torch.randn(self.hidden_size) * embed_std + embed_mean)[None]
        )
        self.encoder_z_tokens = nn.Parameter(
            torch.randn(self.hidden_size)[None].repeat(self.num_z, 1) * embed_std[None] + embed_mean[None]
        )

        # create decoder special tokens
        self.decoder_z_tokens = nn.Parameter(
            torch.randn(self.hidden_size)[None].repeat(self.z_length, 1) * embed_std[None] + embed_mean[None]
        )
        self.decoder_start_output_token = nn.Parameter(
            (torch.randn(self.hidden_size) * embed_std + embed_mean)[None]
        )

        # create the encoder and decoder
        self.encoder = GPT2Model(base_model.transformer.config)
        self.decoder = GPT2Model(base_model.transformer.config)

        # copy the encoder and decoder params from the base model
        transformers_with_strides = [
            (self.encoder, [self.input_length, self.output_length + 1, self.num_z]),
            (self.decoder, [self.input_length, self.z_length, self.output_length]),
        ]
        for transformer, strides in transformers_with_strides:
            transformer.load_state_dict({k: v.clone().detach() for k, v in base_model.transformer.state_dict().items()})

            # replace the layernorms with modulating RMSNorms
            for layer in transformer.h:
                layer.ln_1 = ModulatingLayerNorm(
                    layer.ln_1,
                    strides
                )
                layer.ln_2 = ModulatingLayerNorm(
                    layer.ln_2,
                    strides
                )

        # reset the scales of the encoder norms to 1
        self.encoder.ln_f.weight.data = torch.ones_like(self.encoder.ln_f.weight.data)
        if self.encoder.ln_f.bias is not None:
            self.encoder.ln_f.bias.data = torch.zeros_like(self.encoder.ln_f.bias.data)

        # replace the output norms in the decoder to seperate z and lm outputs
        self.lm_norm = self.decoder.ln_f
        self.z_norm = nn.LayerNorm(
            self.hidden_size,
            eps=base_model.config.layer_norm_epsilon,
        )

        self.decoder.ln_f = nn.Identity()
        
        # create the encoder io linears
        self.encoder_noise_proj_in = nn.Linear(self.latent_size, self.hidden_size, bias=False)
        self.encoder_mu_proj_out = nn.Linear(self.hidden_size, self.latent_size, bias=False)

        self.encoder_noise_proj_in.weight.data *= embed_std[..., None]
        self.encoder_mu_proj_out.weight.data *= self.config.z_init_scale

        # create the decoder io linears
        self.decoder_z_proj_in = nn.Linear(self.latent_size, self.hidden_size, bias=False)
        self.decoder_mu_proj_out = nn.Linear(self.hidden_size, self.latent_size, bias=False)

        self.decoder_z_proj_in.weight.data *= embed_std[..., None]
        self.decoder_mu_proj_out.weight.data *= self.config.z_init_scale

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
            self.num_z,
            self.latent_size,
        ).to(input_tokens)

        # get the encoder input
        encoder_hidden_states = torch.cat(
            [
                input_tokens,
                expand_to_batch(self.encoder_sep_token, input_tokens),
                output_tokens,
                expand_to_batch(self.encoder_z_tokens[:1], input_tokens),
                expand_to_batch(self.encoder_z_tokens[1:], input_tokens) + self.encoder_noise_proj_in(noise[..., :-1, :]),
            ],
            dim=-2
        )

        # pass through the encoder
        encoder_hidden_states = self.encoder(
            inputs_embeds=encoder_hidden_states,
        ).last_hidden_state

        # get the latent distribution from the encoder output
        encoder_mus = self.encoder_mu_proj_out(
            encoder_hidden_states[..., -self.num_z:, :]
        )
        z = noise + encoder_mus

        # get the decoder input
        decoder_hidden_states = torch.cat(
            [
                input_tokens,
                expand_to_batch(self.decoder_z_tokens[:1], input_tokens),
                expand_to_batch(self.decoder_z_tokens[1:], input_tokens) + self.decoder_z_proj_in(z),
                expand_to_batch(self.decoder_start_output_token, input_tokens),
                output_tokens[..., :-1, :],
            ],
            dim=-2
        )

        # pass through the decoder
        decoder_hidden_states = self.decoder(
            inputs_embeds=decoder_hidden_states,
        ).last_hidden_state

        # get the decoder latent distribution
        decoder_mus = self.decoder_mu_proj_out(
            self.z_norm(
                decoder_hidden_states[..., self.input_length:(self.input_length + self.num_z), :]
            )
        )

        # get the decoder lm output
        decoder_lm_logits = self.lm_head(
            self.lm_norm(
                decoder_hidden_states[..., -self.output_length:, :]
            )
        )
        decoder_lm_logits = F.log_softmax(decoder_lm_logits, dim=-1)

        return DotDict(
            encoder_mus=encoder_mus,
            decoder_mus=decoder_mus,
            lm_logits=decoder_lm_logits,
        )
    