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
                old_norm.weight.data.clone().detach() for _ in range(self.num_strides)
            ]
        )

        self.index = None
    

    def _get_scales(self):

        out = []
        for i in range(self.num_strides):
            scale = self.scales[i]
            stride = self.strides[i]

            expanded_scale = scale[None].expand(stride, *([-1] * self.num_d))
            out.append(expanded_scale)
        
        return torch.cat(out, dim=0)


    def forward(self, x):

        x = self.norm(x)

        if self.index is None:
            scales = self._get_scales()
            scales = unsqueeze_to_batch(scales, x)
        else:
            scales = self.scales[self.index]
            scales = unsqueeze_to_batch(scales, x)

        return x * scales


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
        self.num_z = self.z_length - 1

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

        self.hidden_size = base_model.config.hidden_size

        # copy token embeddings (clone to untie)
        self.embed_tokens = base_model.model.embed_tokens
        self.embed_tokens.weight = nn.Parameter(self.embed_tokens.weight.data.clone().detach())

        self.lm_head = base_model.lm_head
        self.lm_head.weight = nn.Parameter(self.lm_head.weight.data.clone().detach())

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
        self.encoder = LlamaModel(base_model.model.config)
        self.decoder = LlamaModel(base_model.model.config)

        # copy the encoder and decoder params from the base model
        transformers_with_strides = [
            (self.encoder, [self.input_length, self.output_length + 1, self.num_z]),
            (self.decoder, [self.input_length, self.z_length, self.output_length]),
        ]
        for transformer, strides in transformers_with_strides:
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

        # reset the scales of the encoder norms to 1
        self.encoder.norm.weight.data = torch.ones_like(self.encoder.norm.weight.data)

        # replace the output norms in the decoder to seperate z and lm outputs
        self.lm_norm = self.decoder.norm
        self.z_norm = nn.RMSNorm(
            self.hidden_size,
            eps=base_model.config.rms_norm_eps
        )

        self.decoder.norm = nn.Identity()
        
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


    def _set_norm_index(self, index):
        for m in self.modules():
            if isinstance(m, ModulatingRMSNorm):
                m.index = index


    @torch.no_grad()
    def sample(
        self,
        input_ids: torch.LongTensor,
        temperature: float = 1.0,
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

            z_prev = noise[..., i:i+1, :] + mu
            mus.append(mu)
            cache = outputs.past_key_values

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

        return output_tokens, torch.cat(mus, dim=-2)
    