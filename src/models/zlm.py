import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, LlamaConfig, LlamaModel, LlamaForCausalLM


class ZLMConfig(LlamaConfig):
    """
    Configuration class for ZLM model.
    This is a subclass of LlamaConfig with additional parameters specific to ZLM.
    """
    model_type = "zlm"
    
    def __init__(
        self,
        latent_size: int = 128,
        use_kl_loss: bool = True,
        latent_activation: str = "tanh",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.latent_size = latent_size
        self.use_kl_loss = use_kl_loss
        self.latent_activation = latent_activation


class ModulatingRMSNorm(nn.Module):

    def __init__(self, old_norm, in_length, z_length, out_length):
        super().__init__()

        self.normalized_shape = old_norm.normalized_shape
        self.num_d = len(self.normalized_shape)

        self.norm = nn.RMSNorm(
            normalized_shape=self.normalized_shape,
            eps=old_norm.eps,
            elementwise_affine=False
        ).to(old_norm.weight.device)

        self.in_length = in_length
        self.z_length = z_length
        self.out_length = out_length

        self.in_scale = nn.Parameter(old_norm.weight.data.clone().detach())
        self.z_scale = nn.Parameter(old_norm.weight.data.clone().detach())
        self.out_scale = nn.Parameter(old_norm.weight.data.clone().detach())
    

    def forward(self, x):
        x = self.norm(x)

        scale = torch.cat(
            [
                self.in_scale[None].expand(self.in_length, *([-1]* self.num_d)),
                self.z_scale[None].expand(self.z_length, *([-1]* self.num_d)),
                self.out_scale[None].expand(self.out_length, *([-1]* self.num_d))
            ],
            dim=0
        )
        while scale.dim() < x.dim():
            scale = scale[None]

        return x * scale


class ZLM(PreTrainedModel):
    """
    ZLM (Zero-shot Latent Model) implementation.
    """
    config_class = ZLMConfig
    
    def __init__(self, config: ZLMConfig):
        super().__init__(config)

        self.in_length = config.in_length
        self.z_length = config.z_length
        self.out_length = config.out_length

        self.hidden_size = config.hidden_size
        self.z_size = config.z_size

        base_model = LlamaForCausalLM.from_pretrained(
            config["base_url"]
        )

        self.embed_tokens = base_model.model.embed_tokens        
        self.lm_head = base_model.lm_head

        self.encoder_sep = nn.Parameter(
            self.embed_tokens.weight.data[config.sep_token_id].clone().detach()
        )
        self.encoder_z_tokens = nn.Parameter(
            self.embed_tokens.weight.data.mean(0, keepdim=True).detach() +
            (
                torch.randn(self.z_length, self.hidden_size)
                * self.embed_tokens.weight.data.std(0, keepdim=True).detach()
            )
        )

        self.decoder_intermed_tokens = nn.Parameter(
            self.embed_tokens.weight.data.mean(0, keepdim=True).detach() +
            (
                torch.randn(self.z_length, self.hidden_size)
                * self.embed_tokens.weight.data.std(0, keepdim=True).detach()
            )
        )

        self.encoder = LlamaModel(base_model.model.config)
        self.decoder = LlamaModel(base_model.model.config)

        for transformer in [self.encoder, self.decoder]:
            transformer.load_state_dict({k: v.clone() for k, v in base_model.model.state_dict()})

            for layer in transformer.layers:
                layer.input_layernorm = ModulatingRMSNorm(
                    layer.input_layernorm,
                    self.in_length,
                    self.z_length,
                    self.out_length
                )
                layer.post_attention_layernorm = ModulatingRMSNorm(
                    layer.post_attention_layernorm,
                    self.in_length,
                    self.z_length,
                    self.out_length
                )

        self.encoder_norm = nn.RMSNorm(
            self.hidden_size,
            eps=config.rms_norm_eps
        )
        self.encoder.norm.weight.data = torch.ones_like(self.encoder.norm.weight.data)

        self.lm_norm = self.decoder.norm
        self.decoder_norm = nn.RMSNorm(
            self.hidden_size,
            eps=config.rms_norm_eps
        )
        self.decoder.norm = nn.Identity()

        self.encoder_z_proj_in = nn.Linear(self.z_size, self.hidden_size, bias=True)
        self.decoder_z_proj_in = nn.Linear(self.z_size, self.hidden_size, bias=True)

        self.encoder_z_proj_out = nn.Linear(self.hidden_size, self.z_size, bias=False)
        self.decoder_z_proj_out = nn.Linear(self.hidden_size, self.z_size, bias=False)

        # Initialize weights and gradient checkpointing
        self.post_init()
    

    def init_weights(self):
        return


    def forward(
        self,
        input_ids: torch.LongTensor,
        output_ids: torch.LongTensor,
    ):
        assert input_ids.shape[-1] == self.in_length, "Input length mismatch."
        assert output_ids.shape[-1] == self.out_length, "Output length mismatch."
        assert input_ids.shape[:-1] == output_ids.shape[:-1], "Batch size mismatch."

        batch_shape = input_ids.shape[:-1]
        batch_dims = len(batch_shape)

        noise = torch.randn(
            *batch_shape, self.z_length-1, self.z_size
        ).to(input_ids.device)

        encoder_states = torch.cat(
            [
                self.embed_tokens(input_ids),
                self.encoder_sep.view(*([1] * batch_dims), -1),
                self.embed_tokens(output_ids),
                self.encoder_z_tokens.view(*([1] * batch_dims), self.encoder_z_tokens.shape[0], -1),
            ]
        )
        encoder_states[..., -(self.z_length-1):] = (
            encoder_states[..., -(self.z_length-1):].clone() +
            self.encoder_z_proj_in(noise)
        )
        encoder_states = self.encoder(encoder_states)[0]

        z = self.encoder_z_proj_out(encoder_states[..., -(self.z_length-1):])
        
        outputs = {}
        
        # Placeholder logic - to be implemented
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            latent = self.encoder(torch.randn(batch_size, self.config.hidden_size).to(input_ids.device))
            outputs["latent"] = latent
        
        return outputs
