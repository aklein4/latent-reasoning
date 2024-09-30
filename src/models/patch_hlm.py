import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.xla import XLAModel
from models.hlm import (
    HLmConfig,
    HLmEncGen,
    HLmDecoder,
)
import utils.constants as constants


class PatchHLmConfig(HLmConfig):
    """
    Args:
        patch_size (`int`):
            The size of the patches to be used in the model.
    """

    model_type = 'hlm'

    def __init__(
        self,
        patch_size=None,
        *args,
        **kwargs,
    ):

        self.patch_size = patch_size

        super().__init__(*args, **kwargs)


class PatchLmHead(nn.Module):

    def special_init(self, config: PatchHLmConfig):

        self.lm_head.weight.data.normal_(
            0.0,
            1/np.sqrt(config.hidden_size // config.patch_size)
        )
        if self.lm_head.bias is not None:
            self.lm_head.bias.data.zero_()
        
        self.lm_head.special_inited = True


    def __init__(self, config: PatchHLmConfig):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        assert self.hidden_size % self.patch_size == 0

        self.vocab_size = config.vocab_size
        
        self.norm = nn.LayerNorm(self.hidden_size, eps=config.norm_eps, elementwise_affine=True)
        self.lm_head = nn.Conv1d(
            self.hidden_size,
            self.patch_size*self.vocab_size,
            kernel_size=1,
            groups=self.patch_size,
            bias=False
        )


    def forward(self, hidden_states):
        bs, seq_len = hidden_states.shape[:2]

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(-1, self.hidden_size, 1)
        if constants.XLA_AVAILABLE:
            hidden_states = hidden_states.to(torch.bfloat16)

        log_master_print(f"Before: {hidden_states.dtype}")
        lm_logits = self.lm_head(hidden_states)
        log_master_print(f"After: {lm_logits.dtype}")
        lm_logits = lm_logits.view(bs, seq_len*self.patch_size, self.vocab_size)

        out = F.log_softmax(
            lm_logits,
            dim=-1,
            dtype=(torch.bfloat16 if constants.XLA_AVAILABLE else None)
        )
        log_master_print(f"Out: {out.dtype}")

        return out

from utils.logging_utils import log_master_print

class PatchHLmEncGen(HLmEncGen):

    def init_input_params(self, config: PatchHLmConfig):
        self.patch_size = config.patch_size
        
        self.enc_embs = nn.Embedding(config.vocab_size, config.hidden_size)
        self.enc_patch_proj = nn.Linear(
            config.patch_size*config.hidden_size,
            config.hidden_size,
            bias=False
        )

        self.gen_embs = nn.Embedding(1+config.vocab_size, config.hidden_size)
        self.gen_patch_proj = nn.Linear(
            config.patch_size*config.hidden_size,
            config.hidden_size,
            bias=False
        )
    

    def get_states(self, input_ids, mask):
        bs, seq_len = input_ids.shape[:2]

        encoder_states = self.enc_embs(input_ids).view(bs, seq_len, self.patch_size*self.hidden_size)
        encoder_states = self.enc_patch_proj(encoder_states)

        generator_states = torch.where(
            mask.unsqueeze(-1).unsqueeze(-1),
            self.gen_embs(torch.zeros_like(input_ids)),
            self.gen_embs(input_ids+1)
        ).view(bs, seq_len, self.patch_size*self.hidden_size)
        generator_states = self.gen_patch_proj(generator_states)

        return encoder_states, generator_states


class PatchHLmDecoder(HLmDecoder):

    lm_head_type = PatchLmHead


class PatchHLmModel(XLAModel):

    config_class = PatchHLmConfig


    def _init_weights(self, module):

        if hasattr(module, 'special_inited') and module.special_inited:
            return
        
        if hasattr(module, 'special_init'):
            module.special_init(self.config)

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(0.0, 1/np.sqrt(module.weight.shape[1]))
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(0.0, 1.0)


    def __init__(self, config: PatchHLmConfig, fast_start=False):
        super().__init__(config, fast_start=fast_start)

        self.z_size = config.z_size
        self.num_layers = config.num_layers
        self.z_output_layers = config.z_output_layers
        self.z_output_size = config.z_output_layers * config.z_size
        self.patch_size = config.patch_size

        self.enc_gen = PatchHLmEncGen(config)
        self.decoder = PatchHLmDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids,
        mask,
        num_uncond=None
    ):
        if num_uncond is not None:
            assert num_uncond > 0
            assert num_uncond <= input_ids.shape[0]

        # reshape to patches
        input_ids = input_ids.view(
            input_ids.shape[0],
            input_ids.shape[1]//self.patch_size,
            self.patch_size
        )
        og_mask = mask
        mask = mask.view(
            mask.shape[0],
            mask.shape[1]//self.patch_size,
            self.patch_size
        ).any(dim=-1)

        bs, seq_len, _ = input_ids.shape

        # sample noise for the encoder
        noise = torch.randn(
            [bs, seq_len, self.num_layers, self.z_size],
            device=input_ids.device, dtype=self.enc_gen.enc_embs.weight.dtype
        )

        # pass through the encoder
        z, enc_mu, enc_sigma, gen_mu = self.enc_gen(input_ids, mask, noise, num_uncond=num_uncond)

        # get z for the decoder
        z_out = z[:, :, -self.z_output_layers:].view(bs, seq_len, self.z_output_size)

        # pass through the decoder
        lm_logits = self.decoder(mask, z_out)

        kl = (
            -torch.log(enc_sigma)
            + 0.5 * (enc_sigma**2 + (enc_mu-gen_mu)**2)
            - 0.5
        ).sum(-1).sum(-1)

        # expand kl to the expected shape, and remove areas that should be masked
        kl = torch.repeat_interleave(kl/self.patch_size, self.patch_size, dim=1)
        kl = torch.where(og_mask, kl, torch.zeros_like(kl))

        if num_uncond is not None:
            uncond_kl = kl[:num_uncond]
            kl = kl[num_uncond:]
            return lm_logits, kl, uncond_kl

        return lm_logits, kl
