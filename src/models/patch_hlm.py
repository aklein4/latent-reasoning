import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.xla import XLAModel
from models.hlm import (
    HLmConfig,
    HLmEncoderLayer,
    HLmDecoderLayer,
    HLmGeneratorLayer,
)


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


class UnconditionalIO(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-5, elementwise_affine=True)
        self.filter = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def enter(self, x):
        return self.norm(x)
    
    def exit(self, hidden_states, y):
        return hidden_states + self.filter * y


class ConditionalIO(nn.Module):

    def special_init(self, config: HLmConfig): 
        self.scale.weight.data.zero_()
        self.bias.weight.data.zero_()
        self.filter.weight.data.zero_()
        self.scale.special_inited = True
        self.bias.special_inited = True
        self.filter.special_inited = True


    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-5, elementwise_affine=False)
        self.scale = nn.Embedding(2, hidden_size)
        self.bias = nn.Embedding(2, hidden_size)
        self.filter = nn.Embedding(2, hidden_size)
    

    def enter(self, x, mask):
        return (
            self.bias(mask) + 
            (1+self.scale(mask)) * self.norm(x)
        )
    def exit(self, hidden_states, y, mask):
        return (
            hidden_states +
            self.filter(mask) * y
        )


class PatchHLmEncoder(nn.Module):

    def __init__(self, config: PatchHLmConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.z_size = config.z_size
        self.num_layers = config.num_layers
        self.patch_size = config.patch_size

        self.embs = nn.Embedding(config.vocab_size, config.hidden_size)
        self.patch_proj = nn.Conv1d(
            self.patch_size*config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            groups=self.patch_size,
            bias=False
        )

        self.layers = nn.ModuleList([
            HLmEncoderLayer(config, i)
            for i in range(config.num_layers)
        ])

    
    def forward(
        self,
        input_ids,
        mask,
        noise
    ):
        bs, seq_len, _ = input_ids.shape
        long_mask = mask.long()
        
        hidden_states = self.embs(input_ids).view(bs*seq_len, self.patch_size*self.hidden_size, 1)
        hidden_states = self.patch_proj(hidden_states).view(bs, seq_len, self.hidden_size)

        # mask out conditionals
        attn_mask = torch.zeros(1, 1, seq_len, seq_len, device=input_ids.device, dtype=hidden_states.dtype)
        attn_mask = torch.where(
            mask.unsqueeze(1).unsqueeze(1), # [bs, 1=head, 1=q, seq_len=k]
            torch.zeros_like(attn_mask),
            torch.full_like(attn_mask, float('-inf'))
        )
        attn_mask = self.layers[0].get_iaf_attn_mask(attn_mask)

        # pad noise for last layer iaf heads
        padded_noise = torch.cat([noise, torch.zeros_like(noise[:, :, -1:])], dim=2)

        zs = []
        mus = []
        sigmas = []
        for i, layer in enumerate(self.layers):
            
            hidden_states, z, mu, sigma = layer(
                hidden_states,
                long_mask,
                noise[:, :, i],
                padded_noise[:, :, i+1],
                attn_mask
            )

            zs.append(z)
            mus.append(mu)
            sigmas.append(sigma)
        
        return (
            torch.stack(zs, dim=2),
            torch.stack(mus, dim=2),
            torch.stack(sigmas, dim=2)
        )
    

class PatchHLmDecoder(nn.Module):

    def __init__(self, config: PatchHLmConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.z_size = config.z_size
        self.num_layers = config.num_layers
        self.patch_size = config.patch_size

        self.embs = nn.Embedding(1, config.hidden_size)

        self.layers = nn.ModuleList([
            HLmDecoderLayer(config, i)
            for i in range(config.num_layers)
        ])

        # outputs
        self.norm = nn.LayerNorm(config.hidden_size, config.norm_eps, elementwise_affine=True)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.patch_proj = nn.Linear(config.hidden_size, self.patch_size*config.hidden_size, bias=False)


    def forward(
        self,
        input_ids,
        mask,
        z,
    ):
        bs, seq_len, _ = input_ids.shape
        long_mask = mask.long()
        
        hidden_states = self.embs(torch.zeros_like(input_ids[:, :, 0]))
        
        # mask out conditionals
        attn_mask = torch.zeros(1, 1, seq_len, seq_len, device=input_ids.device, dtype=hidden_states.dtype)
        attn_mask = torch.where(
            mask.unsqueeze(1).unsqueeze(1), # [bs, 1=head, 1=q, seq_len=k]
            torch.zeros_like(attn_mask),
            torch.full_like(attn_mask, float('-inf'))
        )

        for i, layer in enumerate(self.layers):
            
            hidden_states = layer(
                hidden_states,
                long_mask,
                z[:, :, i],
                attn_mask
            )

        hidden_states = self.norm(hidden_states)
        patch_states = self.patch_proj(hidden_states).view(bs, seq_len, self.patch_size, self.hidden_size)

        lm_logits = self.lm_head(patch_states)
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        return lm_logits


class PatchHLmGenerator(nn.Module):

    def __init__(self, config: PatchHLmConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.z_size = config.z_size
        self.num_layers = config.num_layers
        self.patch_size = config.patch_size

        self.embs = nn.Embedding(1+config.vocab_size, config.hidden_size)
        self.patch_proj = nn.Conv1d(
            self.patch_size*config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            groups=self.patch_size,
            bias=False
        )

        self.layers = nn.ModuleList([
            HLmGeneratorLayer(config, i)
            for i in range(config.num_layers)
        ])


    def forward(
        self,
        input_ids,
        mask,
        z,
    ):
        bs, seq_len, _ = input_ids.shape
        long_mask = mask.long()
        
        # mask is zero, otherwise keep conditionals
        hidden_states = torch.where(
            mask.unsqueeze(-1).unsqueeze(-1),
            self.embs(torch.zeros_like(input_ids)),
            self.embs(input_ids+1)
        ).view(bs*seq_len, self.patch_size*self.hidden_size, 1)
        hidden_states = self.patch_proj(hidden_states).view(bs, seq_len, self.hidden_size)
        
        # we don't need an attention mask (yet)
        attn_mask = None

        mus = []
        for i, layer in enumerate(self.layers):
            
            hidden_states, mu = layer(
                hidden_states,
                long_mask,
                attn_mask,
                z=z[:, :, i],
            )

            mus.append(mu)

        return torch.stack(mus, dim=2)


class PatchHLmModel(XLAModel):

    config_class = HLmConfig


    def _init_weights(self, module):

        if hasattr(module, 'special_inited') and module.special_inited:
            return
        
        if hasattr(module, 'special_init'):
            module.special_init(self.config)

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(0.0, 1/np.sqrt(module.weight.shape[1]))
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(0.0, 1.0)


    def __init__(self, config: PatchHLmConfig, fast_start=False):
        super().__init__(config, fast_start=fast_start)

        self.z_size = config.z_size
        self.num_layers = config.num_layers
        self.patch_size = config.patch_size

        self.encoder = PatchHLmEncoder(config)
        self.decoder = PatchHLmDecoder(config)
        self.generator = PatchHLmGenerator(config)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids,
        mask
    ):
        input_ids = input_ids.view(
            input_ids.shape[0],
            input_ids.shape[1]//self.patch_size,
            self.patch_size
        )
        mask = mask.view(
            mask.shape[0],
            mask.shape[1]//self.patch_size,
            self.patch_size
        ).any(dim=-1)

        bs, seq_len, _ = input_ids.shape

        # sample noise for the encoder
        noise = torch.randn(
            [bs, seq_len, self.num_layers, self.z_size],
            device=input_ids.device, dtype=self.encoder.embs.weight.dtype
        )

        z, enc_mu, enc_sigma = self.encoder(input_ids, mask, noise)
        lm_logits = self.decoder(input_ids, mask, z)
        gen_mu = self.generator(input_ids, mask, z)

        kl = (
            -torch.log(enc_sigma)
            + 0.5 * (enc_sigma**2 + (enc_mu-gen_mu)**2)
            - 0.5
        ).sum(-1).sum(-1).sum(-1)

        uncond_kl = (
            -torch.log(enc_sigma)
            + 0.5 * (enc_sigma**2 + enc_mu**2)
            - 0.5
        ).sum(-1).sum(-1).sum(-1)

        lm_logits = lm_logits.view(bs, seq_len*self.patch_size, -1)

        return lm_logits, kl, uncond_kl
