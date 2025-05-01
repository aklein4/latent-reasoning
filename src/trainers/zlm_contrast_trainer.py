import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

import numpy as np
from wandb import Image

from trainers.base_trainer import BaseTrainer
from utils.model_utils import scale_gradient
from utils.dot_dict import DotDict


class ZLmContrastTrainer(BaseTrainer):
    
    def train_step(self, step, model, input_ids, output_ids):

        # get model predictions
        model_out = model(input_ids, output_ids)

        # extract probabilities
        logp = torch.take_along_dim(
            model_out.lm_logits,
            output_ids[..., None],
            dim=-1,
        )[..., 0]

        # calculate lm metrics
        results = DotDict(
            lm_loss = -logp.mean(),
            lm_pcorr = logp.exp().mean(),
            lm_acc = (model_out.lm_logits.argmax(-1) == output_ids).float().mean(),
        )

        lm_mask = 1e-7 + 1 - torch.clip(
            (logp - np.log(self.lower_lm_bound)) / (np.log(self.upper_lm_bound) - np.log(self.lower_lm_bound)),
            min=0.0,
            max=1.0,
        ).detach()
        results.lm_mask = lm_mask.mean()

        results.lm_loss_scaled = -(logp * lm_mask).mean()

        # calculate kl metrics
        kl = (
            model_out.encoder_mus - model_out.generator_mus
        ).pow(2).sum(-2) / 2

        kl_control = (
            model_out.encoder_mus - model_out.encoder_mus.mean(0, keepdim=True)
        ).pow(2).sum(-2) / 2

        results.kl_per_token = (kl.mean(0).sum() / model.output_length)
        results.kl_per_channel = (kl.mean() / model.latent_size_per_layer)
        
        results.kl_per_token_control = (kl_control.mean(0).sum() / model.output_length)
        results.kl_per_channel_control = (kl_control.mean() / model.latent_size_per_layer)

        results.elbo = results.lm_loss + results.kl_per_token

        # get the contrastive loss
        enc_embs = einops.rearrange(
            model_out.flat_encoder_mus / (model_out.flat_encoder_mus.norm(dim=-1, keepdim=True) + 1e-7),
            "b s h -> s h b"
        )
        gen_emb = einops.rearrange(
            model_out.flat_generator_mus / (model_out.flat_generator_mus.norm(dim=-1, keepdim=True) + 1e-7),
            "b s h -> s b h"
        )

        contrast_sims = gen_emb @ enc_embs
        contrast_logits = F.log_softmax(contrast_sims * self.contrast_temperature, dim=-1)
        results.contrast_loss = -torch.diagonal(contrast_logits, dim1=-2, dim2=-1).mean()

        # balance the kl weights
        mean_kl = kl.mean(0)
        normalized_kl_weights = mean_kl / (mean_kl.max() + 1e-7)
        clipped_kl_weights = torch.clip(
            normalized_kl_weights - self.kl_weight_threshold,
            min=0.0,
        )
        kl_val_per_token = (kl.mean(0) * clipped_kl_weights).sum() / model.output_length
        kl_val_multiplier = results.kl_per_token / (1e-7 + kl_val_per_token)
        final_kl_weights = clipped_kl_weights * kl_val_multiplier.detach().item()

        kl_to_loss = (
            scale_gradient(model_out.encoder_mus, final_kl_weights[None, :, None, :].detach()) -
            model_out.generator_mus
        ).pow(2).sum(-2) / 2
        results.kl_loss = kl_to_loss.mean(0).sum() / model.output_length

        results.contrast_scale = self.contrast_scale * (1e-7 + 1 - min(
            1.0, step / self.contrast_steps
        ))
        results.kl_scale = self.kl_scale * min(
            1.0, step / self.contrast_steps
        )

        results.loss = (
            results.lm_loss_scaled +
            results.kl_loss * results.kl_scale +
            results.contrast_loss * results.contrast_scale
        )

        # get the latent usage
        p_kl = mean_kl / (mean_kl.sum() + 1e-7)
        results.effective_parties = (1 / (p_kl ** 2).sum().item()) / p_kl.numel()

        p_mean_kl = kl_control.mean(0) / (kl_control.mean(0).sum() + 1e-7)
        results.effective_parties_control = (1 / (p_mean_kl ** 2).sum().item()) / p_mean_kl.numel()

        if step % self.log_image_interval == 0:

            results.kl_weights = Image(
                mean_kl.cpu().detach().numpy().reshape(model.z_length, model.num_latent_layers).T / mean_kl.max().item(),
                mode='L'
            )

            results.kl_weights_control = Image(
                kl_control.mean(0).cpu().detach().numpy().reshape(model.z_length, model.num_latent_layers).T / kl_control.mean(0).max().item(),
                mode='L'
            )

            results.kl_mask = Image(
                final_kl_weights.cpu().detach().numpy().reshape(model.z_length, model.num_latent_layers).T / final_kl_weights.max().item(),
                mode='L'
            )

            enc_mus_to_plot = model.shaper.unlayerfy(model_out.encoder_mus).detach()
            gen_mus_to_plot = model.shaper.unlayerfy(model_out.generator_mus).detach()

            dists = torch.cdist(
                gen_mus_to_plot,
                enc_mus_to_plot,
            ).mean(0)
            quant = torch.quantile(dists.flatten(), 0.90, dim=-1).item()
            results.mu_dists = Image(
                np.clip(dists.cpu().detach().numpy() / quant, 0.0, 1.0),
                mode='L'
            )

            sims = torch.cdist(
                enc_mus_to_plot,
                enc_mus_to_plot,
            ).mean(0)
            sim_quant = torch.quantile(sims.flatten(), 0.90, dim=-1).item()
            results.mu_sims = Image(
                np.clip(sims.cpu().detach().numpy() / sim_quant, 0.0, 1.0),
                mode='L'
            )

        return results
    