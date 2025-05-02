import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from wandb import Image

from trainers.base_trainer import BaseTrainer
from utils.model_utils import scale_gradient
from utils.dot_dict import DotDict


class ZLmAsymTrainer(BaseTrainer):
    
    hooked = False
    hooked_steps = 0


    def __init___(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hooked = self.init_hooked
        self.hooked_steps = self.init_hooked_steps


    def train_step(self, step, model, input_ids, output_ids):

        # get model predictions
        model_out = model(input_ids, output_ids, disable_generator=(not self.hooked))

        # extract probabilities
        logp = torch.take_along_dim(
            model_out.lm_logits,
            output_ids[..., None],
            dim=-1,
        )[..., 0]

        # calculate lm metrics
        results = DotDict(
            mu_scale = model_out.mu_scale,

            lm_loss = -logp.mean(),
            lm_pcorr = logp.exp().mean(),
            lm_acc = (model_out.lm_logits.argmax(-1) == output_ids).float().mean(),
        )

        results.lm_scale = 1e-7 + 1 - torch.clip(
            (results.lm_acc - self.lower_acc_bound) / (self.upper_acc_bound - self.lower_acc_bound),
            min=0.0,
            max=1.0,
        ).detach()

        lm_mask = 1e-7 + 1 - torch.clip(
            (logp - np.log(self.lower_lm_bound)) / (np.log(self.upper_lm_bound) - np.log(self.lower_lm_bound)),
            min=0.0,
            max=1.0,
        ).detach()
        results.lm_mask = lm_mask.mean()

        results.lm_loss_scaled = -(logp * lm_mask).mean() * results.lm_scale

        # handle hook
        if not self.hooked:
            if results.lm_acc.item() >= self.acc_hook:
                self.hooked = True
                results.reset_optimizer = 1.0

                # model.generator_mu_bias.data = model.shaper.unlayerfy(
                #     model_out.encoder_mus
                # ).mean(0).clone().detach()

        if self.hooked:
            self.hooked_steps += 1
        results.hooked = 1.0 if self.hooked else 0.0

        # calculate kl metrics
        kl = (
            model_out.encoder_mus - model_out.generator_mus
        ).pow(2).sum(-2) / 2

        mean_kl = (
            model_out.encoder_mus - model_out.encoder_mus.mean(0, keepdim=True)
        ).pow(2).sum(-2) / 2

        zero_kl = (
            model_out.encoder_mus - torch.zeros_like(model_out.encoder_mus)
        ).pow(2).sum(-2) / 2

        results.kl_per_token = (kl.mean(0).sum() / model.output_length)
        results.kl_per_channel = (kl.mean() / model.latent_size_per_layer)
        
        results.mean_kl_per_token = (mean_kl.mean(0).sum() / model.output_length)
        results.mean_kl_per_channel = (mean_kl.mean() / model.latent_size_per_layer)

        results.zero_kl_per_token = (zero_kl.mean(0).sum() / model.output_length)
        results.zero_kl_per_channel = (zero_kl.mean() / model.latent_size_per_layer)

        results.elbo = results.lm_loss + results.kl_per_token

        kl_to_loss = (
            model_out.encoder_mus.detach() - model_out.generator_mus
        ).pow(2).sum(-2) / 2
        results.kl_loss = (kl_to_loss.mean(0).sum() / model.output_length)

        results.kl_scale = self.kl_scale * min(1.0, self.hooked_steps / self.hook_warmup_steps)

        results.loss = (
            results.lm_loss_scaled +
            results.kl_loss * results.kl_scale
        )

        # get the latent usage
        p_kl = self.running_kls_per_channel / (self.running_kls_per_channel.sum() + 1e-7)
        results.effective_parties = (1 / (p_kl ** 2).sum().item()) / p_kl.numel()

        p_mean_kl = self.running_mean_kls_per_channel / (self.running_mean_kls_per_channel.sum() + 1e-7)
        results.effective_parties_mean = (1 / (p_mean_kl ** 2).sum().item()) / p_mean_kl.numel()

        p_zero_kl = self.running_zero_kl_per_channel / (self.running_zero_kl_per_channel.sum() + 1e-7)
        results.effective_parties_zero = (1 / (p_zero_kl ** 2).sum().item()) / p_zero_kl.numel()

        if step % self.log_image_interval == 0:

            results.kl_levels = Image(
                kl.mean(0).cpu().numpy().reshape(model.z_length, model.num_latent_layers).T / kl.mean(0).max().item(),
                mode='L'
            )

            results.mean_kl_levels = Image(
                mean_kl.mean(0).cpu().numpy().reshape(model.z_length, model.num_latent_layers).T / mean_kl.mean(0).max().item(),
                mode='L'
            )

            results.zero_kl_levels = Image(
                zero_kl.mean(0).cpu().numpy().reshape(model.z_length, model.num_latent_layers).T / zero_kl.mean(0).max().item(),
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
                np.clip(dists.cpu().numpy() / quant, 0.0, 1.0),
                mode='L'
            )

            sims = torch.cdist(
                enc_mus_to_plot,
                enc_mus_to_plot,
            ).mean(0)
            sim_quant = torch.quantile(sims.flatten(), 0.90, dim=-1).item()
            results.mu_sims = Image(
                np.clip(sims.cpu().numpy() / sim_quant, 0.0, 1.0),
                mode='L'
            )

        return results
    