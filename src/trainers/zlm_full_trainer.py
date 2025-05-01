import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from wandb import Image

from trainers.base_trainer import BaseTrainer
from utils.model_utils import scale_gradient
from utils.dot_dict import DotDict


class ZLmFullTrainer(BaseTrainer):
    
    running_kls_per_channel = None
    running_mean_kls_per_channel = None
    running_zeros_per_channel = None

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

        # save the running metrics
        kl_per_channel_mean = (kl.mean(0) / model.latent_size_per_layer).detach().clone().float()
        mean_kl_per_channel_mean = (mean_kl.mean(0) / model.latent_size_per_layer).detach().clone().float()
        zero_kl_per_channel_mean = (zero_kl.mean(0) / model.latent_size_per_layer).detach().clone().float()

        if self.running_kls_per_channel is None:
            self.running_kls_per_channel = kl_per_channel_mean
            self.running_mean_kls_per_channel = mean_kl_per_channel_mean
            self.running_zeros_per_channel = zero_kl_per_channel_mean
        else:
            self.running_kls_per_channel = self.running_kls_per_channel * self.running_beta + kl_per_channel_mean * (1 - self.running_beta)
            self.running_mean_kls_per_channel = self.running_mean_kls_per_channel * self.running_beta + mean_kl_per_channel_mean * (1 - self.running_beta)
            self.running_zeros_per_channel = self.running_zeros_per_channel * self.running_beta + zero_kl_per_channel_mean * (1 - self.running_beta)

        # balance the kl weights
        normalized_kl_weights = self.running_kls_per_channel / (self.running_kls_per_channel.max() + 1e-7)
        clipped_kl_weights = torch.clip(
            normalized_kl_weights - self.kl_weight_threshold,
            min=0.0,
        )
        kl_val_per_token = (kl.mean(0) * clipped_kl_weights).sum() / model.output_length
        kl_val_multiplier = results.kl_per_token / (1e-7 + kl_val_per_token)
        final_kl_weights = clipped_kl_weights * kl_val_multiplier.detach().item()

        kl_to_loss = (
            scale_gradient(
                model_out.encoder_mus.detach() if self.zero_optimize else model_out.encoder_mus,
                final_kl_weights[None, :, None, :].detach()
            ) -
            model_out.generator_mus
        ).pow(2).sum(-2) / 2
        results.kl_per_token_weighted = kl_to_loss.mean(0).sum() / model.output_length

        # balance the zero kl weights
        normalized_zero_kl_weights = self.running_zeros_per_channel / (self.running_zeros_per_channel.max() + 1e-7)
        clipped_zero_kl_weights = torch.clip(
            normalized_zero_kl_weights - self.kl_weight_threshold,
            min=0.0,
        )
        zero_kl_val_per_token = (zero_kl.mean(0) * clipped_zero_kl_weights).sum() / model.output_length
        zero_kl_val_multiplier = results.zero_kl_per_token / (1e-7 + zero_kl_val_per_token)
        final_zero_kl_weights = clipped_zero_kl_weights * zero_kl_val_multiplier.detach().item()

        zero_kl_to_loss = (
            scale_gradient(
                model_out.encoder_mus if self.zero_optimize else model_out.encoder_mus.detach(),
                final_zero_kl_weights[None, :, None, :].detach()
            ) -
            torch.zeros_like(model_out.encoder_mus)
        ).pow(2).sum(-2) / 2
        results.zero_kl_per_token_weighted = zero_kl_to_loss.mean(0).sum() / model.output_length


        results.kl_loss = results.kl_per_token_weighted + results.zero_kl_per_token_weighted
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

        p_zero_kl = self.running_zeros_per_channel / (self.running_zeros_per_channel.sum() + 1e-7)
        results.effective_parties_zero = (1 / (p_zero_kl ** 2).sum().item()) / p_zero_kl.numel()

        if step % self.log_image_interval == 0:

            results.kl_weights = Image(
                self.running_kls_per_channel.cpu().numpy().reshape(model.z_length, model.num_latent_layers).T / self.running_kls_per_channel.max().item(),
                mode='L'
            )

            results.mean_kl_weights = Image(
                self.running_mean_kls_per_channel.cpu().numpy().reshape(model.z_length, model.num_latent_layers).T / self.running_mean_kls_per_channel.max().item(),
                mode='L'
            )

            results.kl_mask = Image(
                final_kl_weights.detach().cpu().numpy().reshape(model.z_length, model.num_latent_layers).T / final_kl_weights.max().item(),
                mode='L'
            )

            results.zero_kl_mask = Image(
                final_zero_kl_weights.cpu().numpy().reshape(model.z_length, model.num_latent_layers).T / final_zero_kl_weights.max().item(),
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
    