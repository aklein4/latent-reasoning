import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from wandb import Image

from trainers.base_trainer import BaseTrainer
from utils.model_utils import scale_gradient
from utils.dot_dict import DotDict


class ZLmHybridTrainer(BaseTrainer):
    
    inited = False

    hooked = False
    hooked_steps = 0


    def extra_init(self):
        self.hooked = self.init_hooked
        self.hooked_steps = self.init_hooked_steps


    def train_step(self, step, model, input_ids, output_ids):
        if not self.inited:
            self.extra_init()
            self.inited = True

        # calculate alpha
        alpha_prog = 1 - (
            np.sin(
                min(1.0, self.hooked_steps / self.alpha_steps) * np.pi - (np.pi / 2)
            ) + 1
        ) / 2
        alpha = 1e-7 + alpha_prog * np.sqrt(
            self.base_alpha
            * 2 / (model.total_latent_size * model.z_length / model.output_length)
        )
        
        # get model predictions
        model_out = model(input_ids, output_ids, disable_generator=(not self.hooked), alpha=alpha)

        # extract probabilities
        logp = torch.take_along_dim(
            model_out.lm_logits,
            output_ids[..., None],
            dim=-1,
        )[..., 0]

        # calculate lm metrics
        results = DotDict(
            alpha = alpha,

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

        if self.hooked:
            self.hooked_steps += 1
        results.hooked = 1.0 if self.hooked else 0.0

        get_kl = lambda a, b: (
            a - b
        ).pow(2).sum(-2) / 2

        # calculate kl metrics
        kl = get_kl(model_out.encoder_mus_base, model_out.generator_mus)
        kl_true = get_kl(model_out.encoder_mus, model_out.generator_mus)

        kl_mean = get_kl(
            model_out.encoder_mus_base,
            model_out.encoder_mus_base.mean(0, keepdim=True)
        )
        kl_mean_true = get_kl(
            model_out.encoder_mus,
            model_out.encoder_mus.mean(0, keepdim=True)
        )

        kl_mean_extra = get_kl(
            model_out.encoder_mus_extra,
            model_out.encoder_mus_extra.mean(0, keepdim=True)
        )
            
        results.kl_per_token = (kl.mean(0).sum() / model.output_length)
        results.kl_per_channel = (kl.mean() / model.latent_size_per_layer)
        
        results.kl_per_token_true = (kl_true.mean(0).sum() / model.output_length)
        results.kl_per_channel_true = (kl_true.mean() / model.latent_size_per_layer)

        results.kl_per_token_mean = (kl_mean.mean(0).sum() / model.output_length)
        results.kl_per_channel_mean = (kl_mean.mean() / model.latent_size_per_layer)

        results.kl_per_token_mean_true = (kl_mean_true.mean(0).sum() / model.output_length)
        results.kl_per_channel_mean_true = (kl_mean_true.mean() / model.latent_size_per_layer)

        results.kl_per_token_mean_extra = (kl_mean_extra.mean(0).sum() / model.output_length)
        results.kl_per_channel_mean_extra = (kl_mean_extra.mean() / model.latent_size_per_layer)

        results.elbo = results.lm_loss + results.kl_per_token

        # calculate kl loss
        normalized_kl_weights = kl.mean(0) / (kl.mean(0).max() + 1e-7)
        clipped_kl_weights = torch.clip(
            normalized_kl_weights - self.kl_weight_threshold,
            min=0.0,
        )

        kl_val_per_token = (kl.mean(0) * clipped_kl_weights).sum() / model.output_length
        kl_val_multiplier = results.kl_per_token / (1e-7 + kl_val_per_token)
        final_kl_weights = clipped_kl_weights * kl_val_multiplier.detach().item()

        kl_to_loss = (
            scale_gradient(model_out.encoder_mus_base, final_kl_weights[None, :, None, :].detach()) -
            model_out.generator_mus
        ).pow(2).sum(-2) / 2
        results.kl_loss = kl_to_loss.mean(0).sum() / model.output_length

        if not self.hooked:
            results.kl_loss = results.kl_loss.detach()

        results.kl_scale = self.kl_scale * (1 + np.sin(
            min(1.0, self.hooked_steps / self.hook_warmup_steps) * np.pi - (np.pi / 2)
        )) / 2

        results.loss = (
            results.lm_loss_scaled +
            results.kl_loss * results.kl_scale
        )

        # get the latent usage
        try:
            p_kl = kl.mean(0) / (kl.mean(0).sum() + 1e-7)
            results.effective_parties = (1 / (p_kl ** 2).sum().item()) / p_kl.numel()

            p_true_kl = kl_true.mean(0) / (kl_true.mean(0).sum() + 1e-7)
            results.effective_parties_true = (1 / (p_true_kl ** 2).sum().item()) / p_true_kl.numel()

            p_mean_kl = kl_mean.mean(0) / (kl_mean.mean(0).sum() + 1e-7)
            results.effective_parties_mean = (1 / (p_mean_kl ** 2).sum().item()) / p_mean_kl.numel()

            p_mean_true_kl = kl_mean_true.mean(0) / (kl_mean_true.mean(0).sum() + 1e-7)
            results.effective_parties_mean_true = (1 / (p_mean_true_kl ** 2).sum().item()) / p_mean_true_kl.numel()

            p_mean_extra_kl = kl_mean_extra.mean(0) / (kl_mean_extra.mean(0).sum() + 1e-7)
            results.effective_parties_mean_extra = (1 / (p_mean_extra_kl ** 2).sum().item()) / p_mean_extra_kl.numel()
        except ZeroDivisionError:
            pass

        if step % self.log_image_interval == 0:

            results.kl_levels = Image(
                kl.mean(0).detach().cpu().numpy().reshape(model.z_length, model.num_latent_layers).T / kl.mean(0).detach().max().item(),
                mode='L'
            )

            results.kl_levels_true = Image(
                kl_true.mean(0).detach().cpu().numpy().reshape(model.z_length, model.num_latent_layers).T / kl_true.mean(0).detach().max().item(),
                mode='L'
            )

            results.kl_levels_mean = Image(
                kl_mean.mean(0).detach().cpu().numpy().reshape(model.z_length, model.num_latent_layers).T / kl_mean.mean(0).detach().max().item(),
                mode='L'
            )

            results.kl_levels_mean_true = Image(
                kl_mean_true.mean(0).detach().cpu().numpy().reshape(model.z_length, model.num_latent_layers).T / kl_mean_true.mean(0).detach().max().item(),
                mode='L'
            )

            results.kl_levels_mean_extra = Image(
                kl_mean_extra.mean(0).detach().cpu().numpy().reshape(model.z_length, model.num_latent_layers).T / kl_mean_extra.mean(0).detach().max().item(),
                mode='L'
            )

            # results.zero_kl_levels = Image(
            #     zero_kl.mean(0).detach().cpu().numpy().reshape(model.z_length, model.num_latent_layers).T / zero_kl.mean(0).detach().max().item(),
            #     mode='L'
            # )

            enc_mus_to_plot = model.shaper.unlayerfy(model_out.encoder_mus).detach()
            gen_mus_to_plot = model.shaper.unlayerfy(model_out.generator_mus).detach()

            dists = torch.cdist(
                gen_mus_to_plot,
                enc_mus_to_plot,
            ).mean(0)
            quant = torch.quantile(dists.flatten(), 0.90, dim=-1).item()
            results.mu_dists = Image(
                np.clip(dists.detach().cpu().numpy() / quant, 0.0, 1.0),
                mode='L'
            )

            sims = torch.cdist(
                enc_mus_to_plot,
                enc_mus_to_plot,
            ).mean(0)
            sim_quant = torch.quantile(sims.flatten(), 0.90, dim=-1).item()
            results.mu_sims_true = Image(
                np.clip(sims.detach().cpu().numpy() / sim_quant, 0.0, 1.0),
                mode='L'
            )

            sims = torch.cdist(
                model.shaper.unlayerfy(model_out.encoder_mus_base).detach(),
                model.shaper.unlayerfy(model_out.encoder_mus_base).detach(),
            ).mean(0)
            sim_quant = torch.quantile(sims.flatten(), 0.90, dim=-1).item()
            results.mu_sims = Image(
                np.clip(sims.detach().cpu().numpy() / sim_quant, 0.0, 1.0),
                mode='L'
            )

            sims = torch.cdist(
                model.shaper.unlayerfy(model_out.encoder_mus_extra).detach(),
                model.shaper.unlayerfy(model_out.encoder_mus_extra).detach(),
            ).mean(0)
            sim_quant = torch.quantile(sims.flatten(), 0.90, dim=-1).item()
            results.mu_sims_extra = Image(
                np.clip(sims.detach().cpu().numpy() / sim_quant, 0.0, 1.0),
                mode='L'
            )

        return results
    