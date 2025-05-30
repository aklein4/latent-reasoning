import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from wandb import Image

from trainers.base_trainer import BaseTrainer
from utils.model_utils import scale_gradient
from utils.dot_dict import DotDict


class Roller:

    def __init__(self, beta):
        self.data = None
        self.beta = beta

    def __call__(self, x):
    
        if self.data is None:
            self.data = x
        else:
            self.data = self.beta * self.data + (1 - self.beta) * x

        return self.data


def clamped_linear(x, min_val, max_val, steps):
    """
    Clamped linear function that maps x to [min_val, max_val] over the range [0, steps].
    """
    t = np.clip(x / steps, 0, 1)
    return min_val + (max_val - min_val) * t


class ZLmTrainer(BaseTrainer):
    
    inited = False

    hooked = False
    hooked_steps = 0

    rolling_acc = None
    rolling_kl = None
    

    def extra_init(self):

        self.hooked = self.init_hooked
        self.hooked_steps = self.init_hooked_steps

        self.rolling_acc = Roller(self.acc_rolling_beta)
        self.rolling_kl = Roller(self.kl_rolling_beta)


    def train_step(self, step, model, input_ids, output_ids):
        if not self.inited:
            self.extra_init()
            self.inited = True

        # calculate alpha
        base_alpha_1 = np.sqrt(
            self.base_alpha_1
            * 2 / (model.latent_size * model.z_length / model.output_length)
        )
        base_alpha_2 = np.sqrt(
            self.base_alpha_2
            * 2 / (model.latent_size * model.z_length / model.output_length)
        )

        alpha_prog_1 = clamped_linear(
            self.hooked_steps, 1.0, 0.0, self.alpha_steps_1
        )
        alpha_prog_2 = clamped_linear(
            self.hooked_steps, 1.0, 0.0, self.alpha_steps_2
        )

        alpha = 1e-7 + (
            alpha_prog_1 * base_alpha_1 +
            alpha_prog_2 * base_alpha_2
        )
        
        # get model predictions
        model_out = model(input_ids, output_ids, alpha=alpha)

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

        rolled_acc = self.rolling_acc(results.lm_acc.item())
        results.lm_scale = self.min_lm_scale + (1 - self.min_lm_scale) * (1 - np.clip(
            (rolled_acc - self.lower_acc_bound) / (self.upper_acc_bound - self.lower_acc_bound),
            a_min=0.0,
            a_max=1.0,
        ))

        results.lm_loss_scaled = results.lm_loss * results.lm_scale

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
        ).pow(2).sum(-1) / 2

        # calculate kl metrics
        kl = get_kl(model_out.encoder_mus_base, model_out.decoder_mus)
        kl_true = get_kl(model_out.encoder_mus, model_out.decoder_mus)

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

        get_per_token = lambda x: x.mean(0).sum() / model.output_length
        get_per_channel = lambda x: x.mean() / model.latent_size
            
        results.kl_per_token = get_per_token(kl)
        results.kl_per_channel = get_per_channel(kl)
        
        results.kl_per_token_true = get_per_token(kl_true)
        results.kl_per_channel_true = get_per_channel(kl_true)

        results.kl_per_token_mean = get_per_token(kl_mean)
        results.kl_per_channel_mean = get_per_channel(kl_mean)

        results.kl_per_token_mean_true = get_per_token(kl_mean_true)
        results.kl_per_channel_mean_true = get_per_channel(kl_mean_true)

        results.kl_per_token_mean_extra = get_per_token(kl_mean_extra)
        results.kl_per_channel_mean_extra = get_per_channel(kl_mean_extra)

        results.elbo = results.lm_loss + results.kl_per_token

        # calculate kl loss
        rolled_kl = self.rolling_kl(kl.mean(0).detach())

        normalized_kl_weights = rolled_kl / (rolled_kl.mean() + 1e-7)

        kl_val_per_token = (kl.mean(0) * normalized_kl_weights).sum() / model.output_length
        kl_val_multiplier = results.kl_per_token / (1e-7 + kl_val_per_token)

        final_kl_weights = normalized_kl_weights * kl_val_multiplier.detach().item()
        if self.ignore_kl_weights:
            final_kl_weights = torch.ones_like(final_kl_weights)

        kl_to_loss = get_kl(
            scale_gradient(model_out.encoder_mus_base, final_kl_weights[None, :, None].detach()),
            model_out.decoder_mus
        )
        results.kl_loss = get_per_token(kl_to_loss)

        if not self.hooked:
            results.kl_loss = results.kl_loss.detach()

        results.loss = (
            results.lm_loss_scaled +
            results.kl_loss * self.kl_scale
        )

        # get the latent usage
        try:
            p_kl = kl.mean(0) / (kl.mean(0).sum() + 1e-7)
            results.effective_parties = (1 / (1e-7 + (p_kl ** 2).sum().item())) / p_kl.numel()

            p_true_kl = kl_true.mean(0) / (kl_true.mean(0).sum() + 1e-7)
            results.effective_parties_true = (1 / (1e-7 + (p_true_kl ** 2).sum().item())) / p_true_kl.numel()

            p_mean_kl = kl_mean.mean(0) / (kl_mean.mean(0).sum() + 1e-7)
            results.effective_parties_mean = (1 / (1e-7 + (p_mean_kl ** 2).sum().item())) / p_mean_kl.numel()

            p_mean_true_kl = kl_mean_true.mean(0) / (kl_mean_true.mean(0).sum() + 1e-7)
            results.effective_parties_mean_true = (1 / (1e-7 + (p_mean_true_kl ** 2).sum().item())) / p_mean_true_kl.numel()

            p_mean_extra_kl = kl_mean_extra.mean(0) / (kl_mean_extra.mean(0).sum() + 1e-7)
            results.effective_parties_mean_extra = (1 / (1e-7 + (p_mean_extra_kl ** 2).sum().item())) / p_mean_extra_kl.numel()
        except ZeroDivisionError:
            pass

        if step % self.log_image_interval == 0:

            results.kl_levels = Image(
                kl.mean(0).detach().cpu().numpy().reshape(model.z_length, 1).T / kl.mean(0).detach().max().item(),
                mode='L'
            )

            results.kl_levels_true = Image(
                kl_true.mean(0).detach().cpu().numpy().reshape(model.z_length, 1).T / kl_true.mean(0).detach().max().item(),
                mode='L'
            )

            results.kl_levels_mean = Image(
                kl_mean.mean(0).detach().cpu().numpy().reshape(model.z_length, 1).T / kl_mean.mean(0).detach().max().item(),
                mode='L'
            )

            results.kl_levels_mean_true = Image(
                kl_mean_true.mean(0).detach().cpu().numpy().reshape(model.z_length, 1).T / kl_mean_true.mean(0).detach().max().item(),
                mode='L'
            )

            results.kl_levels_mean_extra = Image(
                kl_mean_extra.mean(0).detach().cpu().numpy().reshape(model.z_length, 1).T / kl_mean_extra.mean(0).detach().max().item(),
                mode='L'
            )

            # results.zero_kl_levels = Image(
            #     zero_kl.mean(0).detach().cpu().numpy().reshape(model.z_length, 1).T / zero_kl.mean(0).detach().max().item(),
            #     mode='L'
            # )

            enc_mus_to_plot = model_out.encoder_mus.detach()
            gen_mus_to_plot = model_out.decoder_mus.detach()

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
                model_out.encoder_mus_base.detach(),
                model_out.encoder_mus_base.detach(),
            ).mean(0)
            sim_quant = torch.quantile(sims.flatten(), 0.90, dim=-1).item()
            results.mu_sims = Image(
                np.clip(sims.detach().cpu().numpy() / sim_quant, 0.0, 1.0),
                mode='L'
            )

            sims = torch.cdist(
                model_out.encoder_mus_extra.detach(),
                model_out.encoder_mus_extra.detach(),
            ).mean(0)
            sim_quant = torch.quantile(sims.flatten(), 0.90, dim=-1).item()
            results.mu_sims_extra = Image(
                np.clip(sims.detach().cpu().numpy() / sim_quant, 0.0, 1.0),
                mode='L'
            )

        return results
    