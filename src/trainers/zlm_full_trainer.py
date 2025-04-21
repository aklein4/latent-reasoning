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


    def train_step(self, step, model, input_ids, output_ids):
        bs = input_ids.shape[0]

        do_negative = step < self.negative_steps

        # get model predictions
        model_out = model(
            input_ids,
            output_ids,
            do_negative=do_negative,
        )

        # extract probabilities
        logp = torch.take_along_dim(
            model_out.lm_logits,
            output_ids[..., None],
            dim=-1,
        )[..., 0]

        # calculate lm metrics
        results = DotDict(
            do_negative=float(do_negative),

            lm_loss = -logp.mean(),
            lm_pcorr = logp.exp().mean(),
            lm_acc = (model_out.lm_logits.argmax(-1) == output_ids).float().mean(),
        )

        # get weighted lm loss
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

        # calculate kl metrics
        kl = (
            model_out.encoder_mus - model_out.generator_mus
        ).pow(2).sum(-2) / 2

        mean_mus = model_out.encoder_mus.mean(0, keepdim=True)
        mean_kl = (
            model_out.encoder_mus - mean_mus
        ).pow(2).sum(-2).detach() / 2

        results.kl_per_token = (kl.mean(0).sum() / model.output_length)
        results.kl_per_channel = (kl.mean() / model.latent_size_per_layer)
        results.mean_kl_per_token = (mean_kl.mean(0).sum() / model.output_length)
        results.elbo = results.lm_loss + results.kl_per_token

        # kl grad scaling
        results.kl_minimum = self.kl_min_start * (1e-6 + 1 - min(1.0, step / self.kl_min_steps))
        results.kl_grad_scale = torch.clip((results.kl_per_token - results.kl_minimum) / results.kl_minimum, 0.0, 1.0).detach().item()
        
        kl = (
            scale_gradient(model_out.encoder_mus, results.kl_grad_scale) -
            model_out.generator_mus
        ).pow(2).sum(-2) / 2

        # save the running metrics
        kl_per_channel_mean = (kl.mean(0) / model.latent_size_per_layer).detach().clone().float()

        if self.running_kls_per_channel is None:
            self.running_kls_per_channel = kl_per_channel_mean
        else:
            self.running_kls_per_channel = self.running_kls_per_channel * self.running_beta + kl_per_channel_mean * (1 - self.running_beta)

        # balance the hidden kl weights
        sequence_kl_weights = self.running_kls_per_channel / (self.running_kls_per_channel.mean() + 1e-7)
        results.kl_per_token_weighted = (kl.mean(0) * sequence_kl_weights).sum() / model.output_length

        results.loss = (
            results.lm_loss_scaled +
            results.kl_per_token_weighted * self.kl_scale
        )

        # do contrastive loss
        negative_kl = (
            model_out.encoder_mus - model_out.generator_mus.flip(0)
        ).pow(2).sum(-2) / 2
        results.negative_kl_per_token = (negative_kl.mean(0).sum() / model.output_length)
        
        contrastive_kl = -F.logsigmoid(
            (
                (negative_kl.view(bs, -1).sum(-1) / model.output_length) - 
                (kl.view(bs, -1).sum(-1) / model.output_length)
            ) / self.contrast_temp
        )

        results.contrast_scale = self.contrast_scale * (1e-7 + 1.0 - min(1.0, step / self.contrast_steps))
        results.contrastive_kl_loss = contrastive_kl.mean() * results.contrast_scale

        if do_negative:
            negative_logp = torch.take_along_dim(
                model_out.negative_lm_logits,
                output_ids[..., None],
                dim=-1,
            )[..., 0]


            gap = logp.mean() - negative_logp.mean()
            gap_mask = 1.0 - (gap >= self.target_gap).float()

            results.gap_acc = gap_mask.mean()
            results.gap_loss = -gap.mean()
            results.gap_loss_masked = -(gap * gap_mask).mean()
            
            results.gap_scale = self.gap_scale * (1e-7 + 1.0 - min(1.0, step / self.negative_steps))
            results.loss = results.loss + results.gap_loss_masked * results.gap_scale

        if step % self.log_image_interval == 0:

            results.kl_weights = Image(
                self.running_kls_per_channel.cpu().numpy().reshape(model.z_length, model.num_latent_layers).T / self.running_kls_per_channel.max().item(),
                mode='L'
            )

            results.mean_kl_weights = Image(
                mean_kl.mean(0).detach().cpu().numpy().reshape(model.z_length, model.num_latent_layers).T / mean_kl.mean(0).detach().max().item(),
                mode='L'
            )

            enc_mus_to_plot = model.shaper.unlayerfy(model_out.encoder_mus).detach()
            gen_mus_to_plot = model.shaper.unlayerfy(model_out.generator_mus).detach()
            dists = torch.cdist(
                gen_mus_to_plot,
                enc_mus_to_plot,
            ).mean(0)
            quant = torch.quantile(dists.flatten(), 0.90, dim=-1).item()

            results.mus_dists = Image(
                np.clip(dists.cpu().numpy() / quant, 0.0, 1.0),
                mode='L'
            )

        return results
    