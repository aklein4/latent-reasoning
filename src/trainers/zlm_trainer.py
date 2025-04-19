import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from wandb import Image

from trainers.base_trainer import BaseTrainer
from utils.model_utils import scale_gradient
from utils.dot_dict import DotDict


class ZLmTrainer(BaseTrainer):
    
    running_hidden_kls_per_channel = None
    running_output_kls_per_channel = None


    def train_step(self, step, model, input_ids, output_ids, target):
        bs = input_ids.shape[0]
        target_size = target[0].numel()

        noise_scale = min(1.0, step / self.noise_scale_steps) if self.noise_scale_steps is not None else 1.0

        # get model predictions
        model_out = model(input_ids, output_ids, target, noise_scale=noise_scale)

        encoder_mus = model_out.encoder_mus
        generator_mus = model_out.generator_mus
        output = model_out.output.reshape(bs, -1)
        target = target.reshape(bs, -1)

        # calculate kl metrics
        hidden_kl = (
            encoder_mus - generator_mus
        ).pow(2).sum(-2) / 2
        hidden_kl = hidden_kl.reshape(bs, -1)

        output_kl = (
            output - target
        ).pow(2).sum(-1)[..., None] / 2

        mean_hidden_kl = (
            encoder_mus - encoder_mus.mean(0, keepdim=True)
        ).pow(2).sum(-2) / 2

        mean_output_kl = (
            target - target.mean(0, keepdim=True)
        ).pow(2).sum(-1)[..., None] / 2

        results = DotDict(
            noise_scale=noise_scale,
            hidden_kl_per_token = (hidden_kl.mean(0).sum() / model.output_length),
            output_kl_per_token = (output_kl.mean(0).sum() / model.output_length),
            hidden_kl_per_channel = (hidden_kl.mean() / model.latent_size_per_layer),
            output_kl_per_channel = (output_kl.mean() / target_size),

            mean_hidden_kl_per_token = (mean_hidden_kl.mean(0).sum() / model.output_length),
            mean_output_kl_per_token = (mean_output_kl.mean(0).sum() / model.output_length),
        )
        results.total_kl_per_token = results.hidden_kl_per_token + results.output_kl_per_token

        # save the running metrics
        hidden_kl_per_channel_mean = (hidden_kl.mean(0) / model.latent_size_per_layer).detach().clone().float()
        output_kl_per_channel_mean = (output_kl.mean(0) / target_size).detach().clone().float()

        if self.running_hidden_kls_per_channel is None:
            self.running_hidden_kls_per_channel = hidden_kl_per_channel_mean
            self.running_output_kls_per_channel = output_kl_per_channel_mean
        else:
            self.running_hidden_kls_per_channel = self.running_hidden_kls_per_channel * self.running_beta + hidden_kl_per_channel_mean * (1 - self.running_beta)
            self.running_output_kls_per_channel = self.running_output_kls_per_channel * self.running_beta + output_kl_per_channel_mean * (1 - self.running_beta)

        # balance the hidden vs output kls
        denom = (
            self.running_hidden_kls_per_channel.mean() +
            self.running_output_kls_per_channel.mean() +
            1e-7
        )
        w_hidden_kl = self.running_hidden_kls_per_channel.mean() / denom
        w_output_kl = self.running_output_kls_per_channel.mean() / denom

        # make sure the total weights are unchanged
        og_total = encoder_mus[0].numel() + target_size
        w_total = (
            w_hidden_kl * encoder_mus[0].numel() +
            w_output_kl * target_size +
            1e-7
        )
        results.w_hidden_kl = w_hidden_kl * og_total / w_total
        results.w_output_kl = w_output_kl * og_total / w_total

        # balance the hidden kl weights
        sequence_kl_weights = self.running_hidden_kls_per_channel / (self.running_hidden_kls_per_channel.mean() + 1e-7)
        sequence_kl_per_token = (hidden_kl.mean(0) * sequence_kl_weights).sum() / model.output_length

        # get the weighted kls
        results.weighted_hidden_kl_per_token = sequence_kl_per_token * w_hidden_kl
        results.weighted_output_kl_per_token = results.output_kl_per_token * w_output_kl

        results.loss = (
            results.weighted_hidden_kl_per_token +
            results.weighted_output_kl_per_token
        )

        if step % self.log_image_interval == 0:

            results.kl_weights = Image(
                self.running_hidden_kls_per_channel.cpu().numpy().reshape(model.z_length, model.num_latent_layers).T / self.running_hidden_kls_per_channel.max().item(),
                mode='L'
            )

            results.mean_kl_weights = Image(
                mean_hidden_kl.mean(0).detach().cpu().numpy().reshape(model.z_length, model.num_latent_layers).T / mean_hidden_kl.mean(0).detach().max().item(),
                mode='L'
            )

        return results
    