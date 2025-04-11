import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import wandb

from trainers.base_trainer import BaseTrainer
from utils.model_utils import scale_gradient
from utils.dot_dict import DotDict


class UncondZLmTrainer(BaseTrainer):
    
    def train_step(self, step, model, input_ids, output_ids):

        # get model predictions
        output = model(input_ids, output_ids)

        # extract probabilities
        logp = torch.take_along_dim(
            output.lm_logits,
            output_ids[..., None],
            dim=-1,
        )[..., 0]

        kl = (output.encoder_mus - output.generator_mus).pow(2).sum(-1) / 2

        mean_mus = output.encoder_mus.mean(0, keepdim=True)
        mean_kl = (
            output.encoder_mus - mean_mus
        ).pow(2).sum(-1).detach() / 2

        # calculate lm metrics
        results = DotDict(
            lm_loss = -logp.mean(),
            lm_pcorr = logp.exp().mean(),
            lm_acc = (output.lm_logits.argmax(-1) == output_ids).float().mean(),
        )
        
        # calculate kl metrics
        results.kl_per_channel = kl.mean() / model.latent_size_per_layer
        results.kl_per_token = kl.mean() * (model.z_length * model.num_latent_layers) / model.output_length
        results.kl_per_token_mean = mean_kl.mean() * (model.z_length * model.num_latent_layers) / model.output_length
        results.elbo = results.lm_loss + results.kl_per_token

        # calculate weighted lm loss
        results.lm_scale = 1e-7 + 1 - torch.clip(
            (results.lm_acc - self.lower_acc_bound) / (self.upper_acc_bound - self.lower_acc_bound),
            min=0.0,
            max=1.0,
        ).detach()
        results.lm_loss_scaled = -(logp * results.lm_scale).mean()

        # get weighted kl
        results.kl_scale = 1e-7 + torch.clip(
            (results.kl_per_token - self.lower_kl_bound) / (self.upper_kl_bound - self.lower_kl_bound),
            min=0.0,
            max=1.0,
        ).detach().item()

        kl = (
            scale_gradient(output.encoder_mus, results.kl_scale) - output.generator_mus
        ).pow(2).sum(-1) / 2

        seq_kl = kl.reshape(-1, kl.shape[-2] * kl.shape[-1]).mean(0)
        weighted_kl = seq_kl * (seq_kl / (seq_kl.mean() + 1e-7)).detach()
        results.kl_per_token_weighted = weighted_kl.sum() / model.output_length
        results.kl_per_token_weighted_scaled = self.kl_scale * results.kl_per_token_weighted

        results.loss = (
            results.lm_loss_scaled +
            results.kl_per_token_weighted_scaled
        )

        return results
    