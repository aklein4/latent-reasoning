import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import wandb

from trainers.base_trainer import BaseTrainer
from utils.model_utils import scale_gradient
from utils.dot_dict import DotDict


class UncondZLmTrainer(BaseTrainer):
    
    hooked = False


    def train_step(self, step, model, input_ids, output_ids):

        # get model predictions
        output = model(input_ids, output_ids)

        # extract probabilities
        logp = torch.take_along_dim(
            output.lm_logits,
            output_ids[..., None],
            dim=-1,
        )[..., 0]

        # calculate lm metrics
        results = DotDict(
            lm_loss = -logp.mean(),
            lm_pcorr = logp.exp().mean(),
            lm_acc = (output.lm_logits.argmax(-1) == output_ids).float().mean(),
        )
        
        # handle hook
        if not self.hooked:
            if results.lm_acc.item() >= self.acc_hook:
                self.hooked = True
                results.reset_optimizer = 1.0
        results.hooked = 1.0 if self.hooked else 0.0

        # calculate kl metrics
        kl = (
            (output.encoder_mus if self.hooked else output.encoder_mus.detach()) -
            output.generator_mus
        ).pow(2).sum(-1) / 2

        mean_mus = output.encoder_mus.mean(0, keepdim=True)
        mean_kl = (
            output.encoder_mus - mean_mus
        ).pow(2).sum(-1).detach() / 2

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
        seq_kl = kl.reshape(-1, kl.shape[-2] * kl.shape[-1]).mean(0)
        weighted_kl = seq_kl * (seq_kl / (seq_kl.mean() + 1e-7)).detach()
        
        results.kl_per_token_weighted = weighted_kl.sum() / model.output_length
        results.kl_per_token_weighted_scaled = self.kl_scale * results.kl_per_token_weighted

        results.loss = (
            results.lm_loss_scaled +
            results.kl_per_token_weighted_scaled
        )

        return results
    