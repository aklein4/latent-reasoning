import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from wandb import Histogram

from trainers.base_trainer import BaseTrainer
from utils.model_utils import scale_gradient
from utils.dot_dict import DotDict


class ZAETrainer(BaseTrainer):
    
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

        kl = (
            output.encoder_mus - output.generator_mus
        ).pow(2).sum(-1) / 2

        mean_kl = (
            output.encoder_mus - 0.0
        ).pow(2).sum(-1).detach() / 2

        # calculate lm metrics
        results = DotDict(
            lm_loss = -logp.mean(),
            lm_pcorr = logp.exp().mean(),
            lm_acc = (output.lm_logits.argmax(-1) == output_ids).float().mean(),
        )
        
        # calculate kl metrics
        results.kl_per_channel = kl.mean() / model.latent_size
        results.kl_per_token = kl.mean() * model.latent_length / model.output_length
        results.kl_per_token_mean = mean_kl.mean() * model.latent_length / model.output_length
        results.elbo = results.lm_loss + results.kl_per_token

        # calculate weighted lm loss
        lm_mask = 1e-7 + 1 - torch.clip(
            (logp - np.log(self.lower_p_bound)) / (np.log(self.upper_p_bound) - np.log(self.lower_p_bound)),
            min=0.0,
            max=1.0,
        ).detach()

        results.lm_mask = lm_mask.mean()
        results.lm_loss_masked = -(lm_mask * logp).mean()

        if not self.hooked:
            if results.lm_acc >= self.hook_acc:
                self.hooked = True
                results.reset_optimizer = 1.0

        results.loss = results.lm_loss_weighted
        if self.hooked:
            results.loss = results.loss + self.kl_weight * (
                results.kl_per_token +
                self.mean_weight * results.kl_per_token_mean
            )

        return results
    