import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from wandb import Histogram

from trainers.base_trainer import BaseTrainer
from utils.model_utils import scale_gradient
from utils.dot_dict import DotDict


class ZLmTrainer(BaseTrainer):
    
    hooked = False
    warmup_step_progress = 0


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
            scale_gradient(output.encoder_mus, self.encoder_grad_scale) - output.decoder_mus
        ).pow(2).sum(-1) / 2

        mean_mus = output.decoder_mus.mean(0, keepdim=True)
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
        results.kl_per_channel = kl.mean() / model.latent_size
        results.kl_per_token = kl.mean() * model.num_z / model.output_length
        results.kl_per_token_mean = mean_kl.mean() * model.num_z / model.output_length
        results.elbo = results.lm_loss + results.kl_per_token

        # calculate weighted lm loss
        sinker = 1e-7 + 1 - torch.clip(
            (logp - np.log(self.lower_p_bound)) / (np.log(self.upper_p_bound) - np.log(self.lower_p_bound)),
            min=0.0,
            max=1.0,
        ).detach()

        results.lm_mask = sinker.mean()
        results.lm_loss_weighted = -(sinker * logp).mean()

        # get weighted kl
        seq_kl = kl.reshape(-1, kl.shape[-1]).mean(0)
        weighted_kl = seq_kl * (seq_kl / (seq_kl.mean() + 1e-7)).detach()

        results.sequence_kl = Histogram(np_histogram=(
            seq_kl.detach().cpu().numpy(),
            np.arange(seq_kl.shape[0]+1).astype(float),
        ))
        results.kl_per_token_weighted = weighted_kl.sum() / model.output_length

        if not self.hooked:
            if results.lm_acc >= self.hook_acc:
                self.hooked = True
                results.reset_optimizer = 1.0
        else:
            self.warmup_step_progress += 1

        results.kl_weight = min(
            self.warmup_step_progress / self.kl_warmup_steps,
            1.0,
        )

        results.loss = (
            results.lm_loss_weighted +
            (results.kl_weight * results.kl_per_token_weighted if self.hooked else 0.0)
        )

        return results
    