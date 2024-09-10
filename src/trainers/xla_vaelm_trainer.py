import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from trainers.base_xla_trainer import BaseXLATrainer
from utils.data_utils import DotDict
from utils.training_utils import (
    loss, acc, pcorr
)

class XLAVaeLmTrainer(BaseXLATrainer):

    def train_step(self, step, model, x):

        logits, enc_mu, enc_sigma, dec_mu, dec_sigma = model(x)
        
        # kl divergence per token
        kl = (
            torch.log(dec_sigma) - torch.log(enc_sigma)
            + (enc_sigma**2 + (enc_mu - dec_mu)**2) / (2 * (dec_sigma**2))
            - 0.5
        ).sum(-1).sum(-1).sum(-1) / x.shape[1]

        # get beta based on the global stap
        beta = self.kl_beta_start + min(step/self.kl_beta_steps, 1) * (1 - self.kl_beta_start)

        results = DotDict(
            token_loss=loss(logits, x),
            kl=kl.mean(),
            acc=acc(logits, x),
            pcorr=pcorr(logits, x),
        )
        results.nelbo = results.token_loss + results.kl
        results.loss = results.token_loss + beta * results.kl
        results.beta = beta * torch.ones_like(results.kl)

        return results
