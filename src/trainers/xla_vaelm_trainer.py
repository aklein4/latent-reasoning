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

        logits, enc_mu, enc_sigma, dec_mu, dec_sigma = model(x, reparam_scale=self.reparam_scale)
        
        # kl divergence per token
        kl = (
            torch.log(dec_sigma) - torch.log(enc_sigma)
            + (enc_sigma**2 + (enc_mu - dec_mu)**2) / (2 * (dec_sigma**2))
            - 0.5
        ).sum(-1).sum(-1).sum(-1) / x.shape[1]

        # clip kl
        kl_clipped = torch.clamp(kl, min=self.kl_clip)

        results = DotDict(
            token_loss=loss(logits, x, shift=False),
            kl=kl.mean(),
            kl_clipped=kl_clipped.mean(),
            clip_percent=(kl < self.kl_clip).float().mean(),
            acc=acc(logits, x, shift=False),
            pcorr=pcorr(logits, x, shift=False),
        )
        results.nelbo = results.token_loss + results.kl

        results.loss_unscaled = results.token_loss + results.kl_clipped
        results.loss = 2 * results.loss_unscaled / (1 + self.reparam_scale)

        return results
