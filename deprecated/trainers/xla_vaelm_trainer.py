import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from trainers.base_xla_trainer import BaseXLATrainer
from utils.data_utils import DotDict
from utils.training_utils import (
    log_prob, loss, acc, pcorr
)


class XLAVaeLmTrainer(BaseXLATrainer):

    def train_step(self, step, model, x):

        logits, enc_mu, enc_sigma, dec_mu, dec_sigma = model(x)
        
        nlogp = log_prob(logits, x, shift=False, ignore_index=model.pad_token_id)

        # kl divergence per token
        kl = (
            torch.log(dec_sigma) - torch.log(enc_sigma)
            + (enc_sigma**2 + (enc_mu - dec_mu)**2) / (2 * (dec_sigma**2))
            - 0.5
        ).sum(-1).sum(-1).sum(-1)

        # clip kl
        kl_clipped = torch.clamp(
            kl,
            min=(nlogp * self.kl_frac).detach()
        )

        results = DotDict(
            token_loss=loss(logits, x, shift=False, ignore_index=model.pad_token_id),
            kl=kl.sum() / (x != model.pad_token_id).float().sum(),
            kl_clipped=kl_clipped / (x != model.pad_token_id).float().sum(),
            clip_percent=(kl < (nlogp * self.kl_frac).detach()).float().mean(),
            acc=acc(logits, x, shift=False, ignore_index=model.pad_token_id),
            pcorr=pcorr(logits, x, shift=False, ignore_index=model.pad_token_id),
            nlogp=nlogp.mean(),
            kl_total=kl.mean(),
            kl_clipped_total=kl_clipped.mean(),
        )
        results.nelbo = results.token_loss + results.kl

        results.loss = (nlogp.mean() + kl_clipped.mean()) / x.shape[1]

        return results
