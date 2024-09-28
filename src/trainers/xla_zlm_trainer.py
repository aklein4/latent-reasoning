import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from trainers.base_xla_trainer import BaseXLATrainer
from utils.data_utils import DotDict

from utils.training_utils import (
    loss, ppl, acc, pcorr
)

class XLAZLmTrainer(BaseXLATrainer):

    def train_step(self, step, model, x):

        logits, kl = model(x)
        
        # shift kl
        kl = kl[:, :-1].mean()

        # current version is raw logit clipping
        logit_clip = kl > self.kl_max
        kl_clip = kl < self.kl_min

        results = DotDict(
            nlogp=loss(logits, x),
            ppl=ppl(logits, x),
            acc=acc(logits, x),
            pcorr=pcorr(logits, x),
            kl=kl,
            logit_clip=logit_clip.float(),
            kl_clip=kl_clip.float()
        )

        results.elbo = results.nlogp + kl
        results.loss = (
            torch.where(logit_clip, results.nlogp.detach(), results.nlogp) +
            torch.where(kl_clip, kl.detach(), kl)
        )

        return results
