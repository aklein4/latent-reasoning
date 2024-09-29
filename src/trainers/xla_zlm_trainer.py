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
        kl_min = self.anneal_max * max(0, 1 - step/self.anneal_steps)

        logits, kl = model(x)
        
        # shift kl
        kl = kl[:, :-1].mean()
        kl_clip = kl < kl_min

        results = DotDict(
            nlogp=loss(logits, x),
            acc=acc(logits, x),
            pcorr=pcorr(logits, x),
            kl=kl,
            kl_min=torch.full_like(kl, kl_min),
            kl_clip=kl_clip.float()
        )

        results.nelbo = results.nlogp + kl
        results.loss = (
            results.nlogp +
            torch.where(kl_clip, kl.detach(), kl)
        )

        return results
