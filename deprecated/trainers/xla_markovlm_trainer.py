import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from trainers.base_xla_trainer import BaseXLATrainer
from utils.data_utils import DotDict

from utils.training_utils import (
    loss, ppl, acc, pcorr
)

class XLAMarkovLmTrainer(BaseXLATrainer):

    def train_step(self, step, model, x):

        logits, kl = model(x)
        
        # shift kl
        kl = kl[:, :-1].mean()

        results = DotDict(
            nlogp=loss(logits, x, clip=self.clip_prob),
            acc=acc(logits, x),
            pcorr=pcorr(logits, x),
            kl=kl,
        )

        results.elbo = results.nlogp + kl
        results.loss = self.token_w * results.nlogp + kl

        return results
