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

        logits, markov_logits, kl, markov_kl = model(x)
        
        # shift kl
        kl = kl[:, :-1].mean()
        markov_kl = markov_kl[:, :-1].mean()

        results = DotDict(
            nlogp=loss(logits, x),
            acc=acc(logits, x),
            pcorr=pcorr(logits, x),
            markov_nlogp=loss(markov_logits, x),
            markov_acc=acc(markov_logits, x),
            markov_pcorr=pcorr(markov_logits, x),
            kl=kl,
            markov_kl=markov_kl,
        )

        results.elbo = results.nlogp + kl
        results.markov_elbo = results.markov_nlogp + markov_kl

        results.loss = results.elbo + self.markov_w * results.markov_elbo

        return results
