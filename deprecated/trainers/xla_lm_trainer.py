import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_xla_trainer import BaseXLATrainer
from utils.data_utils import DotDict
from  utils.training_utils import loss, ppl, acc, pcorr


class XLALMTrainer(BaseXLATrainer):

    def train_step(self, model, x, _unused):

        out = model(x)
        ignore_index = model.config.pad_token_id

        results = DotDict(
            lm_loss=loss(out, x, ignore_index),
            lm_ppl=ppl(out, x, ignore_index),
            lm_acc=acc(out, x, ignore_index),
            lm_pcorr=pcorr(out, x, ignore_index),
        )
        results.loss = results.lm_loss

        return results
