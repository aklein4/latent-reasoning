import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer
from utils.dot_dict import DotDict


class ZLmTrainer(BaseTrainer):

    def train_step(self, step, model, input_ids, output_ids):

        output = model(input_ids, output_ids)

        logp = torch.take_along_dim(
            output.lm_logits,
            output_ids[..., None],
            dim=-1,
        )[..., 0]

        results = DotDict(
            lm_loss = logp.mean(),
        )
        
        # [..., seq]
        kl = ((output.encoder_mus - output.decoder_mus).pow(2) / 2).sum(-1)
        # [seq]
        seq_kl = kl.reshape(-1, kl.shape[-1]).mean(0)
        results.seq_kl = seq_kl.detach().cpu().numpy()

        # [seq]
        fixed_kl = seq_kl.pow(2) / (seq_kl.sum() + 1e-7)
        results.kl_loss = fixed_kl.sum() / model.output_length

        results.loss = results.lm_loss + self.kl_weight * results.kl_loss

        return results
    