import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer
from utils.dot_dict import DotDict


class ZLmTrainer(BaseTrainer):

    info_hits = 1


    def train_step(self, step, model, input_ids, output_ids):

        output = model(input_ids, output_ids)

        logp = torch.take_along_dim(
            output.lm_logits,
            output_ids[..., None],
            dim=-1,
        )[..., 0]

        results = DotDict(
            lm_loss = -logp.mean(),
        )
        
        self.info_hits += (results.lm_loss <= self.info_threshold).long().detach().item()
        results.info_weight = min(1.0, self.info_hits / self.info_total)        

        # [..., seq]
        kl = ((output.encoder_mus - output.decoder_mus).pow(2) / 2).sum(-1)
        results.kl_raw = kl.mean()
        results.kl_raw_per_token = results.kl_raw * model.num_z / model.output_length

        # [seq]
        seq_kl = kl.reshape(-1, kl.shape[-1]).mean(0)

        # [seq]
        fixed_kl = seq_kl * (seq_kl / (seq_kl.mean() + 1e-7))
        results.kl_loss = fixed_kl.sum() / model.output_length

        results.loss = (
            results.lm_loss +
            results.info_weight * self.kl_weight * results.kl_loss
        )

        return results
    