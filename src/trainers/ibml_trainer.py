import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from trainers.base_trainer import BaseTrainer
from utils.dot_dict import DotDict


class IBMLTrainer(BaseTrainer):
    
    prev_mats = None
    val_results = None
    seen_tokens = 0

    
    def _lm_metrics(self, logits, x, mask):
        logits = logits[:, :-1]
        x = x[:, 1:]
        mask = mask[:, 1:]

        logp = torch.take_along_dim(
            logits,
            x[..., None],
            dim=-1,
        )[..., 0]

        return DotDict(
            lm_loss = -(logp * mask).sum() / mask.sum(),
            lm_pcorr = (logp.exp() * mask).sum() / mask.sum(),
            lm_acc = ((logits.argmax(-1) == x).float() * mask).sum() / mask.sum(),
        )


    def train_step(self, step, model, input_ids, mask):
        input_ids, val_ids = torch.chunk(input_ids, 2, dim=0)
        mask, val_mask = torch.chunk(mask, 2, dim=0)

        min_res = 1 - self.max_beta
        mat_beta = 1 - max(
            math.exp(step * math.log(min_res) / self.beta_steps),
            min_res,
        )

        # get model predictions
        model_out = model(
            input_ids,
            memory_mask=mask,
            prev_mats=self.prev_mats,
            mat_beta=mat_beta,
        )

        self.prev_mats = model_out.mats.detach()
        self.seen_tokens += mask.sum().item()

        results = self._lm_metrics(
            model_out.lm_logits,
            input_ids,
            mask,
        )
        results.mat_beta = mat_beta
        results.seen_tokens = self.seen_tokens

        if step == 0 or step % self.val_interval == 0:
            with torch.no_grad():

                val_out = model(
                    val_ids,
                    memory_mask=val_mask,
                    prev_mats=self.prev_mats,
                    mat_beta=1.0,
                )

                val_results = self._lm_metrics(
                    val_out.lm_logits,
                    val_ids,
                    val_mask,
                )
                self.val_results = {k: v.detach() for k, v in val_results.items()}

        else:
            val_results = self.val_results

        for k, v in val_results.items():
            results[f"{k}_val"] = v

        results.loss = results.lm_loss
        return results
    