import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from trainers.base_xla_trainer import BaseXLATrainer
from utils.data_utils import DotDict


class XLASwiftTrainer(BaseXLATrainer):


    def token_loss(self, log_probs, clip_mask):
        # each token gets the same weight
        clipped_log_probs = torch.where(
            clip_mask,
            log_probs,
            log_probs.detach()
        )
        return -clipped_log_probs.mean()

    def kl_loss(self, kl, mask):
        # return kl with every token weighted equally
        return kl.mean() / mask.shape[1]

    def loss(self, token_loss, kl_loss, curr_token_w):
        return curr_token_w * token_loss + self.kl_w * kl_loss
    

    def clip_perc(self, mask, clip_mask):
        return 1 - (clip_mask.float().sum() / mask.float().sum())

    def acc(self, logits, x, mask):
        correct = (logits.argmax(-1) == x).float()
        correct = torch.where(mask, correct, torch.zeros_like(correct))
        return correct.sum() / mask.float().sum()


    def logp_per_token(self, log_probs, mask):
        return -log_probs.sum() / mask.float().sum()

    def logp_per_token_nopad(self, log_probs, mask, x, pad):
        log_probs = torch.where(mask & (x != pad), log_probs, torch.zeros_like(log_probs))
        return -log_probs.sum() / (mask & (x != pad)).float().sum()


    def kl_per_token(self, kl, mask):
        return kl.sum() / mask.float().sum()

    def kl_per_token_nopad(self, kl, mask, x, pad):
        return kl.sum() / (mask & (x != pad)).float().sum()


    def train_step(self, step, model, x, mask):
        curr_token_w = self.token_w * min(1, step / self.token_w_steps)

        logits, kl = model(x, mask)
        
        # log probs, with zero for unmasked tokens
        ar = torch.arange(x.numel(), device=x.device, dtype=x.dtype)
        log_probs = logits.view(-1, logits.shape[-1])[ar, x.view(-1)].view(*x.shape)
        log_probs = torch.where(mask, log_probs, torch.zeros_like(log_probs))

        # current version is raw logit clipping
        clip_mask = mask & (log_probs < np.log(self.clip_prob))

        results = DotDict(
            token_loss=self.token_loss(log_probs, clip_mask),
            kl_loss=self.kl_loss(kl, mask),
            clip_perc=self.clip_perc(mask, clip_mask),
            acc=self.acc(logits, x, mask),
            logp_per_token=self.logp_per_token(log_probs, mask),
            logp_per_token_nopad=self.logp_per_token_nopad(log_probs, mask, x, model.config.pad_token_id),
            kl_per_token=self.kl_per_token(kl, mask),
            kl_per_token_nopad=self.kl_per_token_nopad(kl, mask, x, model.config.pad_token_id),
        )
        results.loss = self.loss(results.token_loss, results.kl_loss, curr_token_w)

        results.curr_token_w = torch.full_like(results.loss, curr_token_w)

        results.one_minus_acc = 1 - results.acc
        results.one_minus_clip_perc = 1 - results.clip_perc

        return results
