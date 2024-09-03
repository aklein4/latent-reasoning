import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from trainers.base_xla_trainer import BaseXLATrainer
from utils.data_utils import DotDict


class XLASwiftTrainer(BaseXLATrainer):


    def token_loss(self, log_probs, mask):
        clipped_log_probs = torch.where(
            mask,
            torch.clamp(log_probs, max=np.log(self.clip_prob)),
            log_probs
        )
        return -clipped_log_probs.sum(-1).mean() / mask.shape[1]

    def kl_loss(self, kl, mask):
        return kl.mean() / mask.shape[1]

    def loss(self, token_loss, kl_loss):
        return self.token_w * token_loss + self.kl_w * kl_loss
    

    def clip_perc(self, log_probs, mask):
        clipped = (log_probs > np.log(self.clip_prob)).float()
        clipped = torch.where(mask, clipped, torch.zeros_like(clipped))
        return clipped.sum() / mask.float().sum()

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


    def train_step(self, model, x, mask):

        logits, mu, sigma = model(x, mask)
        
        # log probs, with zero for unmasked tokens
        ar = torch.arange(x.numel(), device=x.device, dtype=x.dtype)
        log_probs = logits.view(-1, logits.shape[-1])[ar, x.view(-1)].view(*x.shape)
        log_probs = torch.where(mask, log_probs, torch.zeros_like(log_probs))

        # kl divergence [bs,]
        kl = (-torch.log(sigma) + 0.5 * (mu**2 + sigma**2) - 0.5).sum(0).sum(-1).sum(1)

        results = DotDict(
            token_loss=self.token_loss(log_probs, mask),
            kl_loss=self.kl_loss(kl, mask),
            clip_perc=self.clip_perc(log_probs, mask),
            acc=self.acc(logits, x, mask),
            logp_per_token=self.logp_per_token(log_probs, mask),
            logp_per_token_nopad=self.logp_per_token_nopad(log_probs, mask, x, model.config.pad_token_id),
            kl_per_token=self.kl_per_token(kl, mask),
            kl_per_token_nopad=self.kl_per_token_nopad(kl, mask, x, model.config.pad_token_id),
        )
        results.loss = self.loss(results.token_loss, results.kl_loss)

        return results
