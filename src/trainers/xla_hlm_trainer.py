import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from trainers.base_xla_trainer import BaseXLATrainer
from utils.data_utils import DotDict


class XLAHLmTrainer(BaseXLATrainer):

    def token_loss(self, log_probs, clip_mask):
        # each token gets the same weight, clip with mask
        clipped_log_probs = torch.where(
            clip_mask,
            log_probs,
            log_probs.detach()
        )
        return -clipped_log_probs.mean()

    def kl_loss(self, kl, mask):
        # return kl with every token weighted equally
        return kl.mean() / mask.shape[1]

    def loss(self, token_loss, kl_loss, uncond_kl_loss, kl_clip, acc_clip):
        # if either global clip triggers, only use kl
        return (
            self.kl_w * kl_loss +
            self.uncond_kl_w * uncond_kl_loss +
            self.token_w * torch.where(
                kl_clip | acc_clip,
                token_loss.detach(),
                token_loss,
            )
        )
    

    def acc(self, logits, x, mask):
        correct = (logits.argmax(-1) == x).float()
        correct = torch.where(mask, correct, torch.zeros_like(correct))
        return correct.sum() / mask.float().sum()

    def clip_perc(self, mask, clip_mask):
        return 1 - (clip_mask.float().sum() / mask.float().sum())


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

        logits, kl, uncond_kl = model(x, mask)
        
        # log probs, with zero for unmasked tokens
        ar = torch.arange(x.numel(), device=x.device, dtype=x.dtype)
        log_probs = logits.view(-1, logits.shape[-1])[ar, x.view(-1)].view(*x.shape)
        log_probs = torch.where(mask, log_probs, torch.zeros_like(log_probs))

        # current version is raw logit clipping
        kl_clip = self.kl_per_token(kl, mask) > self.kl_threshold
        acc_clip = self.acc(logits, x, mask) > self.acc_threshold
        clip_mask = mask & (log_probs < np.log(self.clip_prob))  # (torch.argmax(logits, dim=-1) != x)

        results = DotDict(
            token_loss=self.token_loss(log_probs, clip_mask),
            kl_loss=self.kl_loss(kl, mask),
            uncond_kl_loss=self.kl_loss(uncond_kl, mask),
            acc=self.acc(logits, x, mask),
            clip_perc=self.clip_perc(mask, clip_mask),
            kl_clip=kl_clip.float(),
            acc_clip=acc_clip.float(),
            logp_per_token=self.logp_per_token(log_probs, mask),
            logp_per_token_nopad=self.logp_per_token_nopad(log_probs, mask, x, model.config.pad_token_id),
            kl_per_token=self.kl_per_token(kl, mask),
            kl_per_token_nopad=self.kl_per_token_nopad(kl, mask, x, model.config.pad_token_id),
            uncond_kl_per_token=self.kl_per_token(uncond_kl, mask),
            uncond_kl_per_token_nopad=self.kl_per_token_nopad(uncond_kl, mask, x, model.config.pad_token_id),
        )
        results.loss = self.loss(results.token_loss, results.kl_loss, results.uncond_kl_loss, kl_clip, acc_clip)

        results.one_minus_acc = 1 - results.acc
        results.one_minus_clip_perc = 1 - results.clip_perc

        return results
