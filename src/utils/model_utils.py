
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers.activations import ACT2FN


class GaussianMixtureModule(nn.Module):

    def __init__(self, input_dim, output_dim, n_components):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_components = n_components

        self.proj = nn.Linear(input_dim, self.n_components, bias=False)

        self.mu = nn.Parameter(torch.randn(self.n_components, self.output_dim))
        self.log_sigma = nn.Parameter(torch.zeros(self.n_components, self.output_dim))


    def forward(self, x):
        logpi = torch.log_softmax(self.proj(x), dim=-1)

        return GaussianMixtureDistribution(
            logpi,
            self.mu,
            F.softplus(self.log_sigma)
        )


class GaussianMixtureDistribution:

    def __init__(self, logpi, mu, sigma):

        self.shape = logpi.shape[:-1]
        self.k = logpi.shape[-1]
        self.d = mu.shape[-1]

        self.logpi = logpi.view(-1, self.k) # [R, K]
        self.mu = mu # [K, D]
        self.sigma = sigma # [K, D]

    
    def sample(self, n_samples):

        # sample from categorical distribution [n, R]
        z = torch.distributions.Categorical(logits=self.logpi).sample((n_samples,))
        z = z.view(-1)

        # sample from gaussian distribution
        mu = self.mu[z].view(n_samples, *self.shape, -1)
        sigma = self.sigma[z].view(n_samples, *self.shape, -1)
        return mu + sigma * torch.randn_like(mu)
    

    def log_prob(self, x):

        n = x.shape[0]
        x = x.view(n, -1, 1, self.d) # [n, R, 1, D]

        mu_n = self.mu[None, None] # [1, 1, K, D]
        sigma_n = self.sigma[None, None] # [1, 1, K, D]
        logpi_n = self.logpi[None] # [1, R, K]

        log_probs = -0.5 * (
            2 * torch.log(sigma_n) +
            np.log(2 * np.pi) +
            ((x - mu_n) / sigma_n) ** 2
        ) # [n, R, K, D]

        log_probs = torch.sum(log_probs, dim=-1) # [n, R, K]
        log_probs = torch.logsumexp(logpi_n + log_probs, dim=-1)

        return log_probs.view(n, *self.shape)



class RotaryAttention(nn.Module):

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        use_rope,
        rope_fraction,
        max_sequence_length,
        rope_base,
        layer_idx):
        super().__init__()

        self.layer_idx = layer_idx

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.use_rope = use_rope
        if self.use_rope:
            self.rope = RotaryEmbedding(
                self.head_dim, rope_fraction,
                max_sequence_length,
                rope_base
            )
        else:
            self.rope = None


    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask=None,
        past_key_value=None,
    ):

        # get shapes
        bsz, q_len, _ = query_states.shape

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # apply rope
        if self.use_rope:
            query_states, key_states = self.rope(query_states, key_states, position_ids)

        # update/apply cache
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3) / np.sqrt(self.head_dim))
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        return attn_output


class RotaryEmbedding(nn.Module):

    def __init__(self, total_dim, frac, max_position_embeddings, base):
        super().__init__()

        assert total_dim % frac == 0, f'dimension {total_dim} must be divisible by frac {frac}'
        self.total_dim = total_dim
        self.dim = total_dim // frac
        assert self.dim % 2 == 0, f'dimension {self.dim} must be divisible by 2'

        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # inverse frequencies for rotations
        freq_ar = torch.arange(0, self.dim, 2).float()
        inv_freq = (
            1.0 /
            (self.base ** (freq_ar / self.dim))
        ) # [D/2]

        # only use integer positions, so we cache sin/cos as embeddings
        pos = torch.arange(0, self.max_position_embeddings).float()
        freqs = torch.matmul(inv_freq[:, None], pos[None, :]) # [D/2, L]
        freqs = freqs.permute(1, 0) # [L, D/2]

        freqs = torch.cat((freqs, freqs), dim=-1) # [L, D]
        sin = freqs.sin()
        cos = freqs.cos()
        
        self.sin_emb = nn.Embedding(self.max_position_embeddings, self.dim)
        self.sin_emb.weight.data = sin.contiguous()

        self.cos_emb = nn.Embedding(self.max_position_embeddings, self.dim)
        self.cos_emb.weight.data = cos.contiguous()


    def _get_sin_cos(self, position_ids):
        return (
            self.sin_emb(position_ids).detach(),
            self.cos_emb(position_ids).detach()
        )


    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


    def forward(self, q, k, position_ids):
        assert q.shape[-1] == self.total_dim, f'q shape {q.shape} does not match total_dim {self.total_dim}'
        assert k.shape[-1] == self.total_dim, f'k shape {k.shape} does not match total_dim {self.total_dim}'

        sin, cos = self._get_sin_cos(position_ids)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        if self.dim == self.total_dim:
            q = (q * cos) + (self._rotate_half(q) * sin)
            k = (k * cos) + (self._rotate_half(k) * sin)
            return q, k

        q_rot, q_no_rot = q[..., : self.dim], q[..., self.dim :]
        k_rot, k_no_rot = k[..., : self.dim], k[..., self.dim :]

        q_rot = (q_rot * cos) + (self._rotate_half(q_rot) * sin)
        k_rot = (k_rot * cos) + (self._rotate_half(k_rot) * sin)

        q = torch.cat((q_rot, q_no_rot), dim=-1)
        k = torch.cat((k_rot, k_no_rot), dim=-1)

        return q, k


class GLU(nn.Module):

    def __init__(self, activation):
        super().__init__()
        self.activation = ACT2FN[activation]
    

    def forward(self, gate, value):
        return self.activation(gate) * value
    