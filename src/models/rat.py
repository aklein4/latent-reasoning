from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_xla.core.xla_model as xm
except:
    pass

import numpy as np

from models.base import (
    BaseConfig, BaseTransformer, BaseLmModel,
    BaseAttention, BaseMLP
)
from utils.model_utils import _extract_tensors_from_list, fast_checkpoint
import utils.constants as constants


class RatConfig(BaseConfig):

    model_type = 'rat'

    def __init__(
        self,
        residual_channels: int = 4,
        timescale_rank: int = None,
        softplus_init_min: float = 0.01,
        bootstrap_debug: bool = False,
        *args,
        **kwargs,
    ):
        
        self.residual_channels = residual_channels
        self.timescale_rank = timescale_rank
        self.softplus_init_min = softplus_init_min
        self.bootstrap_debug = bootstrap_debug

        super().__init__(*args, **kwargs)


class RatReader(nn.Module):

    def special_init_weights(self, config):
        
        # init such that r~1 after softplus
        initer = torch.randn_like(self.q.data) / np.sqrt(config.residual_channels)
        initer = initer.abs() + config.softplus_init_min
        initer = torch.log(initer.exp() - 1)
        
        self.q.data[:] = initer.detach()


    def __init__(self, config, num_inputs):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.residual_channels = config.residual_channels
        self.num_inputs = num_inputs

        self.residual_size = self.hidden_size * self.residual_channels
        self.input_size = self.hidden_size * self.num_inputs

        self.q = nn.Parameter(
            torch.ones(
                1, 1,
                self.hidden_size, self.residual_channels,
                self.num_inputs
            )
        )
        self.norm = nn.GroupNorm(
            self.num_inputs,
            self.input_size,
            eps=config.layer_norm_eps
        )


    def forward(self, x, normalizer):
        bs, l, _, c = x.shape

        q_plus = F.softplus(self.q)

        y = (x.unsqueeze(-1) * q_plus).sum(dim=-2)
        denom = (normalizer.unsqueeze(-1) * q_plus).sum(dim=-2)
        
        y = y / (denom + 1)

        assert y.shape == (bs, l, self.hidden_size, self.num_inputs)
        y = (
            y
            .permute(0, 1, 3, 2)
            .reshape(bs, l, self.input_size)
        )

        y = y.view(bs*l, self.input_size)
        y = self.norm(y)
        y = y.view(bs, l, self.input_size)

        return y


class RatWriter(nn.Module):

    def special_init_weights(self, config):
        
        initer = torch.randn_like(self.k.data) / np.sqrt(config.residual_channels)
        initer = initer.abs() + config.softplus_init_min
        initer = torch.log(initer.exp() - 1)
        
        self.k.data[:] = initer.detach()

        if self.timescale_rank is not None:
            self.t_up.weight.data.zero_()
            self.t_up.bias.data.fill_(np.log(np.exp(1) - 1))


    def __init__(self, config, raw_size):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.residual_channels = config.residual_channels
        self.raw_size = raw_size

        self.residual_size = self.hidden_size * self.residual_channels

        self.k = nn.Parameter(
            torch.ones(
                1, 1,
                self.hidden_size, self.residual_channels,
            )
        )

        self.timescale_rank = config.timescale_rank
        if self.timescale_rank is not None:
            self.t_down = nn.Linear(self.raw_size, self.timescale_rank, bias=False)
            self.t_up = nn.Linear(self.timescale_rank, self.hidden_size, bias=True)

    
    def forward(self, x, raw):
        k_curr = self._get_k(raw)

        return x.unsqueeze(-1) * k_curr, k_curr
    

    def _get_k(self, raw):
        if self.timescale_rank is None:
            return F.softplus(self.k)

        t = self.t_up(self.t_down(raw))
        t_pos = F.softplus(t)

        return t_pos.unsqueeze(-1) * F.softplus(self.k)


class Tracker(nn.Module):

    def __init__(self):
        super().__init__()
        self.tracked = None
        self.normalizer = None
    
    def forward(self, x, normalizer):
        return track_fn.apply(x, normalizer, self)


    def get(self):
        return self.tracked

    def update(self, x):
        self.tracked = x.detach()


    def get_normalizer(self):
        return self.normalizer
    
    def update_normalizer(self, x):
        self.normalizer = x.detach()


    def clear(self):
        self.tracked = None
        self.normalizer = None


class track_fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, normalizer, tracker):
        ctx.tracker = tracker

        tracker.update(x)
        tracker.update_normalizer(normalizer)
        return x, normalizer


    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output, norm_output):

        ctx.tracker.clear()
        return (grad_output, norm_output) + (None,)


class Block(nn.Module):

    def special_init_weights(self, config):
        self.reader.special_init_weights(config)
        self.writer.special_init_weights(config)


    def __init__(self, config, operation, tracker, num_inputs, raw_size):
        super().__init__()

        self.operation = operation
        self.tracker = tracker

        self.reader = RatReader(config, num_inputs)
        self.writer = RatWriter(config, raw_size)

        self.debug = False


    def compute(self, x, kwargs):
        return self.operation(x, **kwargs)


    def forward(self, x, normalizer, kwargs):
        if self.debug:
            return self.debug_forward(x, normalizer, kwargs)
        return bootstrap_fn.apply(x, normalizer, self, self.tracker, kwargs) # torch.is_grad_enabled())


    def debug_forward(self, s, normalizer, kwargs):
        
        x = self.reader(s, normalizer)
        y, raw = self.compute(x, kwargs)
        a, k = self.writer(y, raw)

        s_new = s + a
        new_norm = normalizer + k

        return s_new, new_norm


class bootstrap_fn(torch.autograd.Function):    

    @staticmethod
    def forward(ctx, s, normalizer, block: Block, tracker, kwargs):

        ctx.gpu_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(),
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled()
        }
        ctx.cpu_autocast_kwargs = {
            "enabled": torch.is_autocast_cpu_enabled(),
            "dtype": torch.get_autocast_cpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled()
        }

        # get x (no grad, s not saved)
        x = block.reader(s, normalizer)
        x.requires_grad = True

        # get y, and link to x
        with torch.enable_grad():
            y, raw = block.compute(x, kwargs)

        # get a (no grad, doesn't matter we save y anyway)
        a, k = block.writer(y, raw)

        # add outputs to streams
        s_new = s + a
        new_norm = normalizer + k

        # save things
        ctx.save_for_backward(x, y, raw) # raw is already saved in compute
        ctx.block = block
        ctx.tracker = tracker

        tracker.update(s_new)
        tracker.update_normalizer(new_norm)

        s_new.requires_grad = True
        new_norm.requires_grad = True
        return s_new, new_norm

    
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output, norm_output):
        x, y, raw = ctx.saved_tensors

        if constants.XLA_AVAILABLE:
            xm.optimization_barrier_(
                _extract_tensors_from_list(
                    [x, y, grad_output, norm_output] +
                    list(ctx.block.parameters()) +
                    list(ctx.block.buffers()) +
                    [ctx.tracker.get(), ctx.tracker.get_normalizer()]
                )
            )

        # get a, populate x with gradient
        with torch.enable_grad(), \
            torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
            torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
                a, k = ctx.block.writer(y, raw)
        torch.autograd.backward((a, k), (grad_output, norm_output))

        # reconstruct orignal input
        s = (ctx.tracker.get() - a).detach()
        normalizer = (ctx.tracker.get_normalizer() - k).detach()

        # reconstruct x and populate s with gradient
        with torch.enable_grad(), \
            torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
            torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
                s.requires_grad = True
                normalizer.requires_grad = True
                new_x = ctx.block.reader(s, normalizer)
        torch.autograd.backward(new_x, x.grad)

        # save s for previous block
        ctx.tracker.update(s)
        ctx.tracker.update_normalizer(normalizer)

        # add grad to residual
        grad_output = grad_output + s.grad
        norm_output = norm_output + normalizer.grad

        return (grad_output, norm_output) + (None,)*3


class RatAttention(BaseAttention):

    def init_qkv_proj(self, config):
        self.qkv_proj = nn.Conv1d(
            3 * config.hidden_size, 3 * config.hidden_size,
            kernel_size=1, bias=config.use_qkv_bias,
            groups=3
        )
    

    def get_qkv(self, hidden_states):
        bs, l, _ = hidden_states.shape

        hidden_states = hidden_states.view(bs*l, 3 * self.hidden_size, 1)
        hidden_states = self.qkv_proj(hidden_states)
        hidden_states = hidden_states.view(bs, l, 3 * self.hidden_size)

        return hidden_states.chunk(3, dim=-1)


    def get_o(self, hidden_states):
        return super().get_o(hidden_states), hidden_states


class RatMLP(BaseMLP):

    def init_mlp_input(self, config):
        self.in_proj = nn.Conv1d(
            2*config.hidden_size, 2*config.mlp_size,
            kernel_size=1, bias=False,
            groups=2
        )

    
    def get_mlp_input(self, hidden_states):
        bs, l, _ = hidden_states.shape

        hidden_states = hidden_states.view(bs*l, 2 * self.hidden_size, 1)
        hidden_states = self.in_proj(hidden_states)
        hidden_states = hidden_states.view(bs, l, 2 * self.mlp_size)

        return hidden_states.chunk(2, dim=-1)


    def get_mlp_output(self, hidden_states):
        return super().get_mlp_output(hidden_states), hidden_states


class RatLayer(nn.Module):

    def special_init_weights(self, config: BaseConfig):
        if config.identity_init:
            raise ValueError("identity_init not supported for RatLayer!")

        self.attn_block.special_init_weights(config)
        self.mlp_block.special_init_weights(config)


    def post_step(self):
        pass


    def enable_debug(self):
        self.attn_block.debug = True
        self.mlp_block.debug = True


    def __init__(self, config: BaseConfig, layer_idx: int, tracker):
        super().__init__()

        self.hidden_size = config.hidden_size

        attn = RatAttention(config, layer_idx)
        mlp = RatMLP(config)

        self.attn_block = Block(config, attn, tracker, 3, config.hidden_size)
        self.mlp_block = Block(config, mlp, tracker, 2, config.mlp_size)


    def forward(
        self,
        hidden_states: torch.Tensor,
        normalizer: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value=None,
    ):

        # Self Attention
        hidden_states, normalizer = self.attn_block(
            hidden_states,
            normalizer,
            {
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'past_key_value': past_key_value
            }
        )

        # GLU MLP
        hidden_states, normalizer = self.mlp_block(
            hidden_states, 
            normalizer,
            {}
        )

        return hidden_states, normalizer


class RatTransformer(BaseTransformer):
    
    tracker = Tracker()
    def layer_type(self, conf, idx):
        return RatLayer(conf, idx, self.tracker)


    def get_extras(self, config):
        self.main_writer = RatWriter(config, config.hidden_size)
        self.main_reader = RatReader(config, 1)


    def special_init_weights(self, config):
        super().special_init_weights(config)

        self.main_writer.special_init_weights(config)
        self.main_reader.special_init_weights(config)


    def enable_debug(self):
        for layer in self.layers:
            layer.enable_debug()


    def __init__(self, config: RatConfig):
        super().__init__(config)

        if self.gradient_checkpointing:
            raise NotImplementedError("Gradient Checkpointing not supported for RatTransformer!")

        self.debug = config.bootstrap_debug
        if self.debug:
            self.enable_debug()


    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ):
        hidden_states = super().get_hidden_states(input_ids, position_ids)
        return fast_checkpoint(
            self.main_writer,
            hidden_states,
            hidden_states
        )


    def get_output(
        self,
        hidden_states: torch.Tensor,
        normalizer: torch.Tensor
    ):
        return fast_checkpoint(
            self.main_reader,
            hidden_states,
            normalizer
        )
    

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv=None
    ):

        # get inputs
        position_ids = self._get_position_ids(input_ids, position_ids)
        attention_mask = self._get_mask(input_ids, segment_ids)
        hidden_states, normalizer = self.get_hidden_states(input_ids, position_ids)

        # start tracking
        self.tracker.clear()
        hidden_states, normalizer = self.tracker(hidden_states, normalizer)

        for layer in self.layers:
            hidden_states, normalizer = layer(
                hidden_states=hidden_states,
                normalizer=normalizer,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_value=kv
            )

        hidden_states = self.get_output(hidden_states, normalizer)

        return hidden_states


class RatLmModel(BaseLmModel):

    transformer_type = RatTransformer
    