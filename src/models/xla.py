import torch

from transformers.modeling_utils import PretrainedConfig, PreTrainedModel

from utils.logging_utils import log_print
from utils.model_utils import checkpoint_barrier
import utils.constants as constants


class XLAConfig(PretrainedConfig):
 
    model_type = 'xla'

    def __init__(
        self,
        gradient_checkpointing=False,
        *args,
        **kwargs,
    ):
        # requires workaround
        tmp_gradient_checkpointing = gradient_checkpointing

        # init with work arounds
        super().__init__(*args, **kwargs)
        self.gradient_checkpointing = tmp_gradient_checkpointing


class XLAModel(PreTrainedModel):

    config_class = XLAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    requires_barrier = False


    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={}):
        log_print(f"Gradient checkpointing enabled for {self.__class__.__name__}!")


    def __init__(self, *args, fast_start=False, **kwargs):
        super().__init__(*args, **kwargs)

        self._fast_start = fast_start
        self.requires_barrier = self.requires_barrier or self.config.gradient_checkpointing


    def init_weights(self):
        if self._fast_start:
            return

        super().init_weights()
        self.special_init_weights()
        self.post_step()


    def special_init_weights(self):
        pass

    def post_step(self):
        pass


    def post_forward(self, outputs):
        if self.requires_barrier and self.training:
            checkpoint_barrier(outputs)

        return outputs
