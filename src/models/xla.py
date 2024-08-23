import torch

from transformers.modeling_utils import PretrainedConfig, PreTrainedModel

from utils.logging_utils import log_print


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


    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={}):
        raise NotImplementedError("gradient_checkpointing_enable not implemented for this model!")
        log_print(f"Gradient checkpointing enabled for {self.__class__.__name__}!")


    def __init__(self, *args, fast_start=False, **kwargs):
        super().__init__(*args, **kwargs)

        self._fast_start = fast_start


    def init_weights(self):
        if self._fast_start:
            return

        super().init_weights()
