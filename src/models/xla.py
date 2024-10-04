import torch

from transformers.modeling_utils import PretrainedConfig, PreTrainedModel

from utils.logging_utils import log_print
from utils.model_utils import apply_fsdp
import utils.constants as constants


class XLAConfig(PretrainedConfig):
 
    model_type = 'xla'


    def __init__(
        self,
        vocab_size=None,
        max_sequence_length=None,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
        gradient_checkpointing=False,
        reshard_after_forward=True,
        *args,
        **kwargs,
    ):
        """
        Things initialized here are needed for the rest of the training pipeline.

        Args:
            vocab_size (`int`):
                Vocabulary size of the model. Defines the number of different tokens that
                can be represented by the `inputs_ids`.
            max_sequence_length (`int`):
                The maximum sequence length that this model might ever be used with.
            bos_token_id (int, *optional*, defaults to 0):
                The id of the `BOS` token in the vocabulary.
            eos_token_id (int, *optional*, defaults to 0):
                The id of the `EOS` token in the vocabulary.
            pad_token_id (int, *optional*, defaults to 0):
                The id of the `PAD` token in the vocabulary.
            gradient_checkpointing (bool, *optional*, defaults to False):
                Whether to use gradient checkpointing to save memory at the cost of extra computation.
        """

        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length

        self.reshard_after_forward = reshard_after_forward

        # requires workaround
        tmp_gradient_checkpointing = gradient_checkpointing

        # init with work arounds
        super().__init__(
            *args, 
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        self.gradient_checkpointing = tmp_gradient_checkpointing


class XLAModel(PreTrainedModel):

    config_class = XLAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True


    # converted from torch to torch xla
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={}):
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")
        
        if constants.XLA_AVAILABLE:
            self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=None)
            
            log_print(f"Gradient checkpointing enabled for {self.__class__.__name__}")


    def __init__(self, *args, fast_start=False, **kwargs):
        super().__init__(*args, **kwargs)

        self._fast_start = fast_start


    def init_weights(self):
        if self._fast_start:
            return

        super().init_weights()

    
    def init_fsdp(self):
        return apply_fsdp(self, False)
