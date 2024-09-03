import torch

from transformers.modeling_utils import PretrainedConfig, PreTrainedModel

from utils.logging_utils import log_print


class XLAConfig(PretrainedConfig):
 
    model_type = 'xla'


    def __init__(
        self,
        vocab_size,
        max_sequence_length,
        bos_token_id,
        eos_token_id,
        pad_token_id,
        gradient_checkpointing,
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
