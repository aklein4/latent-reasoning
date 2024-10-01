import torch

from transformers import AutoTokenizer

from models.patch_hlm import PatchHLmConfig, PatchHLmModel
from utils.config_utils import load_model_config
import utils.constants as constants


MODEL_CONFIG = 'test-patch_hlm'


def main():

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(constants.GPT2_TOKENIZER, resume_download=None)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    assert len(tokenizer) == constants.GPT2_VOCAB_SIZE
    assert tokenizer.pad_token_id == constants.GPT2_PAD_TOKEN
    assert tokenizer.bos_token_id == constants.GPT2_BOS_TOKEN
    assert tokenizer.eos_token_id == constants.GPT2_EOS_TOKEN

    x = tokenizer(["Hello, my dog is cute", "His dog is cute too", "All dogs are cute"], return_tensors="pt", padding="max_length", max_length=16).input_ids
    mask = torch.ones_like(x).bool() # torch.randint_like(x, 2).bool()
    mask[0, :3] = False
    mask[1, :8] = False

    print("loading model...")
    config = load_model_config(MODEL_CONFIG)
    model = PatchHLmModel(PatchHLmConfig(**config), fast_start=True)

    # for name, module in model.enc_gen.layers[0].named_modules():
    #     print(f"{name}: {sum([p.numel() for p in module.parameters()]):_}")
    # exit()

    logits, kl, uncond_kl = model(x, mask, num_uncond=1)

    print(logits.shape)
    print(uncond_kl)
    print(kl)


if __name__ == '__main__':

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()
