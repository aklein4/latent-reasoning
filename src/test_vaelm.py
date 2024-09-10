import torch

from transformers import AutoTokenizer

from models.vaelm import VaeLmConfig, VaeLmModel
from utils.config_utils import load_model_config
import utils.constants as constants


MODEL_CONFIG = 'test-vaelm'


def main():

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(constants.GPT2_TOKENIZER, resume_download=None)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    assert len(tokenizer) == constants.GPT2_VOCAB_SIZE
    assert tokenizer.pad_token_id == constants.GPT2_PAD_TOKEN
    assert tokenizer.bos_token_id == constants.GPT2_BOS_TOKEN
    assert tokenizer.eos_token_id == constants.GPT2_EOS_TOKEN

    x = tokenizer(["Hello, my dog is cute", "His dog is cute too", "All dogs are cute"], return_tensors="pt", padding="max_length", max_length=16).input_ids

    print("loading model...")
    config = load_model_config(MODEL_CONFIG)
    model = VaeLmModel(VaeLmConfig(**config))

    def pack_hook(x):
        print(f"packing {x.shape}")
        return x

    def unpack_hook(x):
        print(f"unpacking {x.shape}")
        return x

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        out, enc_mus, enc_sigmas, dec_mus, dec_sigmas = model(x)

        # print(out)
        kl = torch.log(dec_sigmas/enc_sigmas) + (enc_sigmas**2 + (enc_mus - dec_mus)**2)/(2*(dec_sigmas**2)) - 0.5
        # print(kl.sum(-1).sum(-1))

        loss = out.sum() + kl.sum()
        loss.backward()


if __name__ == '__main__':

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()
