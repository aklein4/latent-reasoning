import torch

from models.zlm_full import ZLmFullConfig, ZLmFullModel
from utils.config_utils import load_config
import utils.constants as constants


MODEL_CONFIG = 'test-zlm-full'


def main():

    print("loading model...")
    config = load_config(MODEL_CONFIG, "model")
    model = ZLmFullModel(ZLmFullConfig(**config), cpu=True)
    print("Model loaded!")

    input_ids = torch.randint(
        0, 100,
        size=(3, model.input_length),
        dtype=torch.long
    )
    output_ids = torch.randint(
        0, 100,
        size=(3, model.output_length),
        dtype=torch.long
    )

    print("Running model...")
    out = model(input_ids, output_ids)
    loss = out.lm_logits.mean() + (out.encoder_mus - out.generator_mus).mean()
    loss.backward()
    print("Model run complete!")

    with open("gradients.txt", "w") as f:

        f.write("\n === GRADIENTS === \n\n")
        for n, p in model.named_parameters():
            if p.grad is not None:
                f.write(f"{n}\n")

        f.write("\n === NO GRADIENT === \n\n")
        for n, p in model.named_parameters():
            if p.grad is None:
                f.write(f"{n}\n")

    print("Output shapes:")
    for k, v in out.items():
        print(k, v.shape)


if __name__ == '__main__':

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()
