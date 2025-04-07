import torch

from models.zlm import ZLmConfig, ZLmModel
from utils.config_utils import load_config
import utils.constants as constants


MODEL_CONFIG = 'test-zlm'


def main():

    print("loading model...")
    config = load_config(MODEL_CONFIG, "model")
    model = ZLmModel(ZLmConfig(**config))
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
    print("Model run complete!")

    print("Output shapes:")
    for k, v in out.items():
        print(k, v.shape)


if __name__ == '__main__':

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()
