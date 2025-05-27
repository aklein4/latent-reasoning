import torch

from models.ibml import IBMLConfig, IBMLModel
from utils.config_utils import load_config
import utils.constants as constants


MODEL_CONFIG = 'test-ibml'


def main():

    print("loading model...")
    config = load_config(MODEL_CONFIG, "model")
    model = IBMLModel(IBMLConfig(**config)).to(constants.DEVICE)
    print("Model loaded!")

    input_ids = torch.randint(
        0, 100,
        size=(3, 7),
        dtype=torch.long
    ).to(constants.DEVICE)
    memory_mask = torch.randn((3, 7)).to(constants.DEVICE)
    prev_mats = torch.randn(
        (22, 2048, 2048),
    ).to(constants.DEVICE)

    print("Running model...")
    with torch.autocast("cuda", torch.bfloat16):
        out = model(input_ids, memory_mask=memory_mask, prev_mats=prev_mats, mat_beta=0.9)
    print("Model run complete!")

    print("Output shapes:")
    for k, v in out.items():
        print(k, v.shape)


if __name__ == '__main__':
    main()
