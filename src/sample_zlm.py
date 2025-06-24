import torch

import os
import numpy as np

from transformers import AutoTokenizer

from models.zlm import ZLmConfig, ZLmModel
from utils.config_utils import load_config
import utils.constants as constants

import matplotlib.pyplot as plt


MODEL_CONFIG = 'proto-zlm'
CHECKPOINT = 'proto-zlm_beta/000000014000/checkpoint.ckpt'


def slerp(val, low, high):
    """ Batched spherical interpolation between high and low.
    val in [0, 1], 0=low, 1=high.
    """

    assert low.shape == high.shape
    og_shape = low.shape

    low = low.reshape(low.shape[0], -1)
    high = high.reshape(high.shape[0], -1)

    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high

    return res.reshape(og_shape)


def main():


    print("loading model...")
    config = load_config(MODEL_CONFIG, "model")
    model = ZLmModel(ZLmConfig(**config), cpu=True)
    print("Model loaded!")

    print("loading checkpoint...")
    state_dict = torch.load(
        os.path.join(constants.LOCAL_DATA_PATH, CHECKPOINT),
        map_location='cpu'
    )['model']
    model.load_state_dict(state_dict, strict=True)
    print("Checkpoint loaded!")

    print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model.config.base_url
    )
    print("Tokenizer loaded!")

    prompt = """Minecraft is a 3D sandbox video game that has no required goals to accomplish, allowing players a large amount of freedom in choosing how to play the game.[3] The game also features an optional achievement system.[4] Gameplay is in the first-person perspective by default, but players have the option of a second and third-person perspective.[5] The game world is composed of rough 3D objects—mainly cubes, referred to as blocks—representing various materials, such as dirt, stone, ores, tree trunks, water, and lava. The core gameplay revolves around picking up and placing these objects. These blocks are arranged in a 3D grid, while players can move freely around the world. Players can break, or mine, blocks and then place them elsewhere, enabling them to build things.[6] The game also contains a material called redstone, which can be used to make primitive mechanical devices, electrical circuits, and logic gates, allowing for the construction of many complex systems.[7][8] Comparatively, the game's physics system has been described as unrealistic, with nearly all blocks unaffected by gravity."""
    tokens = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=model.input_length).input_ids
    assert tokens.shape[-1] == model.input_length, f"Input length {tokens.shape[-1]} does not match model input length {model.input_length}."

    prompt_2 = """The European hedgehog (Erinaceus europaeus), also known as the West European hedgehog or common hedgehog, is a hedgehog species native to Europe from Iberia and Italy northwards into Scandinavia and westwards into the British Isles.[3] It is a generally common and widely distributed species that can survive across a wide range of habitat types. It is a well-known species, and a favourite in European gardens, both for its endearing appearance and its preference for eating a range of garden pests. While populations are currently stable across much of its range, it is declining severely in Great Britain[2] where it is now Red Listed,[4] meaning that it is considered to be at risk of local extinction. Outside its native range, the species was introduced to New Zealand during the late nineteenth and early twentieth centuries."""
    tokens_2 = tokenizer(prompt_2, return_tensors='pt', truncation=True, max_length=model.input_length).input_ids
    assert tokens_2.shape[-1] == model.input_length, f"Input length {tokens.shape[-1]} does not match model input length {model.input_length}."

    tokens = torch.cat([tokens, tokens_2], dim=0)
    noise = model.sample_noise(tokens)

    output = model.sample(
        tokens,
        temperature=1.0,
        # guidance_scale=3.0,
        # dropout_level=0.05
        noise=noise,
    ).mus

    dists = (output[0][None]- output[1][:, None]).pow(2).mean(-1)

    plt.matshow(dists.detach().cpu().numpy())
    plt.colorbar()

    plt.title("Distance between two samples")
    plt.xlabel("Token index")
    plt.ylabel("Token index")

    plt.tight_layout()
    plt.savefig("distances.png")


if __name__ == '__main__':

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()
