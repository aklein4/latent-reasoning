import torch

import os
import numpy as np

from transformers import AutoTokenizer

from models import load_checkpoint
import utils.constants as constants

import matplotlib.pyplot as plt


CHECKPOINT = 'local_data/proto-zlm_hybrid-alpha/000000030000'


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
    model = load_checkpoint(
        CHECKPOINT,
        cpu=True,
    )
    print("Model loaded!")

    print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model.config.base_url
    )
    print("Tokenizer loaded!")

    prompt = """Minecraft is a 3D sandbox video game that has no required goals to accomplish, allowing players a large amount of freedom in choosing how to play the game.[3] The game also features an optional achievement system.[4] Gameplay is in the first-person perspective by default, but players have the option of a second and third-person perspective.[5] The game world is composed of rough 3D objects—mainly cubes, referred to as blocks—representing various materials, such as dirt, stone, ores, tree trunks, water, and lava. The core gameplay revolves around picking up and placing these objects. These blocks are arranged in a 3D grid, while players can move freely around the world. Players can break, or mine, blocks and then place them elsewhere, enabling them to build things.[6] The game also contains a material called redstone, which can be used to make primitive mechanical devices, electrical circuits, and logic gates, allowing for the construction of many complex systems.[7][8] Comparatively, the game's physics system has been described as unrealistic, with nearly all blocks unaffected by gravity."""
    tokens = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=model.input_length).input_ids
    assert tokens.shape[-1] == model.input_length, f"Input length {tokens.shape[-1]} does not match model input length {model.input_length}."

    noise_1 = model.sample_noise(tokens)
    noise_2 = model.sample_noise(tokens)

    tokens = tokens.repeat(7, 1)
    noise = torch.cat(
        [
            slerp(t, noise_1, noise_2)
            for t in np.linspace(0.0, 0.1, 7)
        ],
        dim=0
    )

    output = model.sample(
        tokens,
        noise=noise,
        temperature=0.75,
        # boost_scale=boosts,
    )

    with open("output.txt", "w") as f:

        f.write("\n === INPUT === \n\n")
        f.write(tokenizer.decode(tokens[0], skip_special_tokens=False))

        for i in range(len(output.tokens)):
            f.write(f"\n\n === OUTPUT {i} === \n\n")
            f.write(tokenizer.decode(output.tokens[i], skip_special_tokens=False))


if __name__ == '__main__':

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()
