import torch

import os
import matplotlib.pyplot as plt

from transformers import AutoTokenizer

from models.zlm import ZLmConfig, ZLmModel
from utils.config_utils import load_config
import utils.constants as constants


MODEL_CONFIG = 'proto-zlm'
CHECKPOINT = 'proto-zlm_beta/000000014000/checkpoint.ckpt'


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

    prompt_2 = """In the heart of the rain-slicked town of Briarhold, tucked between crooked alleys and the smell of wet hay, stood The Soot & Lantern, a tavern known more for its strong mead than its sturdy roof.

Inside, the fire crackled lazily, casting long shadows over the mismatched chairs and scarred tables. It was here, on a particularly stormy evening, that four strangers found themselves drawn together.

First was Kael, a half-elf rogue with eyes like storm clouds and fingers faster than a sneeze. He sat in the corner, nursing a drink and watching the room like a cat in tall grass.

Next came Brina, a dwarven cleric in mud-caked armor, her holy symbol swinging with each heavy step. She ordered a pint, clanked it on Kael’s table without a word, and sat.

Moments later, a lanky human wizard named Fen strode in, robes dripping, nose buried in a book that glowed faintly. He mumbled an incantation, dried his clothes instantly, and joined the growing circle, more interested in the candle’s flicker than the conversation.

Last came Rook—a towering dragonborn fighter with a voice like gravel and a smile that didn’t quite reach his eyes. He kicked the tavern door open, scaring the cat, and asked, “Anyone here looking to split a treasure map four ways?”"""
    tokens_2 = tokenizer(prompt_2, return_tensors='pt', truncation=True, max_length=model.input_length).input_ids
    assert tokens_2.shape[-1] == model.input_length, f"Input length {tokens.shape[-1]} does not match model input length {model.input_length}."

    # print(f"INPUT:\n{tokenizer.decode(tokens[0])}")

    output_tokens, mus = model.sample(
        torch.cat([tokens, tokens], dim=0),
        temperature=1.0
    )
    
    kl = (
        mus[0] - mus[1]
    ).pow(2).sum(-1)

    plt.plot(kl.cpu().detach().numpy())
    plt.savefig("kl.png")


if __name__ == '__main__':

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()