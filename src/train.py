
import argparse

from loaders import get_loader
from models import CONFIG_DICT, MODEL_DICT
from trainers import TRAINER_DICT

import utils.constants as constants
from utils.config_utils import load_config


def main(args):

    print("Loading configs...")
    model_config = load_config(args.model_config, kind="model")
    train_config = load_config(args.train_config, kind="train")

    print("Loading model...")
    model_type = model_config.pop("model_type")
    model_config_obj = CONFIG_DICT[model_type](**model_config)
    model = MODEL_DICT[model_type](model_config_obj).to(constants.DEVICE)

    print("Loading data...")
    loader = get_loader(
        train_config["ds_type"],
        train_config["ds_kwargs"],
        train_config["bs"],
    )

    print("Loading trainer...")
    trainer_type = train_config["trainer_type"]
    trainer = TRAINER_DICT[trainer_type](
        args.project,
        args.name,
        train_config,
        debug=args.debug,
        notes=args.notes,
    )

    print("Entering trainer...")
    trainer.train(
        model,
        loader
    )


if __name__ == '__main__':
  
    args = argparse.ArgumentParser()
    args.add_argument("--project", type=str, required=True)
    args.add_argument("--name", type=str, required=True)
    args.add_argument("--model_config", type=str, required=True)
    args.add_argument("--train_config", type=str, required=True)
    args.add_argument("--debug", action="store_true")
    args.add_argument("--notes", type=str, default=None)
    args = args.parse_args()

    main(args)
