
import argparse

import torch

from loaders import get_loader
from models import CONFIG_DICT, MODEL_DICT, load_checkpoint
from trainers import TRAINER_DICT

import utils.constants as constants
from utils.config_utils import load_config


def main(args):

    print("Loading configs...")
    train_config = load_config(args.train_config, kind="train")

    print("Loading model...")
    if args.model_checkpoint is not None:
        model = load_checkpoint(
            args.model_checkpoint,
            strict=(not args.no_strict_load),
            config=load_config(args.model_config, kind="model"),
        )

    else:
        model_config = load_config(args.model_config, kind="model")

        model_type = model_config.pop("model_type")
        model_config_obj = CONFIG_DICT[model_type](**model_config)
        model = MODEL_DICT[model_type](model_config_obj)

    model = model.to(constants.DEVICE)
    model.gradient_checkpointing_enable()
    
    print("Loading data...")
    ds_config = train_config["dataset"]
    loader = get_loader(
        ds_config["name"],
        "train",
        train_config["bs"],
        ds_config["type"],
        ds_config["kwargs"],
        ds_config["branch"],
        streaming=True
    )

    print("Loading trainer...")
    trainer_type = train_config["trainer_type"]
    trainer = TRAINER_DICT[trainer_type](
        args.project,
        args.name,
        train_config,
        debug=args.debug,
        notes=args.notes,
        resume_id=args.resume_id,
        resume_step=args.resume_step,
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

    args.add_argument("--resume_id", type=str, required=False, default=None)
    args.add_argument("--resume_step", type=int, required=False, default=None)

    args.add_argument("--model_config", type=str, required=False, default=None)
    args.add_argument("--model_checkpoint", type=str, required=False, default=None)
    args.add_argument("--no_strict_load", action="store_true")

    args.add_argument("--train_config", type=str, required=True)

    args.add_argument("--debug", action="store_true")
    args.add_argument("--notes", type=str, default=None)

    args = args.parse_args()

    main(args)
