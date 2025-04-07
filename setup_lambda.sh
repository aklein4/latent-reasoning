#!/bin/bash

: '
Setup a TPU VM to use the repo.
 - MUST RUN WITH dot (.) command to set the environment variables in the current shell.

Arguments:
    $1: Huggingface token
    $2: wandb token

Example:
    . setup_vm.sh <HF_TOKEN> <WANDB_TOKEN>
'

# install torch stuff
pip install torch==2.6.0

# install extras
pip install -U transformers datasets webdataset wandb

# login to huggingface
huggingface-cli login --token $1 --add-to-git-credential

# login to wandb
python -m wandb login $2
