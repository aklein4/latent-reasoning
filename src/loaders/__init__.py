""" Dataloaders for the project. """

from torch.utils.data import DataLoader

from loaders.simple import SimpleCollator
from loaders.split import SplitCollator


COLLATOR_DICT = {
    "split": SplitCollator,
    "simple": SimpleCollator
}


import torch

import datasets

import utils.constants as constants


def get_loader(
    name: str,
    split: str,
    bs: int,
    collator_type: str,
    collator_kwargs: dict,
    streaming: bool = True,
)
    dataset = datasets.load_dataset(
        "webdataset",
        data_files=_get_data_files(name),
        split=split,
        streaming=streaming
    )

    collator = COLLATOR_DICT[collator_type](**collator_kwargs["collator_kwargs"])

    return DataLoader(
        dataset=dataset,
        batch_size=bs,
        collate_fn=collator,
        shuffle=False,
        drop_last=True,
    )



def _get_data_files(
    name: str
):
    """ Get datafile urls for the given dataset name.
     - see example at https://huggingface.co/docs/hub/en/datasets-webdataset 
     - see data_prep.token_wds for repo layout
     
    Args:
        name (str): name of the repo to load

    Returns:
        Dict[str, str]: dict of splits and their urls
    """
    data_files = {}
    for split in ["train", "val", "test"]:

        data_files[split] = f"https://huggingface.co/datasets/{constants.HF_ID}/{name}/resolve/main/{split}/*"
    
    return data_files
