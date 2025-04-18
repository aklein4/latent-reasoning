""" Dataloaders for the project. """

from loaders.split import SplitCollator
from loaders.zae import ZAECollator

COLLATOR_DICT = {
    "split": SplitCollator,
    "zae": ZAECollator,
}


from torch.utils.data import DataLoader

import datasets

from utils.data_utils import get_hf_files
import utils.constants as constants


def get_loader(
    name: str,
    split: str,
    bs: int,
    collator_type: str,
    collator_kwargs: dict,
    branch: str = "main",
    streaming: bool = True,
):
    dataset = datasets.load_dataset(
        "webdataset",
        data_files=get_hf_files(constants.HF_ID, name, branch=branch),
        split=split,
        streaming=streaming
    )

    collator = COLLATOR_DICT[collator_type](**collator_kwargs)

    return DataLoader(
        dataset=dataset,
        batch_size=bs,
        collate_fn=collator,
        shuffle=False,
        drop_last=True,
    )
