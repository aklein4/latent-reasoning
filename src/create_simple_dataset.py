
import datasets
from transformers import AutoTokenizer

from data_prep.simple_data import create_split, TokenizerMap
import utils.constants as constants


TOKENIZER_URL = "TinyLlama/TinyLlama_v1.1"

DATA_URL = 'HuggingFaceFW/fineweb-edu'
DATA_SUBSET = "sample-100BT"

SAVE_REPO = 'fineweb-edu-TinyLlama'

TRAIN_SIZE = None
VAL_SIZE = 1e8
TEST_SIZE = 1e8

BATCH_SIZE = 1024*8

MAX_LENGTH = 1024


def main():
    
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_URL,
        use_fast=True,
        resume_download=None
    )
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.save_pretrained(
        save_directory=f"{SAVE_REPO}_tokenizer",
        repo_id=f"{constants.HF_ID}/{SAVE_REPO}_tokenizer",
        push_to_hub=True
    )

    dataset = datasets.load_dataset(
        DATA_URL,
        name=DATA_SUBSET,
        streaming=True,
        split="train"
    )
    dataset = dataset.map(
        TokenizerMap(tokenizer, MAX_LENGTH),
        batched=True,
        batch_size=BATCH_SIZE
    )
    data_iterator = iter(dataset)

    create_split(
        data_iterator,
        SAVE_REPO,
        "val",
        VAL_SIZE,
    )

    create_split(
        data_iterator,
        SAVE_REPO,
        "test",
        TEST_SIZE,
    )

    create_split(
        data_iterator,
        SAVE_REPO,
        "train",
        TRAIN_SIZE,
    )


if __name__ == "__main__":
    main()