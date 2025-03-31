
import numpy as np
import os
import webdataset as wds
from tqdm.notebook import tqdm

import huggingface_hub as hf

import utils.constants as constants


TEMP_PATH = "temp.tar.gz"

MAX_FILES_IN_SHARD = 1e12
MAX_SHARD_SIZE = 4e9

MIN_INTERVAL = 10


class HfShardWriter(wds.ShardWriter):
    def __init__(self, out_repo, out_path, temp_path=TEMP_PATH, *args, **kwargs):
        self.out_repo = f'{constants.HF_ID}/{out_repo}'
        self.out_path = out_path

        self.temp_path = temp_path
        self.api = hf.HfApi()
        
        hf.create_repo(
            self.out_repo,
            private=True,
            repo_type="dataset",
            exist_ok=True,
        )

        kwargs["maxcount"] = MAX_FILES_IN_SHARD
        kwargs["maxsize"] = MAX_SHARD_SIZE
        super().__init__("", *args, **kwargs)


    def next_stream(self):
        """Close the current stream and move to the next."""
        self.finish()
        if self.fname is None:
            self.fname = self.temp_path
        if self.verbose:
            print(
                "# writing",
                self.fname,
                self.count,
                "%.1f GB" % (self.size / 1e9),
                self.total,
            )
        self.shard += 1
        self.tarstream = wds.TarWriter(self.fname, **self.kw)
        self.count = 0
        self.size = 0

    
    def finish(self):
      super().finish()
      if self.fname is not None and os.path.exists(self.fname):

        self.api.upload_file(
            repo_id=self.out_repo,
            path_or_fileobj=self.fname,
            path_in_repo=os.path.join(self.out_path, f"{self.shard:012d}.tar.gz"),
            repo_type="dataset"
        )

        os.remove(self.fname)


class TokenizerMap:

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
    

    def __call__(self, d):
        
        # batch encode text
        input_ids = self.tokenizer(
            [t[:20*self.max_length] for t in d["text"]],
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np"
        ).input_ids
        
        # convert to list
        out = []
        for curr in input_ids:

            assert np.max(curr) < 2**16, f"Input IDs are too large for uint16: {np.max(curr)} > {(2**16)-1}"
            curr = curr.astype(np.uint16)

            out.append(curr)

        return {"input_ids": out}


def create_split(
    token_iterator,
    repo,
    split,
    num_tokens,
    piece_length,
):
  
    with HfShardWriter(
        out_repo=repo,
        out_path=split,
    ) as sink:
        with tqdm(
            total=num_tokens,
            desc=split,
            mininterval=MIN_INTERVAL
        ) as pbar:

            curr_ind = 0
            total_tested = 0
            total_tokens = 0
            total_pass = 0

            while num_tokens is None or total_tokens < num_tokens:
                try:
                    input_ids = next(token_iterator)["input_ids"]
                except StopIteration:
                    break

                total_tested += 1
                if len(input_ids) < 2 * piece_length:
                    continue
                total_pass += 1

                inn = input_ids[:piece_length]
                out = input_ids[piece_length:2*piece_length]

                assert inn.shape == (piece_length,)
                assert out.shape == (piece_length,)

                sample = {
                    "__key__": f"{curr_ind:012d}",
                    "input_ids.npy": inn,
                    "output_ids.npy": out,
                }
                sink.write(sample)

                curr_ind += 1
                total_tokens += len(input_ids)

                pbar.update(2*piece_length)
                pbar.set_postfix(
                    total=f"{2*piece_length:_}",
                    ind=f"{curr_ind:_}",
                    total_tested=f"{total_tested:_}",
                    total_pass=f"{total_pass:_}",
                    pass_freq=f"{total_pass/total_tested:.3f}",
                    refresh=False
                )
                