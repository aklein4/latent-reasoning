
import numpy as np
import os
import webdataset as wds
from tqdm import tqdm

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
        
        assert np.max(input_ids) < 2**16, f"Input IDs are too large for uint16: {np.max(input_ids)} > {(2**16)-1}"
        input_ids = input_ids.astype(np.uint16)
        
        # convert to list
        out = []
        for curr in input_ids:
            out.append(curr[curr != self.tokenizer.pad_token_id])

        return {"input_ids": out}


def create_split(
    tokenizer,
    token_iterator,
    repo,
    split,
    num_tokens,
    max_length,
    min_length,
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
            while num_tokens is None or q.trunc_count < num_tokens:
                try:
                    input_ids = next(token_iterator)["input_ids"]
                except StopIteration:
                    break

                input_ids, segment_ids = q(input_ids)
                if input_ids is None:
                    continue

                sample = {
                    "__key__": f"{curr_ind:012d}",
                    "input_ids.npy": input_ids,
                    "segment_ids.npy": segment_ids
                }
                sink.write(sample)
                curr_ind += 1

                pbar.update(q.trunc_count-pbar.n)
                pbar.set_postfix(
                    total=f"{q.trunc_count:_}",
                    ind=curr_ind,
                    perc=(q.trunc_count/q.total_count),
                    q=np.sum(q.filled),
                    q_perc=np.sum(q.filled)/q.q_size,
                    refresh=False
                )
                