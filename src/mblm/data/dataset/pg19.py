__copyright__ = """MIT License

Copyright (c) 2024 - IBM Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

import os
import random
from pathlib import Path
from typing import Generator

import torch
import tqdm
from typing_extensions import Unpack

from mblm.data.datasets import BatchWithLossMask, DistributedDataset, DistributedDatasetConfig
from mblm.data.types import ModelMode
from mblm.data.utils import Bytes


class PG19(DistributedDataset[BatchWithLossMask]):
    """
    https://github.com/google-deepmind/pg19

    The `data_dir` is expected to be the of the exact structure as the
    original dataset, although only the test, train and validation folders
    are strictly needed:
        ├── LICENSE
        ├── README.md
        ├── metadata.csv
        ├── test
        ├── train
        └── validation

    """

    def __init__(
        self,
        data_dir: str | Path,
        mode: ModelMode,
        load_mininterval: int = 30,
        display_load_progress: bool = True,
        **config: Unpack[DistributedDatasetConfig],
    ):
        root = Path(data_dir)
        if mode == ModelMode.VALID:
            data_path = root / "validation"
        else:
            data_path = root / mode.value
        self.txt_files = [data_path / file for file in os.listdir(data_path)]

        data_buff = bytearray()
        for file in tqdm.tqdm(
            self.txt_files,
            desc=f"Loading pg19 {data_path}",
            mininterval=load_mininterval,
            disable=not display_load_progress,
        ):
            with Path.open(file, "rb") as f:
                data_buff.extend(f.read())
        self.data = Bytes.bytes_to_tensor(data_buff)
        super().__init__(
            data_size=self.data.numel(),
            is_sequential=True,
            **config,
        )

    def get_sample(self, from_idx: int) -> BatchWithLossMask:
        """
        Get a sample with a loss mask. This method is required by the
        DistributedDataset superclass.
        """
        sample = self.data[from_idx : from_idx + self.seq_len].long()
        return sample, torch.ones_like(sample, dtype=torch.float16)

    def book(self, name: str) -> str:
        """
        Get a book by its name (e.g., `44381.txt`) return its content as a
        string
        """
        for candidate in self.txt_files:
            if candidate.name == name:
                with Path.open(candidate, "r", encoding="utf8") as f:
                    return f.read()
        raise ValueError(f"Book {name} does not exist")

    def iter_sequences_rand(self) -> Generator[torch.Tensor, None, None]:
        """
        Iterate over random sequences across books of PG19
        """
        max_sample_start_idx = len(self.data) - self.seq_len - 1
        while True:
            idx = random.randint(0, max_sample_start_idx)
            yield self.data[idx : idx + self.seq_len]

    def iter_books(self, shuffle: bool = False) -> Generator[tuple[str, str], None, None]:
        """
        Iterate over all the books in PG19, possibly in random order. Return an
        iterator over the index of the book and its content as a string
        """
        txt_file_idxs = list(range(len(self.txt_files)))
        if shuffle:
            random.shuffle(txt_file_idxs)
        for i in txt_file_idxs:
            book = self.txt_files[i]
            with Path.open(book, "r", encoding="utf8") as f:
                yield book.name, f.read()
