from __future__ import annotations

__copyright__ = """MIT License

Copyright (c) 2025 - IBM Research

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
from typing import TYPE_CHECKING, Generator

import torch
import tqdm
from pydantic import BaseModel
from typing_extensions import Unpack

from mblm.data.datasets import DistributedDataset, DistributedDatasetConfig
from mblm.data.types import BatchMaskedForMLM, ModelMode
from mblm.data.utils import Bytes

# from mblm.train.mblm import masked_dataset_registry

if TYPE_CHECKING:
    from mblm.train.mblm import TrainMaskedEntryConfig


class PG19ModelGeneration(BaseModel):
    id_model: str
    book_id: str
    book_txt_offset: int
    ctx_len: int
    generated: list[int]
    truth: list[int]
    ce: float
    generation_time: float
    timestamp: str


# @masked_dataset_registry.register("maskedPG19")
class PG19Masked(DistributedDataset[BatchMaskedForMLM]):
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
        masked_token_id: int = -100,
        masking_proba: float = 0.15,
        load_mininterval: int = 30,
        display_load_progress: bool = True,
        padding_token_id: int = -101,
        **config: Unpack[DistributedDatasetConfig],
    ):
        root = Path(data_dir)
        if mode == ModelMode.VALID:
            data_path = root / "validation"
        else:
            data_path = root / mode.value
        self.txt_files = [data_path / file for file in os.listdir(data_path)]
        self.masking_proba = masking_proba
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
        self.masked_token_id = masked_token_id
        self.padding_token_id = padding_token_id
        if masked_token_id == padding_token_id:
            raise ValueError("You can't set the padding and the mask with the same value")

        super().__init__(
            data_size=self.data.numel(),
            is_sequential=True,
            **config,
        )

    @staticmethod
    def from_train_entry_config(
        config: TrainMaskedEntryConfig,
        mode: ModelMode,
        worker_id: int,
        num_workers: int,
    ) -> DistributedDataset[BatchMaskedForMLM]:
        return PG19Masked(
            data_dir=config.io.dataset_dir,
            masking_proba=config.train.masking_proba,
            masked_token_id=config.params.mask_token_id,
            mode=mode,
            seq_len=config.params.input_seq_len,
            worker_id=worker_id,
            num_workers=num_workers,
        )

    @staticmethod
    def supports_test_mode() -> bool:
        return True

    def get_sample(self, from_idx: int) -> BatchMaskedForMLM:
        """
        Get a sample with a loss mask. This method is required by the
        DistributedDataset superclass.
        """
        sample = self.data[from_idx : from_idx + self.seq_len].long()
        mask = torch.rand(sample.size()) < self.masking_proba
        tokens_masked = sample.clone()

        # TODO implement same strategy as BERT and even when token is masked, sometimes copy the correct token, not the masked_token_id
        tokens_masked[mask] = self.masked_token_id
        # Padd if necessary, should only be needed when from_idx == len(self)
        if sample.size(-1) != self.seq_len:
            # padd with tensors with padding_token_id
            pad_tensor = self.padding_token_id * torch.ones(self.seq_len - sample.size(-1))
            tokens_masked = torch.concat((tokens_masked, pad_tensor))
            # pad_tensor * 0 ensures that the loss is never computed over the padding tokens
            # as 1 shows MASKED elements and 0 non-MASKED token
            mask = torch.concat((mask, pad_tensor * 0))
            sample = torch.concat((sample, pad_tensor))
        return tokens_masked.long(), mask.bool(), sample.long()

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
