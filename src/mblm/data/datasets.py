from __future__ import annotations

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


from abc import ABC, abstractmethod
from typing import TypeVar

from torch.utils.data import Dataset
from typing_extensions import TypedDict, Unpack

_T = TypeVar("_T")


class DistributedDatasetConfig(TypedDict):
    seq_len: int
    worker_id: int
    num_workers: int


# TODO: Python 3.12, Self
class DistributedDataset(ABC, Dataset[_T]):
    """
    Custom Dataset class that distributes the data across a predefined number of
    workers. It implements all the methods needed to qualify as Dataset. See
    https://pytorch.org/docs/stable/data.html#map-style-datasets

    If `is_sequential` is true, then the underlying data is assumed as
    sequential chunks - a sequence - from an interable (e.g., a list of
    integers). If false, it is assumed that each sampling retrieves only one
    element.

    The most important parameter is `enable_offset`, which affects sampling of
    the underlying data: When set to `True`, the methods `self.offset_to` and
    `self.offset_one` enable dynamic shifting of the starting index when
    retrieving data sequences

    - `self.offset_to(offset)` sets the current sample offset to a specified
      value, allowing data to be accessed from different starting points within
      each sequence, introducing variability in the sampled data
    - `self.offset_one()` increments the sample offset by 1

    When `enable_offset` is false, the dataset fetches sequences starting from
    fixed positions based on the given index, with no offset applied (like a
    regular Dataset). Enabling the offset can be useful in tasks like sequence
    modeling where shifting the sequence start positions can improve the model's
    robustness or generalization by providing varied training inputs. See the
    test for this class for details on the effects
    """

    def __init__(
        self,
        data_size: int,
        is_sequential: bool,
        **kwargs: Unpack[DistributedDatasetConfig],
    ):
        num_workers, worker_id = kwargs["num_workers"], kwargs["worker_id"]
        seq_len = kwargs["seq_len"]

        assert num_workers > 0, "num_workers must be positive"
        assert (
            worker_id < num_workers
        ), f"worker_id ({worker_id}) must be smaller than num_workers ({num_workers})"

        worker_numel = data_size // num_workers
        if is_sequential:
            assert worker_numel >= seq_len, "Worker's data is too small"
        else:
            assert worker_numel >= 1

        self._worker_range_start = worker_id * worker_numel
        self._worker_range_end = self._worker_range_start + worker_numel

        self._worker_numel = worker_numel

        # public properties that may be accessed
        self.is_sequential = is_sequential
        self.seq_len = seq_len
        self.sample_offset = 0

    def offset_to(self, offset: int) -> DistributedDataset[_T]:
        """
        Offset for sequential sampling - has no effect if this `is_sequential`
        is false!
        """
        if not self.is_sequential:
            return self
        self.sample_offset = (offset) % self.seq_len
        return self

    def offset_one(self) -> DistributedDataset[_T]:
        return self.offset_to(self.sample_offset + 1)

    @abstractmethod
    def get_sample(self, from_idx: int) -> _T:
        """
        Abstract method that must be implemented to tell this superclass how to
        get a single data sample. By construction, a dataset is abstract and may
        hold different kinds of data such as a single Tensor containing
        sequences for autoregressive modelling or a tuple of (features, labels)
        Tensors.

        When you want to sample distinct items from your dataset, you can
        retrieve them directly at the index and use `self.seq_len` to pad or
        prepare your data. This is only enabled when the base class has been
        created with `is_sequential` set to false. If created with true, then
        you should sample from as `from_idx : from_idx + self.seq_len`
        """
        ...

    def __len__(self) -> int:
        """
        The total number of indexable elements in the dataset, e.g., the number
        of sequences. This property accounts for both sequence length and
        offsets and is thus strictly smaller or equal to `num_items`. It is
        always strictly positive. This number is used for the `__len__`
        implementation.
        """
        if self.is_sequential:
            available = self._worker_numel - self.sample_offset
            return max(available // self.seq_len, 0)
        else:
            return self._worker_numel

    def __getitem__(self, idx: int) -> _T:
        idx_mult = self.seq_len if self.is_sequential else 1
        from_idx = (idx * idx_mult) + self.sample_offset + self._worker_range_start
        return self.get_sample(from_idx)
