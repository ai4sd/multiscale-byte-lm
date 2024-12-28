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

from dataclasses import dataclass
from typing import Any, Callable, Generator, Generic, Iterator, Protocol, TypeVar

_T = TypeVar("_T", covariant=True)


class _SizedIterable(Protocol[_T]):
    """
    An internal protocol type used by the epoch_cycler, typically to allow Torch
    DataLoaders as iterables for the cycler. Ideally, we'd have the input
    sequence be `Sequence` to facilitate the logic in the epoch cycler. However,
    Dataloaders are not `Sequence` because they do not implement `__getitem__`
    and thus cannot be used to efficiently index. But since DataLoaders are both
    `Sized` and `Iterator`, we can make use that. See
    https://docs.python.org/3/library/collections.abc.html#collections-abstract-base-classes
    """

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[_T]: ...


# TODO: Python 3.12, NamedTuple
@dataclass
class EpochCyclerYield(Generic[_T]):
    epoch: int
    batch: int
    item: _T
    next_epoch: int
    next_batch: int


def epoch_cycler(
    seq: _SizedIterable[_T],
    *,
    before_new_epoch: Callable[[int], Any] | None = None,
    start_epoch: int = 0,
    start_batch: int = 0,
    max_iters: int | None = None,
) -> Generator[EpochCyclerYield[_T], None, None]:
    """
    Infintely iterates over a sequence, yielding batches and their indices as
    well as the current epoch (one complete iteration over the sequence).

    Args:
        seq (_SizedIterable[T]): An object that implements both __size__ and
            __len__ holding items T
        before_new_epoch (Callable[[int], Any] | None = None): Optional
            callback function that is called right **before** the `n`-th starts
            with `n`. Can be used to shuffle a dataset or perform arbitry side effects
        start_epoch (int = 0): Start/resume from this epoch
        start_batch (int = 0): Skip until and yield **starting from** this index
        max_iters (int | None = None): Return when this total amount of batches has
            been yielded across epochs

    Returns:
        Generator (EpochCyclerYield): Yields the epoch, the batch index and the item at
            that index. Batch indices are reset at the start of each epoch (that is,
            they range from 0 to len(seq)). In order to restore the iterator's state,
            the next epoch and next batch index are also returned for convenience.
    """
    epoch = start_epoch
    global_batch_counter = 0
    if start_batch >= len(seq):
        raise IndexError("start_batch is larger than the length of the sequence")

    if before_new_epoch:
        before_new_epoch(epoch)

    it = iter(seq)
    # advance the first iterator to the start batch index
    for _ in range(0, start_batch):
        next(it)

    while True:
        # do not factor out the call to len - if the sequence changes between
        # epochs, we want to account for that. len is O(1), doesn't matter
        for i in range(start_batch, len(seq)):
            if global_batch_counter == max_iters:
                return
            item = next(it)
            if i == len(seq) - 1:
                next_epoch = epoch + 1
                next_batch = 0
            else:
                next_epoch = epoch
                next_batch = i + 1
            yield EpochCyclerYield[_T](epoch, i, item, next_epoch, next_batch)
            global_batch_counter += 1
        start_batch = 0
        epoch += 1

        if before_new_epoch:
            before_new_epoch(epoch)
        it = iter(seq)
