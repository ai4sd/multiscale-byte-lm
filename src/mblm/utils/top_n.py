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

import copy
import heapq
from dataclasses import dataclass, field
from typing import Generic, Iterator, SupportsFloat, TypeVar

_T = TypeVar("_T")


@dataclass(order=True)
class _Entry(Generic[_T]):
    val: float
    item: _T = field(compare=False)


class TopN(Generic[_T]):
    """
    A priority queue based on a heap data structure. By default, a min heap is created,
    with the first element being the smallest. A max heap can be specified via `top_largest`.

    Args:
        n (int): Max number of items to store
        deep_copy (bool = `False`): Create a deep copy of elements
        top_largest (bool = `False`): If true, store the `n` largest items instead of
            the smallest items

    Returns:
        self (TopN[T]): An iterable instance of `self` that returns the smallest items
            (or largest of `top_largest` is set to `True`) first

    **Example**::

        # create the heap
        top_n = TopN[str](2)

        # add items in random order
        top_n.add((1, "a"))
        top_n.add((3, "b"))
        top_n.add((2, "c"))

        for idx, item in enumerate(top_n):
            ...
            # iteration 1: (1, "a")
            # iteration 2: (2, "c")
    """

    def __init__(self, n: int, *, deep_copy: bool = False, top_largest: bool = False):
        self._max_heap_items = n
        self._heap: list[_Entry] = []
        self._top_largest = top_largest
        self._heap_factor = 1.0 if top_largest else -1.0
        self._deep_copy = deep_copy

    def add(self, item: tuple[SupportsFloat, _T]) -> None:
        """
        Add an item to the queue. The first tuple entry is used to
        determine the position of the newly added element
        """
        val, data = item
        if self._deep_copy:
            data = copy.deepcopy(data)
        entry = _Entry(self._heap_factor * float(val), data)
        curr_heap_items = len(self._heap)

        # heap is underfull
        if curr_heap_items < self._max_heap_items:
            heapq.heappush(self._heap, entry)
        # heap is full, maybe add
        elif entry > self._heap[0]:
            heapq.heapreplace(self._heap, entry)

    def get_top(self, best: int | None = None) -> list[tuple[float, _T]]:
        """
        Get the n `best` (i.e., smallest or largest depending on initialization)
        items from the queue. If `best` is not specified, all items are returned
        """
        if not best:
            best = self._max_heap_items
        top_n = sorted(
            [(self._heap_factor * x.val, x.item) for x in self._heap],
            key=lambda x: x[0],
            reverse=self._top_largest,
        )
        return top_n[:best]

    def __iter__(self) -> Iterator[tuple[float, _T]]:
        return iter(self.get_top())

    def __len__(self) -> int:
        return len(self._heap)
