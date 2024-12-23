from collections import Counter

import pytest
import torch
from typing_extensions import Unpack

from mblm.data.datasets import DistributedDataset, DistributedDatasetConfig


class MySequentialDataset(DistributedDataset[list[int]]):
    def __init__(
        self,
        data: list[int],
        seq_len: int,
        *,
        worker_id: int,
        num_workers: int,
    ):
        super().__init__(
            data_size=len(data),
            seq_len=seq_len,
            is_sequential=True,
            worker_id=worker_id,
            num_workers=num_workers,
        )
        self._data = data

    def get_sample(self, from_idx):
        return self._data[from_idx : from_idx + self.seq_len]


class MyDataset(DistributedDataset[int]):
    def __init__(
        self,
        data: list[int],
        seq_len: int,
        *,
        worker_id: int,
        num_workers: int,
    ):
        super().__init__(
            data_size=len(data),
            seq_len=seq_len,
            is_sequential=False,
            worker_id=worker_id,
            num_workers=num_workers,
        )
        self._data = data

    def get_sample(self, from_idx):
        return self._data[from_idx]


class TestDistributedDataset:
    def test_offset_single_worker(self):
        data = list(range(14))
        sequence_len = 3
        dataset = MySequentialDataset(data, sequence_len, worker_id=0, num_workers=1)
        assert len(dataset) == 4  # (14 // 1) // 3

        assert dataset[0] == [0, 1, 2]
        assert dataset[1] == [3, 4, 5]
        assert dataset[2] == [6, 7, 8]
        assert dataset[3] == [9, 10, 11]

        dataset.offset_one()
        assert len(dataset) == 4
        assert dataset[0] == [1, 2, 3]
        assert dataset[1] == [4, 5, 6]
        assert dataset[2] == [7, 8, 9]
        assert dataset[3] == [10, 11, 12]

        dataset.offset_to(2)
        assert len(dataset) == 4
        assert dataset[0] == [2, 3, 4]
        assert dataset[1] == [5, 6, 7]
        assert dataset[2] == [8, 9, 10]
        assert dataset[3] == [11, 12, 13]

        # full cycle
        dataset.offset_one()
        assert len(dataset) == 4
        assert dataset[0] == [0, 1, 2]
        assert dataset[1] == [3, 4, 5]
        assert dataset[2] == [6, 7, 8]
        assert dataset[3] == [9, 10, 11]

    def test_offset_single_worker_long_seq(self):
        # for long sequences, we expect that length decreases over time
        data = list(range(13))
        sequence_len = 6
        dataset = MySequentialDataset(data, sequence_len, worker_id=0, num_workers=1)
        assert len(dataset) == 2
        assert dataset[0] == [0, 1, 2, 3, 4, 5]
        assert dataset[1] == [6, 7, 8, 9, 10, 11]

        dataset.offset_one()
        assert len(dataset) == 2
        assert dataset[0] == [1, 2, 3, 4, 5, 6]
        assert dataset[1] == [7, 8, 9, 10, 11, 12]

        dataset.offset_to(4)
        assert len(dataset) == 1  # only one full sequence can be retrieved now
        assert dataset[0] == [4, 5, 6, 7, 8, 9]

        dataset.offset_one()
        dataset.offset_one()
        assert len(dataset) == 2
        assert dataset[0] == [0, 1, 2, 3, 4, 5]
        assert dataset[1] == [6, 7, 8, 9, 10, 11]

    @pytest.mark.parametrize("seq_len,range_end", [(4, 33), (8, 55), (7, 42), (5, 21), (5, 20)])
    def test_offset_two_workers(self, seq_len: int, range_end: int):
        num_workers = 2
        data = list(range(0, range_end))
        d1 = MySequentialDataset(data, seq_len, worker_id=0, num_workers=num_workers)
        d2 = MySequentialDataset(data, seq_len, worker_id=1, num_workers=num_workers)

        all_items: Counter[int] = Counter()
        # test modulo op, cycle offset twice
        for _ in range(seq_len * 2):
            d1.offset_one()
            d2.offset_one()
            assert len(d1) == len(d2)

            for seq_idx in range(len(d1) - 1):
                assert d1[seq_idx][-1] < d1[seq_idx + 1][0]
                assert len(d1[seq_idx]) == seq_len
                # add all sequence start elements to the counter
                all_items.update([d1[seq_idx][0]])
                all_items.update([d2[seq_idx][0]])

        # make sure every element has been the first item twice because we've
        # cycled the offset twice
        assert all([count == 2 for count in all_items.values()])

    def test_non_sequential_ds(self):
        dataset = MyDataset(
            data=list(range(21)),
            seq_len=-1,  # does not matter
            worker_id=0,
            num_workers=1,
        )
        assert len(dataset) == 21
        assert dataset[0] == 0
        assert dataset[1] == 1
        assert dataset[9] == 9

        dataset.offset_one()  # should have no effect but warn
        assert dataset[0] == 0

        dataset = MyDataset(
            data=list(range(21)),
            seq_len=-1,  # does not matter
            worker_id=1,
            num_workers=2,
        )
        assert len(dataset) == 10
        assert dataset[0] == 10
        assert dataset[1] == 11
        assert dataset[9] == 19

    def test_internal_validation_sequential(self):
        # data from 0 to 9, 10 elements
        sequence_len = 10
        data = torch.arange(start=0, end=sequence_len)

        class MyDataset(DistributedDataset):
            def __init__(
                self,
                data: torch.Tensor,
                is_sequential: bool,
                **config: Unpack[DistributedDatasetConfig],
            ):
                super().__init__(
                    data.numel(),
                    is_sequential=is_sequential,
                    **config,
                )

            def get_sample(self, from_idx): ...

        with pytest.raises(AssertionError) as msg:
            MyDataset(
                data,
                seq_len=sequence_len,
                is_sequential=True,
                worker_id=1,
                num_workers=1,
            )
            assert msg.value == "worker_id (2) must be smaller than num_workers (2)"

        with pytest.raises(AssertionError) as msg:
            MyDataset(
                data,
                seq_len=1,
                is_sequential=True,
                worker_id=0,
                num_workers=1,
            )
            assert msg.value == "Worker's data is too small"
        with pytest.raises(AssertionError) as msg:
            MyDataset(
                data,
                seq_len=1,
                is_sequential=True,
                worker_id=0,
                num_workers=1,
            )
            assert msg.value == "Worker's data is too small"
        try:
            MyDataset(
                data,
                is_sequential=False,
                seq_len=9999,
                worker_id=0,
                num_workers=1,
            )
        except AssertionError:
            pytest.fail(
                "Sequence length should not be validated for non-sequential dataset",
            )
