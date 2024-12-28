import random

import pytest
from pytest_mock import MockerFixture

from mblm.train.core.iter import epoch_cycler


class TestEpochCycler:
    def test_callback(self, mocker: MockerFixture):
        # kind of a verbose test overall but readable
        data: list[int] = [-1] * 5

        new_epoch_stub = mocker.stub("before_new_epoch")

        cycler = epoch_cycler(
            data,
            before_new_epoch=new_epoch_stub,
        )

        assert all([v == -1 for v in data])

        # ------  epoch 0
        n = next(cycler)
        assert n.epoch == 0 and n.batch == 0
        assert n.next_epoch == 0 and n.next_batch == 1
        new_epoch_stub.assert_called_with(0)

        n = next(cycler)
        assert n.epoch == 0 and n.batch == 1

        n = next(cycler)
        assert n.epoch == 0 and n.batch == 2

        n = next(cycler)
        assert n.epoch == 0 and n.batch == 3

        # last iteration of of epoch 0
        n = next(cycler)
        assert n.epoch == 0 and n.batch == 4
        assert n.next_epoch == 1 and n.next_batch == 0  # peek into the future

        # ------ epoch 1
        n = next(cycler)
        assert n.epoch == 1 and n.batch == 0
        assert n.next_epoch == 1 and n.next_batch == 1
        new_epoch_stub.assert_called_with(1)

        n = next(cycler)
        assert n.epoch == 1 and n.batch == 1

        n = next(cycler)
        assert n.epoch == 1 and n.batch == 2

        n = next(cycler)
        assert n.epoch == 1 and n.batch == 3

        n = next(cycler)
        assert n.epoch == 1 and n.batch == 4
        assert n.next_epoch == 2 and n.next_batch == 0  # peek into the future

        # ------ epoch 2
        n = next(cycler)
        assert n.epoch == 2 and n.batch == 0
        assert n.next_epoch == 2 and n.next_batch == 1
        new_epoch_stub.assert_called_with(2)

        assert new_epoch_stub.call_count == 3

    @pytest.mark.parametrize("exp_start_epoch,exp_start_batch", [(1, 3), (2, 3), (0, 1)])
    def test_resume_middle(self, exp_start_epoch: int, exp_start_batch: int):
        epoch_len = 5

        data = list(range(epoch_len))
        cycler = epoch_cycler(
            data,
            start_batch=exp_start_batch,
            start_epoch=exp_start_epoch,
        )

        # a single epoch has length 5
        n = next(cycler)
        epoch, batch, item = n.epoch, n.batch, n.item

        # make sure we start at an arbitrary epoch with a start index
        assert epoch == exp_start_epoch
        assert batch == exp_start_batch
        assert item == data[exp_start_batch]

        # skip through
        while epoch == exp_start_epoch:
            n = next(cycler)
            epoch, batch, item = n.epoch, n.batch, n.item

        # after an epoch has passed, resume from beginning of sequence
        assert epoch == exp_start_epoch + 1
        assert batch == 0
        assert item == data[0]

    @pytest.mark.parametrize("max_iters", [0, 1, 2, 3, 4, 5, 6, 7])
    def test_max_iters(self, max_iters: int):
        epoch_len = 5
        data = list(range(epoch_len))
        # regardless of the start, we aim for a number of target batches
        start_epoch = random.choice(data)
        start_batch = random.choice(data)

        cycler = epoch_cycler(
            data,
            max_iters=max_iters,
            start_epoch=start_epoch,
            start_batch=start_batch,
        )
        num_iters = 0
        for _ in cycler:
            num_iters += 1

        assert num_iters == max_iters

    def test_changing_epoch_len(self):
        epoch_len = 2
        data = list(range(epoch_len))
        cycler = epoch_cycler(data)
        # ------  epoch 0
        # cycle to the end of epoch, 2 iterations
        n = next(cycler)  # 1
        n = next(cycler)  # 2
        assert n.epoch == 0 and n.batch == 1
        assert n.next_epoch == 1 and n.next_batch == 0

        # ------  epoch 1
        # mutate in place -> the next epoch should only have 1 iteration!
        data.pop()
        n = next(cycler)  # should reach end of epoch
        assert n.epoch == 1 and n.batch == 0
        assert n.next_epoch == 2

        # ------  epoch 2
        # mutate in place -> the next epoch should have 3 iterations!
        data.extend([1, 2])
        assert data == [0, 1, 2]
        n = next(cycler)  # 1
        n = next(cycler)  # 2
        n = next(cycler)  # 3
        assert n.epoch == 2 and n.batch == 2
        assert n.next_epoch == 3 and n.next_batch == 0

    def test_resume_raises(self):
        with pytest.raises(IndexError):
            cycler = epoch_cycler(
                seq=range(2),  # 0, 1
                start_batch=2,
            )
            next(cycler)
