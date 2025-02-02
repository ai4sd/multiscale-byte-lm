from typing import Literal

import pytest
from pytest_mock import MockerFixture

from mblm.utils.misc import once, retry


class FailThenSuccess:
    def __init__(self, num_fails: int):
        self.max_num_fails = num_fails
        self.has_failed_times = 0

    def run(self):
        if self.has_failed_times < self.max_num_fails:
            self.has_failed_times += 1
            raise Exception("error")
        return True


class TestMisc:
    @pytest.mark.parametrize(
        "n_retries,n_inner_fails,expected_calls,expected_result",
        [
            [0, 0, 1, True],
            [3, 0, 1, True],
            [1, 0, 1, True],
            [0, 1, 1, None],
            [1, 1, 2, True],
            [4, 1, 2, True],
            [0, 2, 1, None],
            [2, 2, 3, True],
        ],
    )
    def test_retry(
        self,
        mocker: MockerFixture,
        n_retries: int,
        n_inner_fails: int,
        expected_calls: int,
        expected_result: Literal[True] | None,
    ):
        try_func = FailThenSuccess(n_inner_fails)
        try_func_spy = mocker.spy(try_func, "run")
        on_error_stub = mocker.stub("on_error")

        retry_wrapper = retry(num_retries=n_retries, on_error=on_error_stub)
        func = retry_wrapper(try_func.run)
        result = func()

        assert result is expected_result
        assert try_func_spy.call_count == expected_calls
        assert on_error_stub.call_count == min(n_inner_fails, n_retries + 1)

    def test_once(self):
        @once
        def func():
            return True

        assert func()
        assert func() is None
