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

from typing import Any, Callable, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


def retry(
    num_retries: int,
    on_error: Callable[[Exception, int], Any] | None = None,
):
    """Retry a function `n` times. The function is called at maximum `n` times"""
    assert num_retries >= 0, "num_retries must be non-negative"

    def wrapper(f: Callable[P, T]) -> Callable[P, T | None]:
        def inner(*args: P.args, **kwargs: P.kwargs) -> T | None:
            attempted_retries = 0
            while attempted_retries <= num_retries:
                try:
                    return f(*args, **kwargs)
                except Exception as err:
                    if on_error:
                        on_error(err, num_retries - attempted_retries)
                    attempted_retries += 1

            return None

        return inner

    return wrapper
