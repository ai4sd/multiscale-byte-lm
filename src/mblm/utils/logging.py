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

import logging
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)


def create_logger(
    name: str,
    *,
    log_dir: str | Path | None = None,
    noop: bool = False,
) -> logging.Logger:
    """
    Create a logger instance.

    Args:
        name (str): Name of the logger
        log_dir (str | None = None): If defined, log to a file with `name`
            in the `log_dir`. If `None`, log to `stderr` by default
        noop (bool = False): Set to `True` to make logging a noop operation
            and disable all output. This parameter takes precedence over `log_dir`

    """

    logger = logging.getLogger(name)
    if noop:
        handler: logging.Handler = logging.NullHandler()
    elif log_dir:
        log_file = log_dir if isinstance(log_dir, Path) else Path(log_dir)
        log_file.mkdir(exist_ok=True, parents=True)
        handler = logging.FileHandler(log_file / f"{name}.log")
    else:
        handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # because we use basicConfig, we created a root logger.
    # do not propagate messages to the root logger
    logger.propagate = False
    return logger


def shutdown_log_handlers():
    return logging.shutdown()
