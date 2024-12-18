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
from contextlib import contextmanager
from typing import Literal

import torch
from pydantic import BaseModel
from torch.distributed import destroy_process_group, init_process_group


class ElasticRunVars(BaseModel):
    local_rank: int
    world_size: int
    is_cuda: bool


def _torchrun_env_variables(is_cuda: bool) -> ElasticRunVars:
    """
    Retrieve environment variables set by torchrun in a distributed setting.
    See https://pytorch.org/docs/stable/elastic/run.html#environment-variables.
    """
    local_rank = os.environ.get("LOCAL_RANK")
    world_size = os.environ.get("WORLD_SIZE")
    assert local_rank and world_size
    return ElasticRunVars(local_rank=int(local_rank), world_size=int(world_size), is_cuda=is_cuda)


@contextmanager
def process_group(*, backend: Literal["gloo", "nccl"] | None = None):
    """
    Context manager for initializing a distributed process group and
    automatically destroying resources upon exit. See
    https://pytorch.org/docs/stable/distributed.html

    Args:
        backend ("gloo", "nccl" or None = None): By default, uses nccl if CUDA is
            available, else gloo. Set this explicitly if you want to use nccl in
            the presence of CUDA-incompatible GPUs

    **Example**::

        with process_group() as run_vars:
            trainer = MyTrainer(
                config,
                gpu_id=run_vars.local_rank,
            ).trainer.train()

    """
    if not backend:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    init_process_group(backend=backend)
    is_cuda = backend == "nccl"
    run_vars = _torchrun_env_variables(is_cuda)

    if backend == "nccl":
        torch.cuda.set_device(run_vars.local_rank)
    try:
        yield run_vars
    finally:
        destroy_process_group()
