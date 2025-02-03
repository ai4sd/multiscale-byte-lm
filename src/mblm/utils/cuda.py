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
from functools import partial
from typing import Literal

import torch
import torch.version
from torch.types import Device


@dataclass
class CudaProperties:
    cuda_available: bool
    cuda_version: str | None
    cuda_devices: list[str]


def cuda_properties() -> CudaProperties:
    is_cuda_available = torch.cuda.is_available()
    devices: list[torch._C._CudaDeviceProperties] = [
        torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())
    ]
    return CudaProperties(
        cuda_available=is_cuda_available,
        cuda_version=torch.version.cuda,
        cuda_devices=[f"{d.name}_{d.total_memory}_{d.major}.{d.minor}" for d in devices],
    )


@dataclass
class CudaMemorySnapshot:
    allocated: float
    allocated_max: float
    reserved: float
    reserved_max: float
    total: float
    unit: Literal["GB"] = "GB"


_gb_conv_factor = 1024**3


def cuda_memory_snapshot(device: Device | None, dec_prec: int = 3) -> CudaMemorySnapshot:
    if not torch.cuda.is_available() or device == "cpu":
        return CudaMemorySnapshot(
            allocated=0.0,
            allocated_max=0.0,
            reserved=0.0,
            reserved_max=0.0,
            total=0.0,
        )
    # report only in gb for now
    _round = partial(round, ndigits=dec_prec)
    alloc = _round(torch.cuda.memory_allocated(device) / _gb_conv_factor)
    max_alloc = _round(torch.cuda.max_memory_allocated(device) / _gb_conv_factor)
    res = _round(torch.cuda.memory_reserved(device) / _gb_conv_factor)
    max_res = _round(torch.cuda.max_memory_reserved(device) / _gb_conv_factor)
    current_device: torch._C._CudaDeviceProperties = torch.cuda.get_device_properties(device)
    total = _round(current_device.total_memory / _gb_conv_factor)
    return CudaMemorySnapshot(alloc, max_alloc, res, max_res, total)
