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

from io import BytesIO

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from PIL.Image import Image as PILImage

DEFAULT_ENCODING: str = "utf8"


class Bytes:
    @staticmethod
    def tensor_to_str(t: torch.Tensor, encoding=DEFAULT_ENCODING) -> str:
        return bytes(t.tolist()).decode(encoding)

    @staticmethod
    def tensor_to_image(t: torch.Tensor) -> Image.Image:
        as_np = t.permute(1, 2, 0).numpy()
        return Image.fromarray(as_np)

    @staticmethod
    def str_to_tensor(s: str, encoding=DEFAULT_ENCODING) -> torch.Tensor:
        # frombuffer needs a mutable bytearray, not just bytes
        b = Bytes.str_to_bytearray(s, encoding)
        # note - doing
        # torch.frombuffer(bytearray(s.encode("utf8")), dtype=torch.uint8)
        # is more than 2x as fast as doing
        # torch.tensor(bytearray(s.encode("utf8")), dtype=torch.uint8)
        return torch.frombuffer(b, dtype=torch.uint8)

    @staticmethod
    def str_to_bytearray(s: str, encoding=DEFAULT_ENCODING) -> bytearray:
        # faster than bytearray(string, encoding)
        return bytearray(Bytes.str_to_bytes(s, encoding))

    @staticmethod
    def str_to_bytes(s: str, encoding=DEFAULT_ENCODING) -> bytes:
        return s.encode(encoding)

    @staticmethod
    def bytes_to_str(b: bytes | bytearray, encoding=DEFAULT_ENCODING) -> str:
        return b.decode(encoding)

    @staticmethod
    def byte_list_to_str(b: list[int], encoding=DEFAULT_ENCODING) -> str:
        return bytes(b).decode(encoding)

    @staticmethod
    def byte_list_to_str_safe(b: list[int], default: str, encoding=DEFAULT_ENCODING) -> str:
        try:
            return bytes(b).decode(encoding)
        except Exception:
            return default

    @staticmethod
    def bytes_to_tensor(b: bytes | bytearray) -> torch.Tensor:
        if isinstance(b, bytes):
            # cannot pass bytes directly to torch because bytes are not mutable.
            # wait for https://github.com/pytorch/pytorch/issues/69491
            return torch.tensor(memoryview(b), dtype=torch.uint8)
        return torch.frombuffer(b, dtype=torch.uint8)


class FileStream:
    """
    Wrapper around an ordinary file stream.

    For details on how the buffers are best converted, check
    https://stackoverflow.com/questions/61319551/when-should-one-use-bytesio-getvalue-instead-of-getbuffer
    """

    def __init__(self, buffer: BytesIO):
        self._buffer = buffer

    def to_numpy(self) -> NDArray[np.uint8]:
        """
        Return the buffer as a Numpy uint8 array.
        """
        return np.frombuffer(self._buffer.getvalue(), dtype=np.uint8)

    def to_tensor(self) -> torch.Tensor:
        """
        Return the buffer as a Torch uint8 Tensor.
        """
        return torch.frombuffer(self._buffer.getbuffer(), dtype=torch.uint8)

    def to_buffer(self) -> bytes:
        """
        Return the buffer as bytes.
        """
        return self._buffer.getvalue()

    def to_image(self) -> PILImage:
        """
        Return the buffer as a Pillow image.

        Because the underlying buffer may represent any image format with any
        color space, it is up to you to convert it accordingly.
        """
        return Image.open(self._buffer)
