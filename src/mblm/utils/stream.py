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

import codecs
import io
from types import TracebackType
from typing import IO, BinaryIO, Literal, TextIO, overload


class ByteStreamer:
    @overload
    def __init__(self, stream: TextIO, decode_utf8: bool = ...) -> None: ...
    @overload
    def __init__(self, stream: BinaryIO, decode_utf8: Literal[False] = ...) -> None: ...
    def __init__(self, stream: TextIO | BinaryIO, decode_utf8: bool = False) -> None:
        self._stream: IO = stream
        self._stream_started = False
        self._stream_raw = isinstance(stream, io.BufferedIOBase)
        self._enable_decode_utf8 = decode_utf8
        self._utf8_decoder = codecs.getincrementaldecoder("utf-8")()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        self.flush()
        return False  # don't suppress any exceptions

    def write(self, int_byte: int) -> None:
        if self._stream_raw:
            return self._write_as_byte(int_byte)
        if self._enable_decode_utf8:
            return self._write_decoded_utf8(int_byte)
        return self._write_as_int_str(int_byte)

    def _write_as_int_str(self, int_byte: int) -> None:
        if not self._stream_started:
            self._stream.write(str(int_byte))
            self._stream_started = True
        else:
            self._stream.write(" " + str(int_byte))

    def _write_as_byte(self, int_byte: int) -> None:
        byte = int_byte.to_bytes(1, byteorder="big")
        self._stream.write(byte)

    def _write_decoded_utf8(self, int_byte: int) -> None:
        try:
            byte = int_byte.to_bytes(1, byteorder="big")
            decoded_byte = self._utf8_decoder.decode(byte)
            self._stream.write(decoded_byte)

        except UnicodeDecodeError:
            self._utf8_decoder.reset()

    def flush(self):
        if self._utf8_decoder is None:
            return
        try:
            remaining = self._utf8_decoder.decode(b"", final=True)
            if remaining:
                self._stream.write(remaining)
        except UnicodeDecodeError:
            pass
