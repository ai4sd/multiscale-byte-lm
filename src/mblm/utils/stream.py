__copyright__ = """MIT License

Copyright (c) 2025 - IBM Research

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
    """
    The ByteStreamer allows writing bytes to a given stream with support
    for optional UTF-8 decoding.

    Arguments:
        `stream` (IO): Any file-like object with a `.write()` method
        `decode_utf8` (bool): When `stream` expects a text stream (e.g., TextIO),
            incrementally decode from UTF-8 bytes. Invalid characters will be
            replaced with the Unicode replacement character (U+FFFD). If the stream
            is binary, this option has no effect and bytes are written directly as raw data.
        `ignore_overflowing` (bool): If there is a chance your source writes invalid bytes
            that are larger than 0xFF (256), ignore these. This is enabled by default.

    If the stream expects text and `decode_utf8` is `False`, writes bytes as space-
    separated integer strings.

    It is recommended to use this class as a context manager. If used standalone, its
    `.flush()` method needs to be called after the last write.

    Example:
        ```
        # Using ByteStreamer with a text stream and UTF-8 decoding enabled
        with open("output.txt", "w", encoding="utf-8") as file:
            with ByteStreamer(file, decode_utf8=True) as streamer:
                streamer.write(65)  # Writes 'A' as UTF-8


        ## Using ByteStreamer with a binary stream
        with open("output.bin", "wb") as file:
            with ByteStreamer(file, decode_utf8=False) as streamer:
                streamer.write(65)  # Writes raw byte 0x41 (ASCII 'A')
        ```
    """

    @overload
    def __init__(
        self, stream: TextIO, decode_utf8: bool = ..., ignore_overflowing: bool = ...
    ) -> None: ...
    @overload
    def __init__(
        self, stream: BinaryIO, decode_utf8: Literal[False] = ..., ignore_overflowing: bool = ...
    ) -> None: ...
    def __init__(
        self, stream: IO, decode_utf8: bool = False, ignore_overflowing: bool = True
    ) -> None:
        self._stream: IO = stream
        self._enable_decode_utf8 = decode_utf8
        self.ignore_overflowing = ignore_overflowing
        self._stream_raw = isinstance(stream, io.BufferedIOBase)
        # replace invalid bytes with replacement character u+fffd
        self._utf8_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._stream_started = False

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
        # only relevant when potentially invalid int bytes are fed
        if int_byte > 255 and self.ignore_overflowing:
            return
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
            return

    def flush(self):
        if not self._enable_decode_utf8:
            return
        if remaining := self._utf8_decoder.decode(b"", final=True):
            self._stream.write(remaining)
