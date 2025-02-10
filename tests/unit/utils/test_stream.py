import io

import pytest

from mblm.utils.stream import ByteStreamer

_ENCODING = "utf-8"

UTF8_TEST_TEXT = "こんにちは, just testing a UTF-8 stream🥳!"


class TestByteStream:
    def test_utf8_no_ctx_manager(self):
        """Test decoding valid UTF-8 bytes"""
        buffer = io.StringIO()
        streamer = ByteStreamer(buffer, decode_utf8=True)

        for byte in UTF8_TEST_TEXT.encode(_ENCODING):
            streamer.write(byte)
        streamer.flush()
        buffer.seek(0)
        assert buffer.read() == UTF8_TEST_TEXT

    def test_utf8(self):
        """Test decoding valid UTF-8 bytes"""
        buffer = io.StringIO()

        with ByteStreamer(buffer, decode_utf8=True) as streamer:
            for byte in UTF8_TEST_TEXT.encode(_ENCODING):
                streamer.write(byte)

        buffer.seek(0)
        assert buffer.read() == UTF8_TEST_TEXT

    @pytest.mark.parametrize(
        "input_bytes,result_str",
        (
            (b"h\xc3\xc9llo world", "h\ufffd\ufffdllo world"),
            # corrupted + valid code point
            (b"\xe2\xe2\xe2\xe2\x86\x91", "\ufffd\ufffd\ufffd\u2191"),
        ),
    )
    def test_utf8_corrputed_recover(self, input_bytes: bytes, result_str: str):
        """
        Test recovering from a corrupted UTF-8 byte stream. Our default strategy
        is to replace.

        Mapping of test inputs:
        # Code point for ↑: \u2191 or \xe2\x86\x91
        # Code point for �: \ufffd or \xef\xbf\xbd
        """
        buffer = io.StringIO()

        with ByteStreamer(buffer, decode_utf8=True) as streamer:
            for byte in input_bytes:
                streamer.write(byte)

        buffer.seek(0)
        assert buffer.read() == result_str

    def test_int_str(self):
        """Test decoding bytes as integers to a string"""
        buffer = io.StringIO()
        text_utf8 = UTF8_TEST_TEXT.encode(_ENCODING)

        with ByteStreamer(buffer, decode_utf8=False) as streamer:
            for byte in text_utf8:
                streamer.write(byte)

        buffer.seek(0)
        # [227,129, 147, 227, 130, ... 33]
        str_bytes = list(map(str, UTF8_TEST_TEXT.encode(_ENCODING)))
        assert buffer.read() == " ".join(str_bytes)

    def test_bytes(self):
        """Test writing raw bytes"""
        buffer = io.BytesIO()
        text_utf8 = UTF8_TEST_TEXT.encode(_ENCODING)

        with ByteStreamer(buffer) as streamer:
            for byte in text_utf8:
                streamer.write(byte)

        buffer.seek(0)
        assert buffer.read() == text_utf8
