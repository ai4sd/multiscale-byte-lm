import io

from mblm.utils.stream import ByteStreamer


class TestByteStream:
    UTF_8 = "utf-8"
    UTF_8_TEXT = "こんにちは, just testing a UTF-8 stream🥳!"

    def test_utf8_no_ctx_manager(self):
        buffer = io.StringIO()
        streamer = ByteStreamer(buffer, decode_utf8=True)

        for byte in self.UTF_8_TEXT.encode(self.UTF_8):
            streamer.write(byte)
        streamer.flush()
        buffer.seek(0)
        assert buffer.read() == self.UTF_8_TEXT

    def test_utf8(self):
        buffer = io.StringIO()

        with ByteStreamer(buffer, decode_utf8=True) as streamer:
            for byte in self.UTF_8_TEXT.encode(self.UTF_8):
                streamer.write(byte)

        buffer.seek(0)
        assert buffer.read() == self.UTF_8_TEXT

    def test_utf8_corrputed_recover(self):
        # fmt: off
        _utf_8           = b"h\xc3\xa9llo world"  # héllo world
        utf_8_corrputed  = b"h\xc3\xc9llo world"  # h<corrupted>llo world
        utf_8_result     =  "hllo world"
        # fmt: on

        buffer = io.StringIO()

        with ByteStreamer(buffer, decode_utf8=True) as streamer:
            for byte in utf_8_corrputed:
                streamer.write(byte)

        buffer.seek(0)
        assert buffer.read() == utf_8_result

    def test_int_str(self):
        buffer = io.StringIO()
        text_utf8 = self.UTF_8_TEXT.encode(self.UTF_8)

        with ByteStreamer(buffer, decode_utf8=False) as streamer:
            for byte in text_utf8:
                streamer.write(byte)

        buffer.seek(0)
        str_bytes = list(map(str, self.UTF_8_TEXT.encode(self.UTF_8)))
        assert buffer.read() == " ".join(str_bytes)

    def test_bytes(self):
        buffer = io.BytesIO()
        text_utf8 = self.UTF_8_TEXT.encode(self.UTF_8)

        with ByteStreamer(buffer) as streamer:
            for byte in text_utf8:
                streamer.write(byte)

        buffer.seek(0)
        assert buffer.read() == text_utf8
