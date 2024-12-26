from io import BytesIO

import pytest
import torch

from mblm.data.utils import Bytes, FileStream


class TestByteUtils:
    @pytest.mark.parametrize(
        "string,expected_bytes",
        [
            ["hello", [104, 101, 108, 108, 111]],
            ["über", [195, 188, 98, 101, 114]],
            ["日本", [230, 151, 165, 230, 156, 172]],
            ["🥰", [240, 159, 165, 176]],
        ],
    )
    def test_string_utils(self, string: str, expected_bytes: list[int]):
        b = Bytes.str_to_bytes(string)
        assert b == bytes(expected_bytes) == bytearray(expected_bytes)
        assert Bytes.bytes_to_str(b) == string

        t = Bytes.str_to_tensor(string)
        assert t.equal(torch.tensor(expected_bytes, dtype=torch.uint8))
        assert Bytes.tensor_to_str(t) == string


class TestFileStream:
    def test_file_stream(self):
        data = bytes([1, 2, 3])
        inp_stream = BytesIO()
        inp_stream.write(data)
        stream = FileStream(inp_stream)
        assert stream.to_buffer() == data
        assert stream.to_numpy().tolist() == stream.to_tensor().tolist()
