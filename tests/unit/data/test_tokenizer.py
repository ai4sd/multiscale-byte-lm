import pytest
import torch

from mblm.data.utils import Tokenizer
from mblm.data.utils.tokenizer import TokenizerOptions


class TestTokenizer:
    def test_pipeline(self):
        pipeline = Tokenizer(
            TokenizerOptions(
                pad_token_id=10,
                eom_token_id=11,
                som_image_token_id=12,
                som_text_token_id=13,
            )
        ).pipeline
        inp = torch.arange(0, 10, dtype=torch.uint8)  # length 10
        out = pipeline(inp).with_eom().with_som_text().pad_right_to(15).to_long_tensor()

        assert out.dtype == torch.long
        assert out.size(0) == 15
        assert out[0].item() == 13  # som token
        assert out[11].item() == 11  # eom token
        assert out[12:].equal(torch.tensor([10, 10, 10]))  # padding token

    def test_pipeline_none(self):
        pipeline = Tokenizer(
            TokenizerOptions(
                pad_token_id=10,
                eom_token_id=None,
                som_image_token_id=None,
                som_text_token_id=None,
            )
        ).pipeline
        inp = torch.arange(0, 10, dtype=torch.uint8)  # length 10

        assert pipeline(inp).to_long_tensor().dtype == torch.long
        assert pipeline(inp).pad_right_to(15).to_long_tensor().dtype == torch.long

        out = pipeline(inp).with_eom().with_som_text().pad_right_to(15).to_long_tensor()

        assert out.dtype == torch.long
        assert out.size(0) == 15
        assert out[:10].equal(inp)  # no change
        assert out[10:].equal(torch.tensor([10, 10, 10, 10, 10]))  # rest is padding

    def test_pipeline_pad_err(self):
        pipeline = Tokenizer(
            TokenizerOptions(
                pad_token_id=10,
                eom_token_id=None,
                som_image_token_id=None,
                som_text_token_id=None,
            )
        ).pipeline
        inp = torch.arange(0, 10)

        with pytest.raises(ValueError) as exc_info:
            pipeline(inp).pad_right_to(9).to_long_tensor()
        exp_error = "Tensor at dim 0 (length 10) larger than desired padded size 9"
        assert exp_error in str(exc_info.value)

    def test_pipeline_2d_err(self):
        pipeline = Tokenizer(
            TokenizerOptions(
                pad_token_id=10,
                eom_token_id=None,
                som_image_token_id=None,
                som_text_token_id=None,
            )
        ).pipeline
        inp = torch.randn((2, 1))

        with pytest.raises(ValueError) as exc_info:
            pipeline(inp).to_long_tensor()
        exp_error = "Can only process 1D tensors, input is 2D"
        assert exp_error in str(exc_info.value)
