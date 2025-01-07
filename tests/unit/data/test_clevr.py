from pathlib import Path

import pydantic
import pytest
import torch

from mblm.data.dataset.clevr import Clevr, ClevrOptionalArgs
from mblm.data.types import ModelMode
from mblm.data.utils.bytes import Bytes

CLEVR_FIXTURE_PATH = Path("tests/fixtures/clevr")


def import_clevr_from_fixture(pad_token_id: int, optional_args: ClevrOptionalArgs):
    """
    General helper to directly import the Clevr dataset from the fixture
    """
    return Clevr(
        CLEVR_FIXTURE_PATH,
        mode=ModelMode.VALID,
        pad_token_id=pad_token_id,
        optional_args=optional_args,
        seq_len=500_000,
        num_workers=1,
        worker_id=0,
    )


class TestClevrDataset:
    PAD_TOKEN_ID = 1001
    EOM_TOKEN_ID = 1002
    SOM_IMG_TOKEN_ID = 1003
    SOM_TXT_TOKEN_ID = 1004

    def test_clevr_dummy_data(self):
        clevr = import_clevr_from_fixture(
            pad_token_id=self.PAD_TOKEN_ID,
            optional_args=ClevrOptionalArgs(qiqa_loss_mask=(1.0, 1.0, 1.0, 1.0), target_mode="a"),
        )
        # our dummy dataset has just 1 sample
        assert len(clevr) == 1
        s = clevr.get_sample_raw(0)
        question, answer = s["question"], s["answer"]
        assert len(Bytes.str_to_tensor(question)) == clevr.MAX_QUESTION_LEN_BYTES
        assert len(Bytes.str_to_tensor(answer)) == clevr.MAX_ANSWER_LEN_BYTESN

    def test_sample_modality_indices(self):
        clevr = import_clevr_from_fixture(
            pad_token_id=self.PAD_TOKEN_ID,
            optional_args=ClevrOptionalArgs(
                eom_token_id=self.EOM_TOKEN_ID,
                som_text_token_id=self.SOM_TXT_TOKEN_ID,
                som_image_token_id=self.SOM_IMG_TOKEN_ID,
                qiqa_loss_mask=(1.0, 1.0, 1.0, 1.0),
                target_mode="a",
            ),
        )
        sample, _, (question, image, answer, padding) = clevr.get_sample_with_parts(0)
        assert sample.dtype == torch.long

        qiqa_reconstructed = torch.concat([question, image, question, answer])

        assert len(qiqa_reconstructed) + padding == len(sample)
        assert question[0].item() == self.SOM_TXT_TOKEN_ID
        assert answer[0].item() == self.SOM_TXT_TOKEN_ID
        assert image[0].item() == self.SOM_IMG_TOKEN_ID

        for item in (question, image, answer):
            assert item[-1].item() == self.EOM_TOKEN_ID

    @pytest.mark.parametrize("quality", [-1, 96])
    def test_clevr_args_validation(self, quality: int):
        with pytest.raises(pydantic.ValidationError):
            _ = import_clevr_from_fixture(
                pad_token_id=self.PAD_TOKEN_ID,
                optional_args=ClevrOptionalArgs(
                    qiqa_loss_mask=(1.0, 1.0, 1.0, 1.0),
                    target_mode="a",
                    enable_jpeg_stream_with_quality=quality,
                ),
            )
