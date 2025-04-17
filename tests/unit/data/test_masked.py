from pathlib import Path

import pytest
import torch

from mblm.data.dataset.pg19_masked import PG19Masked
from mblm.data.types import ModelMode

PG_19_MASKED_DIR = Path("tests/fixtures/pg19masked")


def fixture_pg19masked(pad_token_id: int, mode: ModelMode = ModelMode.TEST, mask_id: int = -100):
    """
    General helper to directly import the Clevr dataset from the fixture
    """
    return PG19Masked(
        data_dir=PG_19_MASKED_DIR,
        masked_token_id=mask_id,
        mode=mode,
        padding_token_id=pad_token_id,
        seq_len=10_000,
        num_workers=1,
        worker_id=0,
    )


class Testpg19MaskedDataset:
    MASKED_TOKEN_ID = -100

    def test_dummy(self):
        pad_tk = -300
        ds = fixture_pg19masked(pad_token_id=pad_tk)
        assert len(ds) > 0
        masked_tk, mask, labels = ds[0]
        assert masked_tk.dtype == torch.long
        assert mask.dtype == torch.bool
        assert labels.dtype == torch.long
        assert masked_tk.size() == mask.size() == labels.size()
        # no masked token in the labels
        assert not (labels == self.MASKED_TOKEN_ID).any()

    @pytest.mark.parametrize("padd", [-200, -101])
    def test_padding(self, padd):
        ds = fixture_pg19masked(pad_token_id=padd)
        # Last sample should be padded right
        masked_tk, mask, labels = ds[len(ds)]
        assert (masked_tk == padd).any().item()
        # Last component of the tensor is padding (in the last sample)
        assert (masked_tk[-1] == padd).item()
        # no masked token in the labels
        assert not (labels == self.MASKED_TOKEN_ID).any()
        # No padding in the mask,
        assert not (mask == padd).any()
        # No padding except the last element
        for i in range(len(ds)):
            masked_tk, mask, _ = ds[len(ds)]

    @pytest.mark.parametrize("padd", [-200, -101])
    def test_padding_is_never_masked(self, padd):
        try:
            ds = fixture_pg19masked(pad_token_id=-100)
            raise ValueError("Mask and padding are same, it should fail")
        except Exception as e:
            assert e is not None
        ds = fixture_pg19masked(pad_token_id=padd)
        padded_tks, padded_mask, padded_labels = ds[len(ds)]
        assert padded_mask[-1].item() == 0
        assert padded_tks[-1] == padd

    @pytest.mark.parametrize("book_id", ["10.txt"])
    def test_boook(self, book_id):
        ds = fixture_pg19masked(pad_token_id=-500, mode=ModelMode.TRAIN)

        # Last sample should be padded right
        book_content = ds.book(book_id)
        assert isinstance(book_content, str)

    @pytest.mark.xfail(raises=ValueError)
    @pytest.mark.parametrize("padding", [0, 100])
    @pytest.mark.parametrize("mask", [0, 100])
    def test_init(self, padding, mask):
        fixture_pg19masked(pad_token_id=padding, mask_id=mask)
