import io
import math
from functools import partial

import pytest
import torch

from mblm import MBLM, MBLMModelConfig, MBLMReturnType, TransformerBlock
from mblm.utils.stream import ByteStreamer


class TestMBLM:
    num_tokens = 256 + 1
    pad_token_id = 256
    num_attn_heads = 16
    dim_attn_heads = 64
    ff_mult = 4
    dropout = 0
    use_rot_emb = True
    use_flash_attn = False
    model_fixtures_dims_lens: list[tuple[tuple[int, ...], tuple[int, ...]]] = [
        ((1024, 768, 512), (9, 7, 5)),
        ((1024, 1024), (9, 7)),
        ((1024,), (9,)),
    ]

    @pytest.mark.parametrize("model_dims,seq_lens", model_fixtures_dims_lens)
    def test_masked_loss(
        self,
        model_dims: tuple[int, ...],
        seq_lens: tuple[int, ...],
    ):
        mblm = MBLM(
            MBLMModelConfig(
                num_tokens=self.num_tokens,
                hidden_dims=model_dims,
                seq_lens=seq_lens,
                pad_token_id=self.pad_token_id,
                num_layers=(1,) * len(model_dims),
                train_checkpoint_chunks=None,
                block=TransformerBlock(
                    attn_head_dims=self.dim_attn_heads,
                    attn_num_heads=self.num_attn_heads,
                    attn_dropout=self.dropout,
                    ff_multiplier=self.ff_mult,
                    ff_dropout=self.dropout,
                    pos_emb_type="fixed",
                    attn_use_rot_embs=self.use_rot_emb,
                    use_flash_attn=self.use_flash_attn,
                ),
            )
        )
        input_len = 9
        input_ids = torch.randint(0, self.num_tokens, size=(1, input_len), dtype=torch.long)
        loss = mblm.forward(input_ids, return_type=MBLMReturnType.LOSS)
        loss_with_identity_mask = mblm.forward(
            input_ids,
            loss_mask=torch.ones_like(input_ids),
            return_type=MBLMReturnType.LOSS,
        )
        assert torch.equal(loss, loss_with_identity_mask)
        empty_loss = mblm.forward(
            input_ids,
            loss_mask=torch.zeros_like(input_ids),
            return_type=MBLMReturnType.LOSS,
        )
        assert empty_loss.item() == 0.0

    def test_generate(self):
        ctx_windows = [12, 4]
        mblm = MBLM(
            MBLMModelConfig(
                num_tokens=self.num_tokens,
                hidden_dims=[256, 256],
                seq_lens=ctx_windows,
                # for testing, make sure we don't generate byte literals > 255
                pad_token_id=0,
                num_layers=[1, 1],
                train_checkpoint_chunks=None,
                block=[
                    TransformerBlock(
                        attn_head_dims=self.dim_attn_heads,
                        attn_num_heads=self.num_attn_heads,
                        attn_dropout=self.dropout,
                        ff_multiplier=self.ff_mult,
                        ff_dropout=self.dropout,
                        pos_emb_type="fixed",
                        attn_use_rot_embs=self.use_rot_emb,
                        use_flash_attn=self.use_flash_attn,
                    )
                ]
                * 2,
            )
        )
        total_generation_len = math.prod(ctx_windows)
        generate = partial(mblm.generate, enable_progress=False)

        assert generate().size(0) == total_generation_len
        assert generate(torch.ones(10).long()).size(0) == total_generation_len
        assert generate(num_tokens_to_generate=5).size(0) == 5

        # TODO: What if the prompt is longer than the ctx window?

        buff = io.BytesIO()
        with ByteStreamer(buff) as stream:
            generate(stream=stream)

        assert len(buff.getbuffer()) == total_generation_len
