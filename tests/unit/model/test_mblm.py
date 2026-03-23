import io
import math
from functools import partial

import pytest
import torch

from mblm import MBLM, MBLMModelConfig, MBLMReturnType, TransformerBlock
from mblm.model.config import MBLMEncoderModelConfig
from mblm.model.mblm import MBLMEncoder
from mblm.model.transformer import TransformerEncoderBlock
from mblm.utils.seed import seed_everything
from mblm.utils.stream import ByteStreamer
from typing import List

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

        seed_everything(8)
        mblm = MBLM(
            MBLMModelConfig(
                num_tokens=self.num_tokens,
                hidden_dims=[256, 256],
                seq_lens=ctx_windows,
                pad_token_id=256,
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
            generate(stream=stream, filter_thres=1)

        assert len(buff.getbuffer()) == total_generation_len


class TestMaskedMBLM:
    # padding + mask token
    num_tokens = 256 + 2
    mask_token_id = 257
    pad_token_id = 256
    num_attn_heads = 16
    dim_attn_heads = 64
    ff_mult = 4
    dropout = 0
    use_rot_emb = True
    use_flash_attn = False
    mblm_conf = MBLMModelConfig(
        num_tokens=300,
        pad_token_id=299,
        hidden_dims=[1024, 32],
        seq_lens=[512, 128],
        num_layers=[5, 1],
        train_checkpoint_chunks=None,
        block=[
            TransformerEncoderBlock(
                attn_head_dims=64,
                attn_num_heads=16,
                attn_use_rot_embs=True,
                use_flash_attn=True,
                pos_emb_type="fixed",
            ),
            TransformerEncoderBlock(
                attn_head_dims=16,
                attn_num_heads=4,
                attn_use_rot_embs=True,
                use_flash_attn=True,
                pos_emb_type="fixed",
            ),
        ],
    )

    def _extend_or_shrink(self,values:List,wanted_size:int):
        return (values + [values[-1]] * max(0, wanted_size - len(values)))[:wanted_size]
    @pytest.mark.parametrize(
        "batch_size, seq_lens, stage_idx",
        [(2,(63,),0), (12,(4,3),0), (1,(12, 3),1), (2,(5, 4),1), (1,(3,4,5),2),(1,(6,3,1),2),(2,(4,9,17),0)]
    )
    def test_mblm_compute_mask_at_stage_0(self,seq_lens,batch_size,stage_idx):

        conf = self.mblm_conf.model_copy()
        conf.seq_lens = seq_lens
        if len(seq_lens) != 2:
            conf.hidden_dims = self._extend_or_shrink(conf.hidden_dims, len(seq_lens))
            conf.num_layers = self._extend_or_shrink(conf.hidden_dims, len(seq_lens))
            conf.block = self._extend_or_shrink(conf.block, len(seq_lens))

        mask = torch.ones((batch_size,*seq_lens)).to(torch.bool)

        out = MBLMEncoder.compute_mask_at_stage(mask,stage_idx)
        # mask is BxL
        assert out.ndim == 2
        # mask is BxL
        assert out.size(-1) == seq_lens[stage_idx]
        # # The batch size is the prod of the batch size and the previous stage sequence length, +1 is for the initial
        # batch size
        assert out.size(0) == torch.prod(torch.tensor([batch_size, *seq_lens][:stage_idx+1]))


    def test_masked_mblm_fully_masked_returns_0_as_loss(
        self,
    ):
        conf = self.mblm_conf
        conf.seq_lens = [12,4]
        masked_model = MBLMEncoder(
            MBLMEncoderModelConfig(mask_token_id=self.mask_token_id, mblm_config=conf)
        )
        input_len = int(torch.prod(torch.tensor(conf.seq_lens)).item())
        input_ids = torch.randint(0, self.num_tokens, size=(1, input_len), dtype=torch.long)
        masked_input = input_ids.clone()
        mask = torch.zeros_like(input_ids)
        mask = mask.to(torch.bool)
        masked_input[mask] = self.mask_token_id
        loss = masked_model.forward(
            masked_input, mask=mask, labels=input_ids, return_type=MBLMReturnType.LOSS
        )
        
        assert torch.isclose(loss,torch.tensor(0)), f"Got {loss.item()}"

    def test_masked_mblm_partially_masked_is_float(
        self,
    ):
        masked_model = MBLMEncoder(
            MBLMEncoderModelConfig(mask_token_id=self.mask_token_id, mblm_config=self.mblm_conf)
        )
        input_len = int(torch.prod(torch.tensor(self.mblm_conf.seq_lens)).item())
        input_ids = torch.randint(0, self.num_tokens, size=(1, input_len), dtype=torch.long)
        masked_input = input_ids.clone()
        mask = torch.rand_like(input_ids, dtype=torch.float) < 0.15
        mask = mask.to(torch.bool)
        masked_input[mask] = self.mask_token_id
        loss = masked_model.forward(
            masked_input, mask=mask, labels=input_ids, return_type=MBLMReturnType.LOSS
        )
        assert loss.dtype == torch.float and loss.item() > 0.0

    @pytest.mark.parametrize("batch", [1, 3])
    @torch.no_grad()
    def test_masked_mblm_return_type_shape(self, batch):
        masked_model = MBLMEncoder(
            MBLMEncoderModelConfig(mask_token_id=self.mask_token_id, mblm_config=self.mblm_conf)
        )
        masked_model.eval()
        input_len = int(torch.prod(torch.tensor(self.mblm_conf.seq_lens)).item())
        input_ids = torch.randint(0, self.num_tokens, size=(batch, input_len), dtype=torch.long)
        masked_input = input_ids.clone()
        mask = torch.rand_like(input_ids, dtype=torch.float) < 0.15
        mask = mask.to(torch.bool)
        masked_input[mask] = self.mask_token_id
        loss, logit = masked_model.forward(
            masked_input, mask=mask, labels=input_ids, return_type=MBLMReturnType.LOSS_LOGITS
        )
        hidden_state = masked_model.forward(
            masked_input, mask=mask, labels=input_ids, return_type=MBLMReturnType.HIDDEN_STATE
        )
        assert loss.size() == torch.Size([])
        assert logit.size() == torch.Size([batch, input_len, self.mblm_conf.num_tokens])
        assert hidden_state.size() == torch.Size([batch, input_len, self.mblm_conf.hidden_dims[-1]])

    @torch.no_grad()
    @pytest.mark.parametrize("batch", [1, 3])
    def test_masked_mblm_combined_return(self, batch):
        masked_model = MBLMEncoder(
            MBLMEncoderModelConfig(mask_token_id=self.mask_token_id, mblm_config=self.mblm_conf)
        )
        input_len = int(torch.prod(torch.tensor(self.mblm_conf.seq_lens)).item())
        input_ids = torch.randint(0, self.num_tokens, size=(batch, input_len), dtype=torch.long)
        masked_input = input_ids.clone()
        mask = torch.rand_like(input_ids, dtype=torch.float) < 0.15
        mask = mask.to(torch.bool)
        masked_input[mask] = self.mask_token_id
        loss, logits = masked_model.forward(
            masked_input, mask=mask, labels=input_ids, return_type=MBLMReturnType.LOSS_LOGITS
        )
        loss_only = masked_model.forward(
            masked_input, mask=mask, labels=input_ids, return_type=MBLMReturnType.LOSS
        )
        logits_only = masked_model.forward(
            masked_input, mask=mask, labels=input_ids, return_type=MBLMReturnType.LOGITS
        )
        assert logits.shape == logits_only.shape
        assert loss.shape == loss_only.shape
        assert torch.isclose(loss, loss_only), f"{loss.item()} is not close to {loss_only.item()}"
        assert torch.all(logits == logits_only), f"{logits} does not equal  {logits_only}"

    @torch.no_grad()
    @pytest.mark.parametrize("batch", [1, 3])
    def test_masked_mblm_input_seq_len(self, batch):
        max_input_length = int(torch.prod(torch.tensor(self.mblm_conf.seq_lens)).item())
        for current_input_len in [
            max_input_length // 2,
            max_input_length // 4,
            max_input_length + 1,
        ]:
            masked_model = MBLMEncoder(
                MBLMEncoderModelConfig(mask_token_id=self.mask_token_id, mblm_config=self.mblm_conf)
            )
            input_ids = torch.randint(
                0, self.num_tokens, size=(batch, current_input_len), dtype=torch.long
            )
            masked_input = input_ids.clone()
            mask = torch.rand_like(input_ids, dtype=torch.float) < 0.15
            mask = mask.to(torch.bool)
            masked_input[mask] = self.mask_token_id
            # Each time the sequence length is too big we fail
            if current_input_len > max_input_length:
                with pytest.raises(AssertionError):
                    masked_model.forward(
                        masked_input,
                        mask=mask,
                        labels=input_ids,
                        return_type=MBLMReturnType.LOSS_LOGITS,
                    )
            else:
                loss, logits = masked_model.forward(
                    masked_input,
                    mask=mask,
                    labels=input_ids,
                    return_type=MBLMReturnType.LOSS_LOGITS,
                )
                loss_only = masked_model.forward(
                    masked_input, mask=mask, labels=input_ids, return_type=MBLMReturnType.LOSS
                )
                logits_only = masked_model.forward(
                    masked_input, mask=mask, labels=input_ids, return_type=MBLMReturnType.LOGITS
                )
            assert logits.shape == logits_only.shape
            assert loss.shape == loss_only.shape
            assert torch.isclose(
                loss, loss_only
            ), f"{loss.item()} is not close to {loss_only.item()}"
            assert torch.all(logits == logits_only), f"{logits} does not equal  {logits_only}"
