import math

import pytest
import torch
from MEGABYTE_pytorch import MEGABYTE

from mblm import MBLM, MBLMModelConfig, MBLMReturnType
from mblm.model.transformer import TransformerBlockConfig
from mblm.utils.seed import seed_everything


class TestMegabyte:
    num_tokens = 256 + 1  # for padding id
    pad_token_id = 0
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

    def boostrap_models(
        self,
        model_dims: tuple[int, ...],
        seq_lens: tuple[int, ...],
        use_fixed_pos_encoding: bool = False,
    ):
        num_layers = (1,) * len(model_dims)

        seed_everything(8)
        original_megabyte = MEGABYTE(
            num_tokens=self.num_tokens,
            pad_id=self.pad_token_id,
            dim=model_dims,
            max_seq_len=seq_lens,
            depth=num_layers,
            dim_head=self.dim_attn_heads,
            heads=self.num_attn_heads,
            attn_dropout=self.dropout,
            ff_mult=self.ff_mult,
            ff_dropout=self.dropout,
            rel_pos=self.use_rot_emb,
            pos_emb=use_fixed_pos_encoding,
            flash_attn=self.use_flash_attn,
        )
        # generating random numbers changes the stage of the random number
        # generator
        # - seed again
        seed_everything(8)
        patched_megabyte = MBLM(
            MBLMModelConfig(
                num_tokens=self.num_tokens,
                hidden_dims=model_dims,
                seq_lens=seq_lens,
                pad_token_id=self.pad_token_id,
                num_layers=num_layers,
                train_checkpoint_chunks=None,
                block=TransformerBlockConfig(
                    attn_head_dims=self.dim_attn_heads,
                    attn_num_heads=self.num_attn_heads,
                    attn_dropout=self.dropout,
                    ff_multiplier=self.ff_mult,
                    ff_dropout=self.dropout,
                    pos_emb_type="fixed" if use_fixed_pos_encoding else None,
                    attn_use_rot_embs=self.use_rot_emb,
                    use_flash_attn=self.use_flash_attn,
                ),
            )
        )
        return original_megabyte, patched_megabyte

    def make_input_tensor(self, seq_len: int):
        return torch.randint(1, self.num_tokens, size=(1, seq_len), dtype=torch.long)

    @pytest.mark.parametrize("model_dims,seq_lens", model_fixtures_dims_lens)
    def test_megabyte_models_equal(
        self,
        model_dims: tuple[int, ...],
        seq_lens: tuple[int, ...],
    ):
        original, patched = self.boostrap_models(model_dims, seq_lens)

        input_tensor = self.make_input_tensor(9)
        loss_original = original.forward(input_tensor, return_loss=True)
        loss_patched = patched.forward(input_tensor, return_type=MBLMReturnType.LOSS)
        assert loss_original.isclose(
            loss_patched, atol=0.0001
        ), f"losses ({loss_original}, {loss_patched}) do not match - model dim {model_dims}"
        assert (
            loss_original.dtype == loss_patched.dtype
        ), "loss dtypes do not match - model dim {model_dims}"

        loss_original.backward()
        loss_patched.backward()

        logits_original = original.forward(input_tensor, return_loss=False)
        logits_patched = patched.forward(input_tensor, return_type=MBLMReturnType.LOGITS)
        assert logits_original.equal(logits_patched), f"preds do not match - model dim {model_dims}"
        assert (
            logits_original.dtype == loss_patched.dtype
        ), "pred dtypes do not match - model dim {model_dims}"

    @pytest.mark.parametrize("model_dims,seq_lens", model_fixtures_dims_lens)
    def test_transformer_seq_len_overflow(
        self,
        model_dims: tuple[int, ...],
        seq_lens: tuple[int, ...],
    ):
        original, patched = self.boostrap_models(model_dims, seq_lens, use_fixed_pos_encoding=True)
        input_seq_len = math.prod(seq_lens)  # max sequence length possible
        input_tensor = self.make_input_tensor(input_seq_len)
        input_tensor_too_large = self.make_input_tensor(input_seq_len + 1)

        original.forward(input_tensor)  # should pass
        patched.forward(input_tensor)  # should pass

        with pytest.raises(AssertionError):
            original.forward(input_tensor_too_large)
        with pytest.raises(AssertionError):
            patched.forward(input_tensor_too_large)
