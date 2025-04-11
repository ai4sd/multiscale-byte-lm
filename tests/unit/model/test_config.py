import pytest

from mblm import (
    MambaBlock,
    MBLMModelConfig,
    TransformerBlock,
    TransformerEncoderBlock,
)
from mblm.model.block import StageBlock
from mblm.model.config import MaskedMBLMModelConfig

block_transformer = TransformerBlock(
    attn_head_dims=64,
    attn_num_heads=16,
    attn_dropout=0.0,
    ff_multiplier=2,
    ff_dropout=0.1,
    pos_emb_type="fixed",
    attn_use_rot_embs=True,
    use_flash_attn=True,
)
block_encoder = TransformerEncoderBlock(
    attn_head_dims=64,
    attn_num_heads=16,
    attn_dropout=0.0,
    ff_multiplier=2,
    ff_dropout=0.1,
    pos_emb_type="fixed",
    attn_use_rot_embs=True,
    use_flash_attn=True,
)
block_mamba = MambaBlock(d_conv=4, d_state=128, expand=2, headdim=64, pos_emb_type=None)

TEST_BLOCK_CONFIGS: list[StageBlock | list[StageBlock]] = [
    block_mamba,
    block_transformer,
    [block_mamba, block_transformer, block_transformer],
    [block_mamba, block_transformer, block_transformer],
]


class TestConfigSerialize:
    @pytest.mark.parametrize("block", TEST_BLOCK_CONFIGS)
    def test_serialize(self, block: StageBlock | list[StageBlock]):
        config = MBLMModelConfig(
            num_tokens=257,
            hidden_dims=[1024, 1024, 1024],
            seq_lens=[8192, 16, 8],
            pad_token_id=256,
            num_layers=[1, 1, 1],
            train_checkpoint_chunks=[5, 10],
            block=block,
        )

        assert config == MBLMModelConfig.model_validate(config.model_dump())
        assert config == MBLMModelConfig.model_validate_json(config.model_dump_json())


TEST_ENCODER_CONFIGS: list[StageBlock | list[StageBlock]] = [
    [block_encoder, block_encoder],
]


class TestConfigSerializeEncoder:
    @pytest.mark.parametrize("encoder_block", TEST_ENCODER_CONFIGS)
    def test_serialize(self, encoder_block: StageBlock | list[StageBlock]):
        config = MaskedMBLMModelConfig(
            mask_token_id=-100,
            mblm_config=MBLMModelConfig(
                num_tokens=257,
                hidden_dims=[1024, 1024],
                seq_lens=[8192, 16],
                pad_token_id=256,
                num_layers=[1, 1],
                train_checkpoint_chunks=[5, 10],
                block=encoder_block,
            ),
        )
        assert config == MaskedMBLMModelConfig.model_validate(config.model_dump())
        assert config == MaskedMBLMModelConfig.model_validate_json(config.model_dump_json())
