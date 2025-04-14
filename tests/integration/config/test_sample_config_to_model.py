from pathlib import Path
from typing import Iterable

import pytest

from mblm import MBLM, MambaBlock, MBLMModelConfig, TransformerBlock
from mblm.data.dataset.clevr import ClevrOptionalArgs
from mblm.model.transformer import TransformerEncoderBlock
from mblm.train.mblm import TrainEntryConfig
from mblm.utils.io import load_yml

CONFIG_FILES_DIR = "config"
CONFIG_FILES = [(config,) for config in Path(CONFIG_FILES_DIR).glob("*.yaml")]


class TestConfigToModel:
    def ensure_config_is_valid(self, config: Path):
        try:
            return load_yml(config, parse_to=TrainEntryConfig)
        except Exception:
            pytest.fail(f"Invalid config {config}")

    def ensure_dataset_args_are_valid(self, config: TrainEntryConfig) -> None:
        if config.io.dataset_id == "clevr":
            try:
                ClevrOptionalArgs.model_validate(config.io.dataset_args)
            except Exception:
                pytest.fail(f"Invalid clevr dataset kwargs ({config.io.name_model})")
        return None

    def ensure_model_is_created(self, config: TrainEntryConfig) -> None:
        for b in config.params.stage_blocks():
            assert isinstance(b, (TransformerBlock, MambaBlock, TransformerEncoderBlock))
            if isinstance(b, TransformerBlock):
                assert b.block_type == "transformer"
            elif isinstance(b, TransformerEncoderBlock):
                assert b.block_type == "transformerEncoder"
            else:
                # mamba1, can be mamba2 (only if tested on Linux with mamba_ssm installed)
                assert b.block_type.startswith("mamba")

        _ = MBLM(
            MBLMModelConfig(
                num_tokens=config.params.num_tokens,
                hidden_dims=config.params.hidden_dims,
                seq_lens=config.params.seq_lens,
                num_layers=config.params.num_layers,
                pad_token_id=config.params.pad_token_id,
                train_checkpoint_chunks=config.params.train_checkpoint_chunks,
                block=config.params.block,
            )
        )
        return None

    @pytest.mark.parametrize("config_files", CONFIG_FILES)
    def test_config_to_mbml_transformer(self, config_files: Iterable[Path]):
        for config_file in config_files:
            config = self.ensure_config_is_valid(config_file)
            self.ensure_dataset_args_are_valid(config)
            self.ensure_model_is_created(config)
