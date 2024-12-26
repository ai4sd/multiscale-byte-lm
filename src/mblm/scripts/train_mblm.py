__copyright__ = """MIT License

Copyright (c) 2024 - IBM Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

import math
import os
from pathlib import Path
from typing import Any, Iterator, Literal

import torch
from torch.distributed.elastic.multiprocessing.errors import record
from torch.optim import Adam, Optimizer  # type: ignore
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR

from mblm import MBLM, MBLMModelConfig, MBLMReturnType
from mblm.data.dataset.clevr import Clevr, ClevrOptionalArgs
from mblm.data.dataset.pg19 import PG19
from mblm.data.datasets import BatchWithLossMask
from mblm.data.types import ModelMode
from mblm.model.embeddings import MBLM_TOKEN_EMB_MIGRATION
from mblm.trainer.config import (
    CoreIoConfig,
    CoreModelParams,
    CoreTrainConfig,
    GenericEntryConfig,
    GenericOutputConfig,
)
from mblm.trainer.core import CoreTrainer
from mblm.utils.distributed import process_group
from mblm.utils.io import load_yml
from mblm.utils.logging import create_logger, shutdown_log_handlers
from mblm.utils.misc import count_params


class IoConfig(CoreIoConfig):
    """
    Custom io settings on top of the core/required parameters
    """

    dataset_dir: str
    dataset_id: Literal["pg19", "clevr"]
    dataset_args: dict[str, Any] | None = None
    description: str | None = None


class ModelParams(MBLMModelConfig, CoreModelParams):
    """
    Combine the params required by the MBLM model and the trainer.
    """

    pass


class TrainOutputConfig(GenericOutputConfig[ModelParams, CoreTrainConfig, IoConfig]):
    """
    A class that can be used directly to parse any output generated from
    training with the MBLM model.
    """

    pass


class TrainEntryConfig(GenericEntryConfig[ModelParams, CoreTrainConfig, IoConfig]):
    def import_dataset(self, mode: ModelMode, worker_id: int, num_workers: int):
        if self.io.dataset_id == "clevr":
            # cannot pass None to model_validate
            optional_args = ClevrOptionalArgs.model_validate(self.io.dataset_args or dict())

            return Clevr(
                data_dir=self.io.dataset_dir,
                mode=mode,
                pad_token_id=self.params.pad_token_id,
                seq_len=self.params.input_seq_len,
                worker_id=worker_id,
                num_workers=num_workers,
                optional_args=optional_args,
            )

        return PG19(
            data_dir=self.io.dataset_dir,
            mode=mode,
            seq_len=self.params.input_seq_len,
            worker_id=worker_id,
            num_workers=num_workers,
        )


class MegabyteTrainer(CoreTrainer[MBLM, BatchWithLossMask, ModelParams, CoreTrainConfig, IoConfig]):
    def init_model(self):
        return MBLM(
            MBLMModelConfig(
                # number of tokens
                num_tokens=self.config.params.num_tokens,
                # transformer model dimension (global, local)
                hidden_dims=tuple(self.config.params.hidden_dims),
                # sequence length (global, local)
                seq_lens=tuple(self.config.params.seq_lens),
                pad_token_id=self.config.params.pad_token_id,
                num_layers=tuple(self.config.params.num_layers),
                train_checkpoint_chunks=self.config.params.train_checkpoint_chunks,
                block=self.config.params.block,
            )
        )

    def model_forward(self, model, batch, device) -> torch.Tensor:
        inputs, loss_mask = batch
        inputs = inputs.to(device)
        loss_mask = loss_mask.to(device)
        loss: torch.Tensor = model.forward(
            inputs, return_type=MBLMReturnType.LOSS, loss_mask=loss_mask
        )
        return loss

    def configure_optimizer(self, parameters: Iterator[torch.nn.Parameter]) -> Optimizer:
        return Adam(
            parameters,
            lr=self.config.train.learning_rate,
            betas=(0.9, 0.95),
        )

    def configure_scheduler(self, optimizer, local_gradient_steps) -> LRScheduler:
        warmup_steps = math.floor(local_gradient_steps * self.config.train.warmup_steps_perc)
        linear = LinearLR(
            optimizer,
            total_iters=warmup_steps,
            start_factor=0.1,
            end_factor=1,
        )
        cosine_iters = local_gradient_steps - warmup_steps
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_iters)
        return SequentialLR(
            optimizer,
            [linear, cosine],
            milestones=[warmup_steps],
        )

    def configure_run_id(self) -> str:
        return os.getenv("JOB_ID") or super().configure_run_id()

    def configure_count_parameters(self, model):
        return count_params(model)

    def migrate_embeddings_if_enabled(self):
        # older versions - the pg19 pretrained models - of mblm may have been trained
        # without modality tokens - provide this map to migrate the embeddings.
        # enabled via yaml config
        return MBLM_TOKEN_EMB_MIGRATION

    def rename_modules_if_enabled(self):
        # we have renamed pos_embs to patch_pos_embs in newer versions. enabled
        # via yaml config
        return (("pos_embs", "patch_pos_embs"),)


@record
def main(config: TrainEntryConfig) -> None:
    log = create_logger(__name__, log_dir=config.io.output_dir)

    try:
        with process_group(backend="nccl") as run_vars:
            train_dataset = config.import_dataset(
                mode=ModelMode.TRAIN,
                worker_id=run_vars.local_rank,
                num_workers=run_vars.world_size,
            )
            valid_dataset = config.import_dataset(
                mode=ModelMode.VALID,
                worker_id=0,
                num_workers=1,
            )

            trainer = MegabyteTrainer(config, run_vars=run_vars)
            best_model = trainer.train(train_dataset, valid_dataset)

            supports_test_mode = train_dataset.supports_test_mode()
            if best_model and supports_test_mode:
                test_dataset = config.import_dataset(
                    mode=ModelMode.TEST,
                    worker_id=0,
                    num_workers=1,
                )
                trainer.test(test_dataset, best_model)
    except Exception as error:
        log.fatal(error, exc_info=True)
        shutdown_log_handlers()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        dest="config_path",
        required=True,
        type=Path,
        help="Path to the experiment yaml config file",
    )

    args = parser.parse_args()
    train_cfg = load_yml(args.config_path, parse_to=TrainEntryConfig)
    main(train_cfg)
