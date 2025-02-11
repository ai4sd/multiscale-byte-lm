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
from abc import abstractmethod
from typing import Any, Callable, Iterator, Protocol, TypeVar

import torch
from pydantic import Field
from torch.optim import Adam, Optimizer  # type: ignore
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR

from mblm import MBLM, MBLMModelConfig, MBLMReturnType
from mblm.data.dataset.clevr import Clevr
from mblm.data.dataset.pg19 import PG19
from mblm.data.datasets import DistributedDataset
from mblm.data.types import BatchWithLossMask, ModelMode
from mblm.model.embeddings import MBLM_TOKEN_EMB_MIGRATION
from mblm.model.utils import count_params
from mblm.train.core.config import (
    CoreIoConfig,
    CoreModelParams,
    CoreTrainConfig,
    GenericEntryConfig,
    GenericOutputConfig,
)
from mblm.train.core.trainer import CoreTrainer
from mblm.utils.distributed import process_group
from mblm.utils.logging import create_logger, shutdown_log_handlers


class TrainMBLMParams(MBLMModelConfig, CoreModelParams):
    """
    Combine the params required by the MBLM model and the trainer.
    """

    pass


class TrainMBLMIoConfig(CoreIoConfig):
    """
    Custom io settings on top of the core/required parameters
    """

    dataset_dir: str = Field(description="Path to the dataset folder")
    dataset_id: str = Field(description="The unique identifier of the dataset")
    dataset_args: dict[str, Any] | None = Field(
        default=None, description="Optional arguments passed to the dataset"
    )
    description: str | None = Field(default=None, description="A description for this experiment")


class TrainOutputConfig(GenericOutputConfig[TrainMBLMParams, CoreTrainConfig, TrainMBLMIoConfig]):
    """
    A convenience class that can be used directly to parse any output generated
    from training a MBLM.
    """

    pass


class TrainEntryConfig(GenericEntryConfig[TrainMBLMParams, CoreTrainConfig, TrainMBLMIoConfig]):
    """
    Class used to parse all required input configuration for training an MBLM.
    """

    pass


class MBLMTrainerDatasetImpl(Protocol):
    """
    Any dataset object implementing this protocol can be used with the MBLM trainer.
    """

    @staticmethod
    @abstractmethod
    def from_train_entry_config(
        config: TrainEntryConfig, mode: ModelMode, worker_id: int, num_workers: int
    ) -> DistributedDataset[BatchWithLossMask]:
        """
        How to parse a training config to a dataset.
        """
        ...

    @staticmethod
    @abstractmethod
    def supports_test_mode() -> bool:
        """
        Whether or not this dataset supports a test mode. Some datasets might not
        expose the answers in their test set so we cannot evaluate a model on it.
        Override if necessary
        """
        ...


TDataset = TypeVar("TDataset", bound="MBLMTrainerDatasetImpl")


class DatasetRegistry(dict[str, MBLMTrainerDatasetImpl]):
    """
    Dataset registry that allows (custom) datasets to be registered for usage
    with the MBLM trainer. Any dataset used with the MBLM trainer must be
    registered first.
    """

    def register(self, dataset_id: str) -> Callable[[type[TDataset]], type[TDataset]]:
        """
        Decorator to register a dataset that implements the methods required by
        `MBLMTrainerDatasetImpl`.

        Usage:
            @dataset_registry.register("my_dataset")
            class MyDataset(MBLMTrainerDatasetImpl):
                pass
        """

        def decorator(dataset_klass: type[TDataset]) -> type[TDataset]:
            self.update({dataset_id: dataset_klass})
            return dataset_klass

        return decorator

    def retrieve(self, dataset_id: str) -> MBLMTrainerDatasetImpl:
        """
        Retrieve a dataset. Will raise a `KeyError` if the dataset's id cannot
        be found.
        """
        return self[dataset_id]


class MegabyteTrainer(
    CoreTrainer[
        MBLM,
        BatchWithLossMask,
        TrainMBLMParams,
        CoreTrainConfig,
        CoreIoConfig,
    ]
):
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


dataset_registry = DatasetRegistry()

dataset_registry.register("pg19")(PG19)
dataset_registry.register("clevr")(Clevr)


def train_mblm(config: TrainEntryConfig) -> None:
    log = create_logger(__name__, log_dir=config.io.output_dir)

    try:
        with process_group(backend="nccl") as run_vars:
            dataset = dataset_registry.retrieve(config.io.dataset_id)

            train_dataset = dataset.from_train_entry_config(
                config,
                mode=ModelMode.TRAIN,
                worker_id=run_vars.local_rank,
                num_workers=run_vars.world_size,
            )
            valid_dataset = dataset.from_train_entry_config(
                config,
                mode=ModelMode.VALID,
                worker_id=0,
                num_workers=1,
            )

            trainer = MegabyteTrainer(config, run_vars=run_vars)
            best_model = trainer.train(train_dataset, valid_dataset)

            supports_test_mode = dataset.supports_test_mode()
            if best_model and supports_test_mode:
                test_dataset = dataset.from_train_entry_config(
                    config,
                    mode=ModelMode.TEST,
                    worker_id=0,
                    num_workers=1,
                )
                trainer.test(test_dataset, best_model)
    except Exception as error:
        log.fatal(error, exc_info=True)
        shutdown_log_handlers()
