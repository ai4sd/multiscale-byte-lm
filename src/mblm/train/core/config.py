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

"""
This module defines the core attributes that need to be specified for an
experiment. More granular training runs may subclass the core classes and add
additional attributes.
"""

from typing import Generic, Literal, NamedTuple, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class CoreModelParams(BaseModel):
    """
    The core model parameters that all models must specify when training with
    the `CoreTrainer`.
    """

    input_seq_len: int = Field(
        description="The input sequence length. This can be regarded as the context size when training for language. In other scenarios like linear regression, this might be a single data point, hence 1"
    )


class CoreTrainConfig(BaseModel):
    """
    The core training parameters needed for a training run with the
    `CoreTrainer`.
    """

    target_elements: int = Field(
        description="The desired number of data points to train on. If `None`, defaults to using data points from the training set once, resulting in a single epoch. Note that this is a *lower bound* - due to the sequence length and batch sizes, in effect, we train on more than this target. Use this when you want to train on a fixed subset of data, e.g., a number of bytes"
    )
    target_elements_strategy: Literal["batch", "sequence"] = Field(
        description="The strategy to count elements in a batch, used to determine when `target_elements` is achieved. 'batch' means batch size. 'sequence' will count each element in the batch as contributing to the `target_elements`. E.g., when a batch has `n` sequences of length `L`, then the 'sequence' strategy will count `n` * `L` target elements per batch"
    )
    warmup_steps_perc: float = Field(
        default=0.1,
        description="A float in the range [0, 1] to determine how many of the total gradient steps should be used for a warmup. The rest of the steps follows cosine annealing",
    )
    batch_size: int = Field(description="The batch size")

    shuffle_train: bool = Field(
        default=False,
        description="Shuffle the training data in the data loader. Enable this only if you are sure you're not chaining runs",
    )
    shuffle_eval: bool = Field(
        default=False, description="Shuffle the validation data in the data loader"
    )
    max_eval_steps: int | None = Field(
        default=None, description="Maximum number of evaluation iterations"
    )
    learning_rate: float = Field(description="The learning rate")
    gradient_clipping: float | None = Field(
        default=None, description="A value for gradient clipping"
    )
    gradient_accumulate_every: int = Field(
        description="After how many batches to accumulate the gradient"
    )


class ResumeConfig(BaseModel):
    """
    The resume parameters needed for to resume training from a checkpoint and a
    specific epoch and batch with `CoreTrainer`. In case training starts for the
    first time, this class is not needed. For a completed training run, it is
    automatically populated so that it can be used in a future training run.
    """

    checkpoint_file: str = Field(description="Path to model state checkpoint")
    next_epoch_index: int = Field(description="The epoch to resume from")
    next_batch_index: int = Field(description="The global batch counter to resume from")
    migrate_embeddings: bool = Field(
        default=False, description="Migrate an existing smaller number of embeddings"
    )
    rename_modules: bool = Field(
        default=False,
        description="Rename modules from existing models to new names",
    )
    resumed_from: str | None = Field(
        default=None,
        description="If training has been resumed from a previous checkpoint/experiment, points to the config of that experiment to easily trace chained training runs",
    )


class CoreIoConfig(BaseModel):
    """
    The core input/output parameters needed for a training run with the
    `CoreTrainer`.
    """

    name_model: str = Field(description="Model name for saving checkpoints")
    output_dir: str = Field(
        description="The output directory for all artefacts (will be created automatically based on `model_name` and a unique postfix)"
    )
    num_models_to_save: int = Field(
        description="Max number of best performing models to store. If smaller than `validate_amount`, stores only `validate_amount` models"
    )
    validate_amount: int = Field(
        description="How often (in total) to run the validation set and reserve a model candidate. **Must be >= than `num_models_to_save`**"
    )
    log_train_loss_amount: int = Field(description="How often (in total) to log training loss")
    enabled_loss_log_for_gpus: list[int] = Field(
        default=[0], description="The rank of the GPUs that should write to the CSV loss file"
    )


TModelParams = TypeVar("TModelParams", covariant=True, bound=CoreModelParams)
TTrainConfig = TypeVar("TTrainConfig", covariant=True, bound=CoreTrainConfig)
TIoConfig = TypeVar("TIoConfig", covariant=True, bound=CoreIoConfig)


class SummaryStats(BaseModel):
    """
    Summary stats for a training run
    """

    training_start: str
    training_end: str
    num_workers: int
    cuda_devices: list[str]
    parameter_count: int
    error: str | None


class GenericEntryConfig(BaseModel, Generic[TModelParams, TTrainConfig, TIoConfig]):
    """
    A generic entry config for training with the trainer that can be made more
    specific by subclassing.
    """

    model_config = ConfigDict(strict=True)

    params: TModelParams
    train: TTrainConfig
    io: TIoConfig
    resume: ResumeConfig | None = None


class GenericOutputConfig(BaseModel, Generic[TModelParams, TTrainConfig, TIoConfig]):
    """
    A generic output config for a training run with the trainer that can be made
    more specific by subclassing.
    """

    model_config = ConfigDict(strict=True)

    params: TModelParams
    train: TTrainConfig
    io: TIoConfig
    resume: ResumeConfig
    summary: SummaryStats


class CSVLossEntry(NamedTuple):
    gpu_rank: int
    timestamp: str
    elements_seen: int
    kind: str
    epoch: int
    batch: int
    cum_batch: int
    loss: float
    lr: float
    avg_grad: float
    avg_grad_clipped: float


class CSVTimeAndMemSnapshotEntry(NamedTuple):
    cum_batch: int
    num_items: int
    kind: str
    fw_time: float
    bw_time: float | None
    allocated: float
    allocated_max: float
    reserved: float
    reserved_max: float
    total: float
