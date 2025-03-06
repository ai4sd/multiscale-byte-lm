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

import logging
import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any, Generic, Iterable, Iterator, Literal, Sequence, TypeVar, cast

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer  # type: ignore
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from mblm.data.datasets import DistributedDataset
from mblm.data.types import ModelMode
from mblm.model.utils import count_params
from mblm.train.core.config import (
    CoreIoConfig,
    CSVLossEntry,
    CSVTimeAndMemSnapshotEntry,
    GenericEntryConfig,
    GenericOutputConfig,
    ResumeConfig,
    SummaryStats,
    TIoConfig,
    TModelParams,
    TTrainConfig,
)
from mblm.train.core.iter import epoch_cycler
from mblm.utils.cuda import cuda_memory_snapshot, cuda_properties
from mblm.utils.distributed import ElasticRunVars
from mblm.utils.io import CSVWriter, StateDict, dump_yml, load_model_state, save_model_state
from mblm.utils.logging import create_logger
from mblm.utils.misc import retry
from mblm.utils.top_n import TopN

TModel = TypeVar("TModel", bound=torch.nn.Module)
TBatch = TypeVar("TBatch", bound=torch.Tensor | Sequence[torch.Tensor])


@dataclass
class CoreTrainerOptions:
    config_file_name: str = "config.yaml"
    loss_file_name: str = "loss.csv"
    timemem_file_name: str = "timemem.csv"
    max_train_restarts: int = 0
    skip_validation: bool = False
    display_progress: bool = sys.stdout.isatty()
    train_prog_min_interval_seconds: int = 1
    valid_prog_min_interval_seconds: int = 1
    track_first_fw_bw_exec_times: int | None = 30  # for 30 first passes, track fw/bw time
    amp_dtype: torch.dtype = torch.half  # may use bfloat16


class CoreTrainer(ABC, Generic[TModel, TBatch, TModelParams, TTrainConfig, TIoConfig]):
    """
    An abstract core trainer that provides a set of utility methods for
    training, evaluating and testing. It is held generic to enforce type-safety
    when implementing the abstract methods.

    - All methods that may or may not be implemented are `@classmethod`

    - Methods that _must_ be implemented are concerned with creating the
    model, which is the job of the instantiator. They are abstract and
    type-checkers complain if they are not implemented correctly

    - Methods that _may_ be implemented, i.e., overwritten, have the prefix
    `with_`. For example, for the optimizer, an `Adam` optimizer with sensible
    defaults is provided but it can be overwritten

    """

    # public config
    config: GenericEntryConfig[TModelParams, TTrainConfig, TIoConfig]

    # overridable options with sensible defaults
    options: CoreTrainerOptions

    # private var
    _local_rank: int
    _world_size: int
    _is_main_worker: bool
    _device: str
    _device_type: Literal["cuda", "cpu"]
    _is_cuda: bool

    _model_dist: DistributedDataParallel

    # misc - created internally
    _output_dir: Path
    _running_resume_conf: ResumeConfig
    _running_summary_stats: SummaryStats
    _top_n_models: TopN[StateDict]
    _csv_loss_writer: CSVWriter[CSVLossEntry]
    _csv_timemem_writer: CSVWriter[CSVTimeAndMemSnapshotEntry]
    _log: logging.Logger

    def __init__(
        self,
        config: GenericEntryConfig[TModelParams, TTrainConfig, TIoConfig],
        run_vars: ElasticRunVars,
        options: CoreTrainerOptions | None = None,
    ):
        self.config = config
        self.options = options or CoreTrainerOptions()
        self._world_size = run_vars.world_size
        self._local_rank = run_vars.local_rank
        # used for sending tensors/models to a device
        self._device = f"cuda:{self._local_rank}" if run_vars.is_cuda else "cpu"
        # used for mixed-precision
        self._device_type = "cuda" if run_vars.is_cuda else "cpu"
        self._is_cuda = not self._device == "cpu"
        self._is_main_worker = run_vars.local_rank == 0

        self._output_dir = self._create_output_dir(config.io)

        self._log = self.configure_logger(self._output_dir, self._is_main_worker)
        self._top_n_models = TopN(
            config.io.num_models_to_save,
            deep_copy=True,  # module state_dicts are references
        )

        # the ranks of the gpus that should write to the csv loss file
        gpu_rank_csv_loss = set(self.config.io.enabled_loss_log_for_gpus)
        self._csv_loss_writer = CSVWriter(
            self._output_dir,
            self.options.loss_file_name,
            noop=self._local_rank not in gpu_rank_csv_loss,
        )
        self._csv_timemem_writer = CSVWriter(
            self._output_dir, self.options.timemem_file_name, noop=not self._is_main_worker
        )

        assert config.io.validate_amount > 0, "Validate amount must be strictly positive"
        assert config.io.num_models_to_save > 0, "Must save at least 1 model"

        if config.io.validate_amount < config.io.num_models_to_save:
            self._log.warning(
                f"Validate amount ({config.io.validate_amount}) \
                is less than number of models to save ({config.io.num_models_to_save}).\
                Saving only {config.io.validate_amount} models"
            )

        model = self.init_model().to(self._device)
        if config.resume:
            self._log.info("Initiating model loading from checkpoint")
            map_extend_embeddings = (
                self.migrate_embeddings_if_enabled() if config.resume.migrate_embeddings else None
            )
            map_rename_modules = (
                self.rename_modules_if_enabled() if config.resume.rename_modules else None
            )
            model, model_loss = load_model_state(
                config.resume.checkpoint_file,
                model,
                map_location=self._device,
                map_extend_embeddings=map_extend_embeddings,
                map_rename_modules=map_rename_modules,
                on_success=self._log.debug,
            )
            # save the restored model to the top n to make sure we keep the best
            # model from the previous run should we only make everything worse
            # during this training
            self._top_n_models.add((model_loss, model.state_dict()))
            self._log.info(f"Loaded model with loss {model_loss:.4f} from checkpoint")
        else:
            self._log.info("Creating new model")

        self._model_dist = self._init_distributed_model(model)

        # initialize the running resume/summary configs that are updated on the fly
        self._running_resume_conf = ResumeConfig(
            checkpoint_file="",
            next_batch_index=-1,
            next_epoch_index=-1,
            resumed_from=config.resume.checkpoint_file if config.resume else None,
        )
        cuda_info = cuda_properties()
        main_model_params, submodule_params = self.configure_count_parameters(model)

        self._running_summary_stats = SummaryStats(
            parameter_count=main_model_params,
            num_workers=run_vars.world_size,
            cuda_devices=cuda_info.cuda_devices,
            training_start="",  # temporary, updated when training starts
            training_end="",  # temporary, updated when training ends
            error=None,  # temporary, updated when training fails
        )
        self._dump_output_config()
        self._log.info("Trainer initialized successfully")
        self._log.info(f"Model parameters: {main_model_params}, ({submodule_params})")
        self._log.info(f"Configuration: {config}")
        self._log.info(f"CUDA: {cuda_info}")

    """ Abstract methods that must be implemented """

    @abstractmethod
    def init_model(self) -> TModel:
        """
        Initialize a model of the specified type `TModel`.
        """
        ...

    @abstractmethod
    def model_forward(
        self,
        model: TModel,
        batch: TBatch,
        device: str,
    ) -> torch.Tensor:
        """
        A single forward pass of the model. Both the model and batch are
        generic, their types are inferred according to the type instantiation
        defined when subclassing `CoreTrainer`.

        Args:
            model (TModel): The model (already on the device)
            batch (TBatch): One batch (MUST be put to device)
            device (str): `cuda:n` for the `n`-th GPU or `cpu`

        Returns:
            torch.Tensor: A Tensor with a single element that is the loss
            of the forward pass

        **Example**::

            # here, batch is a tuple of data, target
            @classmethod
            def model_forward(cls, model, batch, device):
                x, y = batch
                output = model.forward(x.to(device))
                loss_function = torch.nn.MSELoss()
                loss: torch.Tensor = loss_function(output, y)
                return loss


            # in other scenarios, batch might be a single Tensor
            @classmethod
            def model_forward(cls, model, batch, device):
                batch = batch.to(device).long()
                loss: torch.Tensor = model.forward(batch, return_loss=True)
                return loss
        """

        ...

    @abstractmethod
    def configure_optimizer(self, parameters: Iterator[torch.nn.Parameter]) -> Optimizer:
        """
        Configure an optimizer
        """
        ...

    """ Default methods that can be overwritten """

    def configure_scheduler(self, optimizer: Optimizer, local_gradient_steps: int) -> LRScheduler:
        """
        Configure a LR scheduler.

        Args:
            optimizer (Optimizer): The optimizer
            local_gradient_steps (int): The total number of gradient steps for this GPU.
        """
        return torch.optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=local_gradient_steps,
            power=1.0,
        )

    def configure_logger(self, output_dir: Path, is_main_worker: bool) -> logging.Logger:
        """
        Customize the logger
        """
        return create_logger(
            name="train",
            log_dir=output_dir,
            # all non-main workers are noop loggers
            noop=not is_main_worker,
        )

    def configure_count_parameters(self, model: TModel) -> tuple[int, dict[str, int]]:
        """
        Determine how to count parameters for this model
        """
        return count_params(model)

    def configure_run_id(self) -> str:
        """
        Set a unique identifier for this experiment. Used as postfix for the
        output directory.
        """
        return f"{time():.0f}"

    def migrate_embeddings_if_enabled(self) -> set[str] | None:
        """
        When resuming training from a model, a smaller number of embeddings can
        be migrated to a larger number. This can be enabled via the resume
        config.
        """
        return None

    def rename_modules_if_enabled(self) -> Iterable[tuple[str, str]] | None:  # noqa: ARG003
        """
        When resuming training from a model where modules are named differently,
        provide a map in the form (source_prefix, target_prefix) to override
        module names.
        """
        return None

    """ Utility functions  """

    def _init_distributed_model(self, base_model: TModel) -> DistributedDataParallel:
        """
        Create a distributed version of the model.
        """

        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
        # for multi-device modules and CPU modules, device_ids must be None
        device_ids = [self._local_rank] if self._is_cuda else None
        return DistributedDataParallel(model, device_ids=device_ids)

    def _unpack_distributed_model(self, module: TModel | DistributedDataParallel) -> TModel:
        if isinstance(module, DistributedDataParallel):
            return module.module
        return module

    def _create_output_dir(self, io_config: CoreIoConfig) -> Path:
        """
        Create a unique output directory (only on main worker).
        """
        run_id = self.configure_run_id()
        output_dir = Path(io_config.output_dir) / f"{io_config.name_model}_{run_id}"

        # only the main worker should do i/o
        if self._is_main_worker:
            output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir

    def _dump_output_config(self):
        """
        Dump all config files to disk (only on main worker).
        """
        if not self._is_main_worker:
            return
        # copy the config over into the output format
        output_config = GenericOutputConfig(
            io=self.config.io,
            params=self.config.params,
            train=self.config.train,
            resume=self._running_resume_conf,
            summary=self._running_summary_stats,
        )
        dump_yml(self._output_dir / self.options.config_file_name, output_config)

    def _write_csv_loss(
        self,
        kind: ModelMode,
        loss: float,
        epoch: int,
        batch: int,
        cum_batch: int,
        elements_seen: int,
        lr: float,
        avg_grad: float,
        avg_grad_clipped: float,
    ) -> None:
        # no need to check for main worker - the writer has been initialized
        # before so that only the main worker performs io
        row = CSVLossEntry(
            gpu_rank=self._local_rank,
            timestamp=str(datetime.now()),
            kind=kind.value,
            elements_seen=elements_seen,
            epoch=epoch,
            batch=batch,
            cum_batch=cum_batch,
            loss=loss,
            lr=lr,
            avg_grad=avg_grad,
            avg_grad_clipped=avg_grad_clipped,
        )
        self._csv_loss_writer.write_row(row)

    def _write_csv_timemem(
        self, cum_batch: int, num_items, kind: ModelMode, fw_time: float, bw_time: float | None
    ):
        mem_snapshot = cuda_memory_snapshot(self._device)
        row = CSVTimeAndMemSnapshotEntry(
            kind=kind.value,
            num_items=num_items,
            cum_batch=cum_batch,
            fw_time=fw_time,
            bw_time=bw_time,
            allocated=mem_snapshot.allocated,
            allocated_max=mem_snapshot.allocated_max,
            reserved=mem_snapshot.reserved,
            reserved_max=mem_snapshot.reserved_max,
            total=mem_snapshot.total,
        )
        self._csv_timemem_writer.write_row(row)

    def _save_best_models(self) -> tuple[int, int, Path]:
        num_written = 0
        num_overwritten = 0
        best_checkpoint = Path()
        if not self._is_main_worker:
            return num_written, num_overwritten, best_checkpoint

        # save final n best models - best models are iterated first
        for idx, (loss, model_state) in enumerate(self._top_n_models):
            did_overwrite, checkpoint_path = save_model_state(
                self._output_dir,
                f"{self.config.io.name_model}_top{idx + 1}.pth",
                model=model_state,
                loss=loss,
            )
            num_written += 1
            if idx == 0:
                best_checkpoint = checkpoint_path
            if did_overwrite:
                num_overwritten += 1
        return num_written, num_overwritten, best_checkpoint

    def _log_cuda_memory_snapshot(self, cumulative_batch_idx: int | None) -> None:
        if self._is_cuda:
            snapshot = cuda_memory_snapshot(self._device)
            prefix = f"[{cumulative_batch_idx}] " if cumulative_batch_idx else ""
            self._log.debug(f"{prefix}CUDA memory: {snapshot}")

    def _calc_logging_points(
        self,
        total_batch_iters: int,
        start_batch_idx: int,
    ) -> tuple[set[int], set[int]]:
        """
        Calculate the indices for logging the training loss and validate based
        on the total amount of batch iterations and offset start batch index
        (when resuming training)
        """
        log_train_loss_amount = self.config.io.log_train_loss_amount
        if total_batch_iters < log_train_loss_amount:
            self._log.warning(
                f"Less batch iterations ({total_batch_iters}) "
                f"than number of train loss log points ({log_train_loss_amount}). "
                f"Clipping train loss log points to {total_batch_iters}"
            )
            log_train_loss_amount = total_batch_iters

        validate_amount = self.config.io.validate_amount
        if total_batch_iters < validate_amount:
            self._log.warning(
                f"Less batch iterations ({total_batch_iters}) "
                f"than number of validation runs ({validate_amount}). "
                f"Clipping validation points to {total_batch_iters}"
            )
            validate_amount = total_batch_iters
        log_train_loss_idxs = set(
            torch.linspace(
                start_batch_idx,
                start_batch_idx + total_batch_iters - 1,
                log_train_loss_amount,
            )
            .long()
            .tolist()
        )
        run_valid_interval_idxs = set(
            torch.linspace(
                start_batch_idx,
                start_batch_idx + total_batch_iters - 1,
                validate_amount,
            )
            .long()
            .tolist()
        )

        return log_train_loss_idxs, run_valid_interval_idxs

    def _save_training_state(self, batch_i: int, epoch: int):
        if not self._is_main_worker:
            return
        num_written, num_overwritten, best_model_path = self._save_best_models()
        self._log.debug(
            f"Saved {num_written} best model(s) (overwrote {num_overwritten})",
        )

        # mutate running config in place, then save back
        self._running_resume_conf.next_batch_index = batch_i
        self._running_resume_conf.next_epoch_index = epoch
        self._running_resume_conf.checkpoint_file = str(best_model_path)

        self._dump_output_config()
        self._log.debug(f"Saved training state at epoch {epoch}, batch {batch_i}")

    def avg_gradient_value(self) -> float:
        gradients = [
            p.grad.mean().item() for p in self._model_dist.parameters() if p.grad is not None
        ]
        return sum(gradients) / len(gradients)

    """ Training and evaluation """

    def _evaluate(
        self,
        model: torch.nn.Module,
        loader: DataLoader[TBatch],
        items_seen_so_far: int,
        cumulative_batch_idx: int,
    ) -> float:
        """
        Evaluate any model on any dataset.
        """
        model.eval()
        loss = 0.0
        time_taken = 0.0
        target_iters = (
            min(self.config.train.max_eval_steps, len(loader))
            if self.config.train.max_eval_steps
            else len(loader)
        )
        for it, batch in enumerate(
            tqdm(
                loader,
                total=target_iters,
                desc="Evaluating",
                leave=False,
                disable=not self.options.display_progress,
                mininterval=self.options.valid_prog_min_interval_seconds,
            )
        ):
            if it == target_iters:
                break
            with torch.autocast(
                device_type=self._device_type,
                dtype=self.options.amp_dtype,
            ):
                with torch.inference_mode():
                    start_eval = time()
                    loss_tensor = self.model_forward(
                        cast(TModel, model),
                        batch=batch,
                        device=self._device,
                    )
                    eval_time = time() - start_eval
                    loss += float(loss_tensor.item())
                    # for the eval dataloader, we don't drop the last batch,
                    # hence, the last batch might have a lower batch size.
                    # therefore, count manually to report accurate times per
                    # element
                    time_taken += eval_time

        if self.options.track_first_fw_bw_exec_times:
            self._write_csv_timemem(
                cum_batch=cumulative_batch_idx,
                kind=ModelMode.VALID,
                fw_time=time_taken,
                bw_time=None,
                num_items=items_seen_so_far,
            )
        return loss / target_iters

    def train(
        self,
        train_dataset: DistributedDataset[TBatch],
        valid_dataset: DistributedDataset[TBatch],
    ) -> TModel | None:
        self._running_summary_stats.training_start = datetime.now().isoformat()

        def on_error(error: Exception, retries_left: int):
            self._log.fatal(
                f"Training failed, {retries_left}/{self.options.max_train_restarts} retries left"
            )
            self._log.fatal(error, exc_info=True)
            self._running_summary_stats.error = str(error)
            if self.options.display_progress:
                print(error)

        train_with_retry = retry(self.options.max_train_restarts, on_error=on_error)(self._train)
        best_model = train_with_retry(train_dataset, valid_dataset)

        self._running_summary_stats.training_end = datetime.now().isoformat()
        self._log_cuda_memory_snapshot(-99)
        self._dump_output_config()
        return best_model

    def _get_dataloader(
        self,
        dataset: DistributedDataset[TBatch],
        data_loader_kwargs: dict[str, Any],
        **additional_data_loader_kwargs: dict[str, Any],
    ) -> DataLoader:
        """Generic data loader instantiation.

        Args:
            dataset: a distributed dataset object.
            data_loader_kwargs: additional arguments for the data loader.

        Returns:
            a data loader.
        """
        return DataLoader(dataset, **{**data_loader_kwargs, **additional_data_loader_kwargs})

    def get_train_dataloader(
        self, dataset: DistributedDataset[TBatch], **additional_data_loader_kwargs: dict[str, Any]
    ) -> DataLoader:
        """Train data loader instantiation.

        Args:
            dataset: a distributed dataset object.
            data_loader_kwargs: additional arguments for the data loader.

        Returns:
            the train data loader.
        """
        return self._get_dataloader(
            dataset=dataset,
            data_loader_kwargs=dict(
                batch_size=self.config.train.batch_size,
                pin_memory=True,
                shuffle=self.config.train.shuffle_train,  # False by default
                # drop the last batch so all batches have the same num of elements
                drop_last=True,
                # no need for a distributed sampler, the dataset is already distributed
                sampler=None,
            ),
            **additional_data_loader_kwargs,
        )

    def get_valid_dataloader(
        self, dataset: DistributedDataset[TBatch], **additional_data_loader_kwargs: dict[str, Any]
    ) -> DataLoader:
        """Validation data loader instantiation.

        Args:
            dataset: a distributed dataset object.
            data_loader_kwargs: additional arguments for the data loader.

        Returns:
            the validation data loader.
        """
        return self._get_dataloader(
            dataset=dataset,
            data_loader_kwargs=dict(
                pin_memory=True,
                shuffle=self.config.train.shuffle_eval,  # False by default
                drop_last=False,
                batch_size=self.config.train.batch_size,
            ),
            **additional_data_loader_kwargs,
        )

    def get_test_dataloader(
        self, dataset: DistributedDataset[TBatch], **additional_data_loader_kwargs: dict[str, Any]
    ) -> DataLoader:
        """Test data loader instantiation.

        Args:
            dataset: a distributed dataset object.
            data_loader_kwargs: additional arguments for the data loader.

        Returns:
            the test data loader.
        """
        return self._get_dataloader(
            dataset=dataset,
            data_loader_kwargs=dict(
                pin_memory=True,
                shuffle=False,
                batch_size=self.config.train.batch_size,
            ),
            **additional_data_loader_kwargs,
        )

    def _train(
        self,
        train_dataset: DistributedDataset[TBatch],
        valid_dataset: DistributedDataset[TBatch],
    ) -> TModel:
        """
        Train a model on a training dataset and occasionally run it on the
        validation set.
        """

        if self._is_cuda:
            torch.cuda.empty_cache()

        train_conf = self.config.train

        # instantiate train and validation data loaders, currently
        # no additional arguments forwarded to instantiation.
        train_loader = self.get_train_dataloader(train_dataset)
        valid_loader = self.get_valid_dataloader(valid_dataset)

        # calculate the number of data elements this worker should train on
        # based on the number of (global) target elements and number of workers
        # (cpus/gpus) available. because we use a distributed sampler, the local
        # test data loader only sees 1/world_size of the training data already
        global_target_elements = train_conf.target_elements
        local_target_elements = global_target_elements // self._world_size

        # by setting drop_last=True in the train loader, we make sure all batches have
        # the same number of elements
        if self.config.train.target_elements_strategy == "batch":
            elements_per_batch = self.config.train.batch_size
        else:
            elements_per_batch = self.config.train.batch_size * self.config.params.input_seq_len

        # because global target elements is a lower bound - we always want to
        # train on at least this number of elements - we may train on one more
        # batch (due to batch sizes and sequence lengths). in order to reach the
        # lower bound, the actual number of elements trained per worker may be
        # slightly higher.
        local_batch_iters = math.ceil(local_target_elements / elements_per_batch)
        expected_local_elements = elements_per_batch * local_batch_iters
        expected_global_elements = expected_local_elements * self._world_size
        delta = expected_global_elements - global_target_elements

        self._log.debug(f"Global target elements: {global_target_elements}")
        self._log.debug(
            f"Local target elements: {local_target_elements} ({self._world_size} workers)"
        )
        self._log.debug(f"Elements per batch: {elements_per_batch} (bs: {train_conf.batch_size}) ")
        self._log.debug(
            f"Expected global target elements (w.r.t batch size): {expected_global_elements}"
        )
        self._log.debug(
            f"Expected local target elements (w.r.t batch size): {expected_local_elements}"
        )

        self._log.debug(f"Target element delta (global): {delta} elements")
        self._log.info(f"Running {local_batch_iters} batch iterations")

        optimizer = self.configure_optimizer(self._model_dist.parameters())

        local_gradient_steps = local_batch_iters // train_conf.gradient_accumulate_every
        scheduler = self.configure_scheduler(optimizer, local_gradient_steps)

        epoch: int = 0
        epoch_batch_idx: int = 0
        if self.config.resume:
            self._log.debug("Resuming training, offsetting start epoch and batch index")
            epoch = self.config.resume.next_epoch_index
            epoch_batch_idx = self.config.resume.next_batch_index
            train_dataset.offset_to(epoch)
        else:
            self._log.info("Starting training from scratch")
        self._log.debug(f"Starting from epoch {epoch}")
        self._log.debug(f"Starting from batch {epoch_batch_idx}")
        self._log_cuda_memory_snapshot(None)

        cumulative_batch_idx_start = len(train_loader) * epoch + epoch_batch_idx
        global_log_train_idxs, global_run_valid_idxs = self._calc_logging_points(
            local_batch_iters,
            start_batch_idx=cumulative_batch_idx_start,
        )

        def before_new_epoch(epoch: int) -> None:
            self._log.info(f"Initializing epoch {epoch}")
            train_dataset.offset_to(epoch)

        gradient_scaler = torch.GradScaler(device=self._device_type)

        # total elements seen during trainings
        elements_seen_total = 0
        curr_avg_grad: float = -1
        curr_avg_grad_clipped: float = -1
        for iteration in tqdm(
            epoch_cycler(
                train_loader,
                before_new_epoch=before_new_epoch,
                start_epoch=epoch,
                start_batch=epoch_batch_idx,
                max_iters=local_batch_iters,
            ),
            desc="Training",
            mininterval=self.options.train_prog_min_interval_seconds,
            disable=not self.options.display_progress,
        ):
            batch: TBatch
            next_epoch: int
            next_batch_idx: int

            epoch, epoch_batch_idx, batch = iteration.epoch, iteration.batch, iteration.item
            next_epoch, next_batch_idx = iteration.next_epoch, iteration.next_batch

            # keep track of the cumulative batch index across epochs for
            # logging. this is used for the log points (when to log train loss
            # and run validation). the value is also used to log the global
            # batch index for convenient post-processing of the logs
            cumulative_batch_idx = len(train_loader) * epoch + epoch_batch_idx
            log_prefix = f"{[cumulative_batch_idx]}"

            self._model_dist.train()

            # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
            with torch.autocast(
                device_type=self._device_type,
                dtype=self.options.amp_dtype,
            ):
                start_fw_measure = time()
                train_loss = self.model_forward(
                    # warning - we cast so that the arguments type hints for
                    # TModel.forward() are preserved, however, because the model
                    # has been wrapped with DistributedDataParallel, other
                    # methods might not be available. Use only model.forward()
                    cast(TModel, self._model_dist),
                    batch=batch,
                    device=self._device,
                )
                fw_exec_time = time() - start_fw_measure
                train_loss_as_flt = float(train_loss.item())
                if math.isnan(train_loss_as_flt):
                    shape = batch.shape if isinstance(batch, torch.Tensor) else batch[0].shape
                    self._log.error(
                        f"Invalid loss at batch {epoch_batch_idx}, epoch {epoch}. Train loss (raw): {train_loss}, batch: {shape}"
                    )

                # scale the gradient
                train_loss = train_loss / train_conf.gradient_accumulate_every
            elements_seen_total += elements_per_batch

            scaled_loss = gradient_scaler.scale(train_loss)

            start_bw_measure = time()
            scaled_loss.backward()
            bw_exec_time = time() - start_bw_measure

            # if enabled, report forward/backward pass execution times for the
            # first track_first_fw_bw_exec_times iterations as well as cuda
            # memory usage. after a few iterations, we can usually be sure
            # memory will not further increase assuming there are no memory
            # leaks. skip the first forward/backward pass, which takes much more
            # time due to the construction of the computation graph, optimizer
            # warmup, etc.

            if self.options.track_first_fw_bw_exec_times:
                self.options.track_first_fw_bw_exec_times -= 1
                self._write_csv_timemem(
                    cum_batch=cumulative_batch_idx,
                    kind=ModelMode.TRAIN,
                    fw_time=fw_exec_time,
                    bw_time=bw_exec_time,
                    # batch size is constant during training
                    num_items=self.config.train.batch_size,
                )

            if cumulative_batch_idx in global_log_train_idxs:
                self._log.info(f"{log_prefix} Training loss: {train_loss_as_flt}")
                self._write_csv_loss(
                    ModelMode.TRAIN,
                    loss=train_loss_as_flt,
                    epoch=epoch,
                    batch=epoch_batch_idx,
                    cum_batch=cumulative_batch_idx,
                    elements_seen=elements_seen_total,
                    lr=scheduler.get_last_lr()[0],
                    avg_grad=curr_avg_grad,
                    avg_grad_clipped=curr_avg_grad_clipped,
                )

            # accumulate the gradient with clipping:
            # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
            if (epoch_batch_idx + 1) % train_conf.gradient_accumulate_every == 0:
                # restore the scaled gradient for clipping
                gradient_scaler.unscale_(optimizer)

                curr_avg_grad = self.avg_gradient_value()

                if (max_clip := self.config.train.gradient_clipping) is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self._model_dist.parameters(),
                        max_clip,
                    )

                curr_avg_grad_clipped = self.avg_gradient_value()

                gradient_scaler.step(optimizer)
                scale = gradient_scaler.get_scale()
                gradient_scaler.update()

                # https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930/7
                skip_lr_sched = scale > gradient_scaler.get_scale()
                if scheduler and not skip_lr_sched:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # before evaluation, we do not perform a gradient update. hence, the
            # "elements_seen_total" we log below might be slightly off.
            # specifically, this number might be larger than the true number.
            # this is because the validation might be performed while gradients
            # are still being accumulated, and thus have the model has not
            # learned from the elements yet. on a large scale, this hardly
            # matters
            if not self.options.skip_validation and cumulative_batch_idx in global_run_valid_idxs:
                valid_loss = self._evaluate(
                    self._model_dist,
                    valid_loader,
                    items_seen_so_far=elements_seen_total,
                    cumulative_batch_idx=cumulative_batch_idx,
                )
                self._log.info(f"{log_prefix} Validation loss: {valid_loss}")
                self._write_csv_loss(
                    ModelMode.VALID,
                    loss=valid_loss,
                    epoch=epoch,
                    batch=epoch_batch_idx,
                    cum_batch=cumulative_batch_idx,
                    elements_seen=elements_seen_total,
                    lr=-1,
                    avg_grad=-1,
                    avg_grad_clipped=-1,
                )

                # after validating, save the state, maybe it's really good!
                # before i/o, use a barrier to make sure training states are in
                # sync (as seen in https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html )
                dist.barrier()
                original_model = self._unpack_distributed_model(self._model_dist)
                self._top_n_models.add(
                    (
                        valid_loss,
                        original_model.state_dict(),
                    )
                )
                self._save_training_state(next_batch_idx, next_epoch)

        else:
            # we have seen exactly local_batch_iters batches
            elements_match = elements_seen_total == expected_local_elements
            if not elements_match:
                self._log.fatal(
                    f"Mismatch between expected and actual elements seen: {expected_local_elements}, {elements_seen_total}"
                )
            self._log.info("Finished training")
            self._log.info(f"Stats (local): Elements seen: {elements_seen_total}")

        best_model = self._unpack_distributed_model(self._model_dist)

        if self._is_main_worker:
            # if, on the main worker, populate the model with the best state
            # non-main workers will simply return the latest model, which won't
            # be used anyway because testing happens only on the main worker
            ((least_loss, best_state),) = self._top_n_models.get_top(1)
            best_model.load_state_dict(best_state)
            self._log.info(f"Returning model with least loss ({least_loss})")

        self._log_cuda_memory_snapshot(None)

        return best_model

    def test(
        self,
        test_dataset: DistributedDataset[TBatch],
        model: torch.nn.Module,
    ) -> None:
        # test only on main worker
        if not self._is_main_worker:
            return

        # instantiate test data loader, currently
        # no additional arguments forwarded to instantiation.
        test_loader = self.get_test_dataloader(test_dataset)
        self._log.info("Started testing")
        model.eval()
        test_loss = self._evaluate(model, test_loader, -1, -1)
        self._log.info(f"Test loss: {test_loss}")
        self._write_csv_loss(
            ModelMode.TEST,
            loss=test_loss,
            elements_seen=-1,
            epoch=-1,
            batch=-1,
            cum_batch=-1,
            lr=-1,
            avg_grad=-1,
            avg_grad_clipped=-1,
        )
        self._log.info("Finished testing")
        return None
