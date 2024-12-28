"""

From the project root, run:
make test_e2e
"""

import math
import os
from pathlib import Path

import torch
from torch.optim import Adam, Optimizer  # type: ignore
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR

from mblm.data.datasets import DistributedDataset
from mblm.train.core.config import (
    CoreIoConfig,
    CoreModelParams,
    CoreTrainConfig,
    GenericEntryConfig,
)
from mblm.train.core.trainer import CoreTrainer, CoreTrainerOptions
from mblm.utils.distributed import process_group
from mblm.utils.io import load_yml
from mblm.utils.logging import create_logger
from mblm.utils.misc import count_params
from mblm.utils.seed import seed_everything

# TODO: Python 3.12, assert_type


STORE_N_MODELS = 2
WRITE_TRAIN_LOSS_TIMES = 20
WRITE_VALID_LOSS_TIMES = 10
NUM_TRAIN_ELEMENTS = 10_000


TBatch = tuple[torch.Tensor, torch.Tensor]


class ModelParams(CoreModelParams):
    hidden_size: int
    output_size: int


class TrainEntryConfig(GenericEntryConfig[ModelParams, CoreTrainConfig, CoreIoConfig]):
    pass


class SimpleNN(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SineDataset(DistributedDataset[TBatch]):
    def __init__(self, num_samples: int, worker_id: int, num_workers: int):
        self.x = torch.linspace(0, 2 * torch.pi, num_samples).unsqueeze(1)
        # complicated dummy function to learn
        self.y = (
            torch.sin(2 * torch.pi * self.x)
            + torch.cos(4 * torch.pi * self.x)
            + 0.5 * self.x**2
            - 0.3 * self.x
        )
        super().__init__(
            data_size=self.x.numel(),
            is_sequential=True,
            seq_len=1,
            worker_id=worker_id,
            num_workers=num_workers,
        )

    def get_sample(self, from_idx):
        to_idx = from_idx + self.seq_len
        return self.x[from_idx:to_idx], self.y[from_idx:to_idx]


class TestTrainer(CoreTrainer[SimpleNN, TBatch, ModelParams, CoreTrainConfig, CoreIoConfig]):
    # note regarding type hints: in the real world, the parameters are inferred
    # correctly via the generics. however, for the purpose of asserting on the
    # output types, we need to annotate them explicitly

    def init_model(self):
        # assert_type(output_dir, Path)
        # assert_type(is_main_worker, bool)
        # assert_type(model_params, ModelParams)
        seed_everything(8)
        return SimpleNN(
            input_size=self.config.params.input_seq_len,
            output_size=self.config.params.output_size,
            hidden_size=self.config.params.hidden_size,
        )

    def model_forward(self, model, batch, device):
        # assert_type(model, SimpleNN)
        # assert_type(batch, BatchType)
        # assert_type(device, str)
        x, y = batch
        output = model.forward(x.to(device))
        loss_function = torch.nn.MSELoss()
        loss: torch.Tensor = loss_function(output, y)
        return loss

    def configure_optimizer(self, parameters) -> Optimizer:
        seed_everything(8)
        return Adam(
            parameters,
            lr=self.config.train.learning_rate,
            betas=(0.9, 0.95),
        )

    def configure_scheduler(self, optimizer: Optimizer, local_gradient_steps: int) -> LRScheduler:
        seed_everything(8)
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

    def configure_count_parameters(self, model: SimpleNN):
        return count_params(model, ("fc_layers", [model.fc1, model.fc2]))

    def configure_run_id(self) -> str:
        return os.environ["TEST_ID"]


def run(config: TrainEntryConfig) -> None:
    log = create_logger(__name__)
    log.info("Initiating distributed training")

    try:
        with process_group(backend="gloo") as run_vars:
            train_dataset = SineDataset(
                NUM_TRAIN_ELEMENTS,
                worker_id=run_vars.local_rank,
                num_workers=run_vars.world_size,
            )
            valid_dataset = SineDataset(
                100,
                worker_id=0,
                num_workers=1,
            )
            test_dataset = SineDataset(
                100,
                worker_id=0,
                num_workers=1,
            )
            trainer = TestTrainer(
                config,
                run_vars=run_vars,
                options=CoreTrainerOptions(
                    # disable tqdm for tests
                    display_progress=False,
                ),
            )
            best_model = trainer.train(train_dataset, valid_dataset)
            if best_model:
                trainer.test(test_dataset, best_model)
    except Exception as error:
        log.fatal(error)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-c", type=Path, required=True, dest="config_path")
    args = parser.parse_args()
    config_path: Path = args.config_path
    cfg = load_yml(config_path, parse_to=TrainEntryConfig)
    run(cfg)
