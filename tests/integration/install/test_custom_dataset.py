# Filename: train_my_mblm.py

import torch
from typing_extensions import Unpack

from mblm import MambaBlock, TransformerBlock
from mblm.data.datasets import DistributedDataset, DistributedDatasetConfig
from mblm.data.types import BatchWithLossMask, ModelMode
from mblm.train.core.config import CoreTrainConfig
from mblm.train.mblm import (
    TrainEntryConfig,
    TrainMBLMIoConfig,
    TrainMBLMParams,
    dataset_registry,
    train_mblm,
)


class MyDataset(DistributedDataset[BatchWithLossMask]):
    def __init__(
        self,
        mode: ModelMode,
        dataset_dir: str,
        **args: Unpack[DistributedDatasetConfig],
    ):
        # Dummy example - Get data from anywhere, e.g., the disk
        print(f"Reading dataset from {dataset_dir}")
        if mode == ModelMode.TRAIN:
            data = list(range(10_000))
        else:
            data = list(range(2_000))
        self._data = data
        super().__init__(
            data_size=len(data),
            is_sequential=True,  # We have a sequential dataset
            **args,
        )

    def get_sample(self, from_idx: int):
        """
        Tell the superclass how to get a single sample - here, a sequence of
        the specified length.
        """
        data = torch.tensor(self._data[from_idx : from_idx + self.seq_len])
        return torch.ones_like(data), data

    @staticmethod
    def from_train_entry_config(
        config: TrainEntryConfig,
        mode: ModelMode,
        worker_id: int,
        num_workers: int,
    ) -> DistributedDataset[BatchWithLossMask]:
        """
        How to parse a training config to a dataset.
        """
        return MyDataset(
            dataset_dir=config.io.dataset_dir,
            mode=mode,
            seq_len=config.params.input_seq_len,
            num_workers=num_workers,
            worker_id=worker_id,
        )

    @staticmethod
    def supports_test_mode() -> bool:
        """
        Whether or not this dataset supports a test mode. Some datasets might not
        expose the answers in their test set so we cannot evaluate a model on it.
        Override if necessary
        """
        return True


# Register dataset with a unique ID
dataset_registry.register("mydataset", MyDataset)

config = TrainEntryConfig(
    io=TrainMBLMIoConfig(
        dataset_dir="data/datasets/my-dataset",
        dataset_id="mydataset",  # Must match the ID above
        name_model="my-model",
        output_dir="data/outputs",
        num_models_to_save=3,
        validate_amount=20,
        log_train_loss_amount=100,
    ),
    train=CoreTrainConfig(
        batch_size=1,
        target_elements=1000,
        target_elements_strategy="sequence",
        learning_rate=0.001,
        gradient_accumulate_every=4,
        gradient_clipping=1,
        shuffle_train=True,
        shuffle_eval=False,
    ),
    params=TrainMBLMParams(
        input_seq_len=128,
        num_tokens=257,
        hidden_dims=[512, 512],
        seq_lens=[16, 8],
        num_layers=[5, 5],
        pad_token_id=256,
        train_checkpoint_chunks=None,
        block=[
            MambaBlock(
                d_state=128,
                d_conv=4,
                expand=2,
                headdim=64,
                pos_emb_type=None,
            ),
            TransformerBlock(
                attn_head_dims=64,
                attn_num_heads=16,
                attn_use_rot_embs=True,
                use_flash_attn=True,
                pos_emb_type="fixed",
            ),
        ],
    ),
)

if __name__ == "__main__":
    train_mblm(config)
