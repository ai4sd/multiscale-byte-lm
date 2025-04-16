from pathlib import Path

from typing_extensions import Unpack

from mblm.data.datasets import DistributedDataset, DistributedDatasetConfig
from mblm.data.types import BatchMaskedForMLM, ModelMode
from mblm.train.mblm import (
    TrainMaskedEntryConfig,
    masked_dataset_registry,
)


@masked_dataset_registry.register("beep")
class Beep(DistributedDataset[BatchMaskedForMLM]):
    """The beep dataset raw data"""

    def __init__(
        self,
        mode: ModelMode,
        data_dir: str | Path,
        **args: Unpack[DistributedDatasetConfig],
    ):
        # Dummy example - Get data from anywhere, e.g., the disk
        print(f"Reading dataset from {data_dir}")
        if mode == ModelMode.TRAIN:
            # TODO Load the train BEEP FILE.
            data = list(range(10_000))
        elif mode == ModelMode.VALID:
            # TODO Load the Beep Validation file
            data = list(range(2_000))
        elif mode == ModelMode.TEST:
            # TODO Load the Beep TEST file
            pass
        else:
            raise ValueError("This variant isn't implemented yet, please update the code")
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
        # data = torch.tensor(self._data[from_idx : from_idx + self.seq_len])
        # return torch.ones_like(data), data
        raise NotImplementedError()

    @staticmethod
    def from_train_entry_config(
        config: TrainMaskedEntryConfig,
        mode: ModelMode,
        worker_id: int,
        num_workers: int,
    ) -> DistributedDataset[BatchMaskedForMLM]:
        """
        How to parse a training config to a dataset.
        """
        return Beep(
            data_dir=config.io.dataset_dir,
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
