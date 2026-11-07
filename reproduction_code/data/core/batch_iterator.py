from collections.abc import Iterator
from typing import Any

import torch
from datasets import Dataset as HFDataset
from torch import Generator
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from reproduction_code.data.core.batch import Batch


class TorchDatasetAdapter(TorchDataset[dict[str, Any]]):
    """
    An adapter that makes a Hugging Face Dataset object compatible with a Pytorch Dataset object.
    """

    def __init__(self, hf_dataset: HFDataset) -> None:
        self.hf_dataset = hf_dataset

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.hf_dataset[index]


class BatchIterator:
    """
    Packages batches inside of one single task into an iterable for easy processing. Wraps Pytorch's DataLoader object.
    """

    def __init__(
        self,
        dataset: TorchDatasetAdapter,
        batch_size: int,
        shuffle: bool,
        random_seed: int,
        num_workers: int = 4,
    ) -> None:
        """
        Creates a DataLoader with the necessary information. Pinned CPU memory is used to allow for more efficient CPU-GPU memory transfer.

        Args:
            dataset (TorchDatasetAdapter): Dataset.
            batch_size (int): Batch size.
            shuffle (bool): Toggle shuffling, the DataLoader picks data points at random to make batches.
            random_seed (int): Random seed.
            num_workers (int, optional): Number of subprocesses used for loading data.
        """
        assert num_workers > 0

        generator = Generator().manual_seed(random_seed)

        # Shuffling orders start at the same point with the same random seed
        self.data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
            generator=generator,
        )

    @staticmethod
    def _collate_fn(raw_batch: list[dict[str, Any]]) -> Batch:
        """
        Packages a specified number (batch size) of data points into a batch, transforming them into a Batch object. Used to initialize the DataLoader.

        Args:
            raw_batch (list[dict[str, Any]]): Raw batch data points.

        Returns:
            Batch: A packaged Batch object.
        """
        x_array = [item["X"] for item in raw_batch]
        y_array = [item["Y"] for item in raw_batch]

        return Batch(
            X=torch.stack(x_array),
            Y=torch.tensor(y_array, dtype=torch.long),
        )

    def __iter__(self) -> Iterator[Batch]:
        return iter(self.data_loader)

    def __len__(self) -> int:
        return len(self.data_loader)
