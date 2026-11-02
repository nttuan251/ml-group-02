from abc import ABC, abstractmethod
from typing import Any, ClassVar

from datasets import Dataset, DatasetDict
from PIL.Image import Image
from torch import Tensor

from reproduction_code.data.core.batch_iterator import (
    BatchIterator,
    TorchDatasetAdapter,
)


class Task(ABC):
    REQUIRED_SPLITS: ClassVar[tuple[str, ...]] = ("train", "test")
    REQUIRED_COLUMNS: ClassVar[tuple[str, ...]] = ("X", "Y")

    def __init__(
        self,
        class_ids: list[int],
        data: DatasetDict,
        random_seed: int,
    ) -> None:
        self.class_ids = class_ids
        self.data = data
        self.random_seed = random_seed

    def get_class_ids(self) -> list[int]:
        return self.class_ids

    def get_train_batches(self, batch_size: int) -> BatchIterator:
        raw_train_dataset = self.data["train"]

        dataset_with_transformer = self._apply_lazy_transformer(raw_train_dataset)

        return self._make_batches(dataset_with_transformer, batch_size, True)

    def get_test_batches(self, batch_size: int) -> BatchIterator:
        raw_test_dataset = self.data["test"]

        dataset_with_transformer = self._apply_lazy_transformer(raw_test_dataset)

        return self._make_batches(dataset_with_transformer, batch_size, False)

    def _make_batches(
        self, formatted_dataset: Dataset, batch_size: int, shuffle: bool
    ) -> BatchIterator:
        return BatchIterator(
            dataset=TorchDatasetAdapter(formatted_dataset),
            batch_size=batch_size,
            shuffle=shuffle,
            random_seed=self.random_seed,
        )

    def _apply_lazy_transformer(self, dataset: Dataset) -> Dataset:
        return dataset.with_transform(self._apply_transform_to_batch)

    def _apply_transform_to_batch(
        self, raw_batch: dict[str, list[Any]]
    ) -> dict[str, list[Any]]:
        raw_X = raw_batch["X"]
        raw_Y = raw_batch["Y"]

        processed_X = [self._preprocess_data_point(data_point) for data_point in raw_X]

        return {
            "X": processed_X,
            "Y": raw_Y,
        }

    @abstractmethod
    def _preprocess_data_point(self, image: Image) -> Tensor:
        pass

    def _assert_required_splits_exist(self, data: DatasetDict) -> None:
        for split_name in self.REQUIRED_SPLITS:
            assert split_name in data

    def _assert_required_columns_exist(self, data: DatasetDict) -> None:
        for split_name in self.REQUIRED_SPLITS:
            split_columns = data[split_name].column_names

            for column_name in self.REQUIRED_COLUMNS:
                assert column_name in split_columns, (
                    f"Missing column '{column_name}' in split '{split_name}'"
                )
