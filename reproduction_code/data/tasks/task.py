from typing import Any, ClassVar

from datasets import Dataset, DatasetDict
from PIL.Image import Image
from torch import Tensor
from torchvision.transforms import Compose, InterpolationMode, Resize, ToTensor

from reproduction_code.data.core.batch_iterator import (
    BatchIterator,
    TorchDatasetAdapter,
)
from reproduction_code.data.tasks.util.convert_channels import ConvertChannels


class Task:
    REQUIRED_SPLITS: ClassVar[tuple[str, ...]] = ("train", "test")
    REQUIRED_COLUMNS: ClassVar[tuple[str, ...]] = ("X", "Y")

    IMAGE_SHAPE: ClassVar[tuple[int, int]] = (224, 224)
    # NOTE: Paper did not specify which interpolation mode to use
    RESIZE_INTERPOLATION_MODE: ClassVar[InterpolationMode] = InterpolationMode.BILINEAR

    def __init__(
        self,
        class_ids: list[int],
        data: DatasetDict,
        random_seed: int,
    ) -> None:
        self._assert_required_splits_exist(data)
        self._assert_required_columns_exist(data)

        self.class_ids = class_ids
        self.data = data
        self.random_seed = random_seed

        self.transform = self._create_data_point_transform_pipeline()

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
        processed_Y = [self._preprocess_label(label) for label in raw_Y]

        return {
            "X": processed_X,
            "Y": processed_Y,
        }

    def _preprocess_data_point(self, data_point: Image) -> Tensor:
        transformed_data_point = self.transform(data_point)
        assert isinstance(transformed_data_point, Tensor)

        return transformed_data_point

    def _preprocess_label(self, label: int) -> int:
        return label

    def _assert_required_splits_exist(self, data: DatasetDict) -> None:
        for split_name in self.REQUIRED_SPLITS:
            assert split_name in data

    def _assert_required_columns_exist(self, data: DatasetDict) -> None:
        for split_name in self.REQUIRED_SPLITS:
            split_columns = data[split_name].column_names

            for column_name in self.REQUIRED_COLUMNS:
                assert column_name in split_columns

    def _create_data_point_transform_pipeline(self) -> Compose:
        pipeline = Compose(
            [
                ConvertChannels("RGB"),
                Resize(self.IMAGE_SHAPE, interpolation=self.RESIZE_INTERPOLATION_MODE),
                ToTensor(),
            ]
        )

        return pipeline
