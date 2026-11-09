from typing import Any, ClassVar

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from PIL.Image import Image
from torch import Tensor
from torchvision.transforms import Compose, InterpolationMode, Resize, ToTensor

from reproduction_code.data.core.batch import Batch
from reproduction_code.data.core.batch_iterator import (
    BatchIterator,
    TorchDatasetAdapter,
)
from reproduction_code.data.tasks.util.convert_channels import ConvertChannels


class Task:
    """
    A task of train and test data. Returns data in the form of batches via a BatchIterator object.
    Data inside a task always have a "train" and a "test" split. Data inside a split always has "X" and "Y" columns. This is ensured through preprocessing after fetching and validation at task initialization.
    In task-agnostic settings. The entire dataset is crammed into one single task.
    """

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
        """
        Returns the "train" split of the dataset in the form of a BatchIterator after applying the lazy transformer (see detailed info at _apply_lazy_transformation and _apply_transform_to_batch)

        Args:
            batch_size (int): Batch size.

        Returns:
            BatchIterator: Returned batches.
        """
        raw_train_dataset = self.data["train"]

        dataset_with_transformer = self._apply_lazy_transformation(raw_train_dataset)

        return self._build_batch_iterator(dataset_with_transformer, batch_size, True)

    def get_test_batches(self, batch_size: int) -> BatchIterator:
        """
        Returns the "test" split of the dataset in the form of a BatchIterator after applying the lazy transformer (see detailed info at _apply_lazy_transformer and _apply_transform_to_batch)

        Args:
            batch_size (int): Batch size.

        Returns:
            BatchIterator: Returned batches.
        """
        raw_test_dataset = self.data["test"]

        dataset_with_transformer = self._apply_lazy_transformation(raw_test_dataset)

        return self._build_batch_iterator(dataset_with_transformer, batch_size, False)

    def get_n_samples_per_class(self, samples_per_class: int) -> Batch:
        """
        Yields a fixed number of samples per class from the data and packages them into a batch before returning (preprocessing included).

        Args:
            samples_per_class (int): Number of samples per class.

        Returns:
            Batch: Packaged preprocessed samples.
        """
        train_dataset = self.data["train"]

        labels = np.asarray(train_dataset["Y"])
        selected_indices = self._sample_indices_of_every_class(
            labels, samples_per_class
        )
        raw_samples: dict[str, list[Any]] = train_dataset.select(selected_indices)[:]

        return self._build_batch(raw_samples)

    def _build_batch(self, raw_samples: dict[str, list[Any]]) -> Batch:
        transformed = self._apply_transform_to_batch(raw_samples)

        X = torch.stack(transformed["X"])
        Y = torch.tensor(transformed["Y"], dtype=torch.long)

        return Batch(X, Y)

    def _build_batch_iterator(
        self, formatted_dataset: Dataset, batch_size: int, shuffle: bool
    ) -> BatchIterator:
        return BatchIterator(
            dataset=TorchDatasetAdapter(formatted_dataset),
            batch_size=batch_size,
            shuffle=shuffle,
            random_seed=self.random_seed,
        )

    def _sample_indices_of_every_class(
        self, labels_column: np.ndarray, samples_per_class: int
    ) -> list[int]:
        rng = np.random.default_rng(self.random_seed)

        selected_indices: list[int] = []
        for class_id in self.class_ids:
            selected_indices.extend(
                self._sample_indices_of_class(
                    labels_column,
                    class_id,
                    samples_per_class,
                    rng,
                )
            )

        return selected_indices

    def _sample_indices_of_class(
        self,
        labels_column: np.ndarray,
        class_id: int,
        samples_per_class: int,
        rng: np.random.Generator,
    ) -> list[int]:
        indices_of_matching_instances = np.flatnonzero(labels_column == class_id)
        assert len(indices_of_matching_instances) >= samples_per_class

        return rng.choice(
            indices_of_matching_instances, size=samples_per_class, replace=False
        ).tolist()

    def _apply_lazy_transformation(self, dataset: Dataset) -> Dataset:
        """
        Applies a lazy transformation to the dataset. "Lazy" means the transformation does not immediately apply to the entire dataset, but only when the DataLoader fetches the needed data.

        Args:
            dataset (Dataset): Dataset.

        Returns:
            Dataset: Dataset after applying lazy transformation.
        """
        return dataset.with_transform(self._apply_transform_to_batch)

    def _apply_transform_to_batch(
        self, raw_batch: dict[str, list[Any]]
    ) -> dict[str, list[Any]]:
        """
        Transforms a batch of data, preprocessing each data point's features and label.

        Args:
            raw_batch (dict[str, list[Any]]): Raw batch of data.

        Returns:
            dict[str, list[Any]]: Transformed batch of data.
        """
        raw_X = raw_batch["X"]
        raw_Y = raw_batch["Y"]

        processed_X = [self._preprocess_data_point(data_point) for data_point in raw_X]
        processed_Y = [self._preprocess_label(label) for label in raw_Y]

        return {
            "X": processed_X,
            "Y": processed_Y,
        }

    def _preprocess_data_point(self, data_point: Image) -> Tensor:
        """
        Applies the transform pipeline to a datapoint's features (see detailed info at _create_data_point_transform_pipeline).

        Args:
            data_point (Image): Data point features.

        Returns:
            Tensor: Transformed data point features.
        """
        transformed_data_point = self.transform(data_point)
        assert isinstance(transformed_data_point, Tensor)

        return transformed_data_point

    def _preprocess_label(self, label: int) -> int:
        """
        Transforms a data point label. Does nothing for most datasets except for 5-datasets, where it applies an offset on the labels.

        Args:
            label (int): Label.

        Returns:
            int: Transformed label.
        """
        return label

    def _create_data_point_transform_pipeline(self) -> Compose:
        """
        Lazy tranformation pipeline, including: converting channels to RGB, resizing image to the correct size (224 x 224), converting image to a Pytorch tensor while also changing layout from HWC to CHW.
        This function is run inside __init__ to create the pipeline function.

        Returns:
            Compose: Returned pipeline.
        """
        pipeline = Compose(
            [
                ConvertChannels("RGB"),
                Resize(self.IMAGE_SHAPE, interpolation=self.RESIZE_INTERPOLATION_MODE),
                ToTensor(),
            ]
        )

        return pipeline

    def _assert_required_splits_exist(self, data: DatasetDict) -> None:
        for split_name in self.REQUIRED_SPLITS:
            assert split_name in data

    def _assert_required_columns_exist(self, data: DatasetDict) -> None:
        for split_name in self.REQUIRED_SPLITS:
            split_columns = data[split_name].column_names

            for column_name in self.REQUIRED_COLUMNS:
                assert column_name in split_columns
