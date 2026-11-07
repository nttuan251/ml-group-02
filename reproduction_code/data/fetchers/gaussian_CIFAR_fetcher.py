import random
from typing import ClassVar

from datasets import Dataset, DatasetDict

from reproduction_code.data.core.data_package import DataPackage
from reproduction_code.data.fetchers.CIFAR_fetcher import CIFARFetcher
from reproduction_code.data.tasks.gaussian_CIFAR_task import GaussianCIFARTask
from reproduction_code.data.tasks.task import Task


class GaussianCIFARFetcher(CIFARFetcher):
    """
    Fetcher fetching data for the CIFAR-100 dataset and applying Gaussian schedule on the data.
    Information already covered in the docstrings inside the base Fetcher class and the CIFARFetcher class won't be specified again.
    """

    APPEARANCE_TIME_STD: ClassVar[float] = 0.05

    def _preprocess(self, data: DatasetDict) -> DataPackage:
        """
        Preprocesses data by applying Gaussian schedule on the data (see more information in _apply_gaussian_schedule). This dataset does not have task boundaries (the entire dataset is one giant task) and is meant to be used in task-agnostic training.

        Args:
            data (DatasetDict): Raw fetched data.

        Returns:
            DataPackage: Preprocessed data.
        """

        normalized_data = self._normalize_columns(data)

        scheduled_train_dataset = self._apply_gaussian_schedule(
            normalized_data["train"]
        )

        normalized_scheduled_data = DatasetDict(
            {
                "train": scheduled_train_dataset,
                "test": normalized_data["test"],
            }
        )

        task = self._build_task_from_data(normalized_scheduled_data)

        package = self._build_package_from_tasks([task])

        return package

    def _apply_gaussian_schedule(self, dataset: Dataset) -> Dataset:
        """
        Applies Gaussian schedule to create a continuous data stream where class labels shift slowly.
        Each class label is allocated a peak appearance time (distributed evenly between 0 and 1). Each data point is sampled a specific timestamp using a Gaussian distribution centered around this peak time with a standard deviation specified in APPEARANCE_TIME_STD.

        Args:
            dataset (Dataset): Dataset.

        Returns:
            Dataset: Dataset after applied Gaussian schedule.
        """
        random_generator = random.Random(self.random_seed)

        appearance_time_array = []

        for label in dataset["Y"]:
            # Class appearances peak evenly between 0.0 and 1.0
            label_peak_appearance_time = label / max(1, self.NUM_CLASSES - 1)

            sampled_appearance_time = random_generator.gauss(
                label_peak_appearance_time, self.APPEARANCE_TIME_STD
            )

            appearance_time_array.append(sampled_appearance_time)

        dataset_with_time = dataset.add_column("appearance_time", appearance_time_array)
        dataset_scheduled_by_classes = dataset_with_time.sort(
            "appearance_time"
        ).remove_columns(["appearance_time"])

        return dataset_scheduled_by_classes

    def _build_task_from_data(self, data: DatasetDict) -> Task:
        class_ids = list(range(self.NUM_CLASSES))

        return GaussianCIFARTask(
            class_ids=class_ids,
            data=data,
            random_seed=self.random_seed,
        )
