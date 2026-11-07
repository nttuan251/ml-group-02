import random
from typing import ClassVar

import numpy as np
from datasets import Dataset, DatasetDict, NamedSplit, load_dataset

from reproduction_code.data.core.data_package import DataPackage
from reproduction_code.data.fetchers.fetcher import Fetcher
from reproduction_code.data.tasks.CIFAR_task import CIFARTask
from reproduction_code.data.tasks.task import Task


class CIFARFetcher(Fetcher):
    """
    Fetcher fetching data for the CIFAR-100 dataset.
    Information already covered in the docstrings inside the base Fetcher class won't be specified again.
    """

    NUM_CLASSES: ClassVar[int] = 100
    NUM_TASKS: ClassVar[int] = 10

    def _fetch_data(self) -> DatasetDict:
        dataset = load_dataset("uoft-cs/cifar100")
        return dataset

    def _preprocess(self, data: DatasetDict) -> DataPackage:
        """
        Preprocesses data by shuffling the 100 class labels, before dividing them into 10 sets, corresponding to 10 tasks. Train and test data inside each task is chosen according to the shuffled class labels inside that task.

        Args:
            data (DatasetDict): Raw fetched data.

        Returns:
            DataPackage: Preprocessed data.
        """
        shuffled_class_ids = self._generate_shuffled_class_ids()
        task_segmented_class_ids = self._perform_class_ids_segmentation_by_task(
            shuffled_class_ids
        )

        tasks = []

        normalized_data = self._normalize_columns(data)

        for class_ids in task_segmented_class_ids:
            filtered_task_data = self._filter_data_by_class_ids(
                normalized_data, class_ids
            )

            task = self._build_task_from_data(filtered_task_data, class_ids)

            tasks.append(task)

        package = self._build_package_from_tasks(tasks)

        return package

    def _generate_shuffled_class_ids(self) -> list[int]:
        """
        Shuffles the 100 class labels.

        Returns:
            list[int]: The shuffled class labels.
        """
        ordered_class_ids = list(range(self.NUM_CLASSES))
        random_generator = random.Random(self.random_seed)
        shuffled_class_ids = random_generator.sample(
            ordered_class_ids, self.NUM_CLASSES
        )

        return shuffled_class_ids

    def _perform_class_ids_segmentation_by_task(
        self, shuffled_class_ids: list[int]
    ) -> list[list[int]]:
        """
        Divides the shuffled class labels into 10 different sets.

        Args:
            shuffled_class_ids (list[int]): Shuffled class labels.

        Returns:
            list[list[int]]: An array of 10 distinct sets of labels.
        """
        classes_per_task = self.NUM_CLASSES // self.NUM_TASKS
        segmented_class_ids = []

        for task_index in range(self.NUM_TASKS):
            start_index = task_index * classes_per_task
            end_index = start_index + classes_per_task

            current_task_class_ids = shuffled_class_ids[start_index:end_index]

            segmented_class_ids.append(current_task_class_ids)

        return segmented_class_ids

    def _normalize_columns(self, data: DatasetDict) -> DatasetDict:
        return data.rename_columns({"img": "X", "fine_label": "Y"}).select_columns(
            ["X", "Y"]
        )

    def _filter_data_by_class_ids(
        self, data: DatasetDict, class_ids: list[int]
    ) -> DatasetDict:
        """
        Filters data points for one task according to a set of shuffled class labels.

        Args:
            data (DatasetDict): Dataset.
            class_ids (list[int]): A single set of shuffled class labels.

        Returns:
            DatasetDict: Chosen data points.
        """
        filtered_data: dict[str | NamedSplit, Dataset] = {}

        for split_name, dataset in data.items():
            labels = np.asarray(dataset["Y"])
            selected_instances_bool = np.isin(labels, class_ids)
            selected_instances_indices = np.flatnonzero(selected_instances_bool)

            filtered_data[split_name] = dataset.select(selected_instances_indices)

        return DatasetDict(filtered_data)

    def _build_task_from_data(self, data: DatasetDict, class_ids: list[int]) -> Task:
        return CIFARTask(
            class_ids=class_ids,
            data=data,
            random_seed=self.random_seed,
        )

    def _build_package_from_tasks(self, tasks: list[Task]) -> DataPackage:
        return DataPackage(
            tasks=tasks,
            num_classes=self.NUM_CLASSES,
        )
