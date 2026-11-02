import random
from typing import ClassVar

from datasets import DatasetDict, load_dataset

from reproduction_code.data.core.data_package import DataPackage
from reproduction_code.data.fetchers.fetcher import Fetcher
from reproduction_code.data.tasks.CIFAR_task import CIFAR_task
from reproduction_code.data.tasks.task import Task


class CIFARFetcher(Fetcher):
    NUM_CLASSES: ClassVar[int] = 100

    def __init__(self, num_tasks: int, random_seed: int) -> None:
        super().__init__()

        self.num_tasks = num_tasks
        self.random_seed = random_seed

    def _fetch_data(self) -> DatasetDict:
        dataset = load_dataset("uoft-cs/cifar100")
        return dataset

    def _preprocess(self, data: DatasetDict) -> DataPackage:
        shuffled_class_ids = self._generate_shuffled_class_ids()
        task_segmented_class_ids = self._perform_class_ids_segmentation_by_task(
            shuffled_class_ids
        )

        tasks = []

        data_with_normalized_columns = self._normalize_columns(data)

        for class_ids in task_segmented_class_ids:
            filtered_task_data = self._filter_data_by_class_ids(
                data_with_normalized_columns, class_ids
            )

            task = self._build_task_from_data(filtered_task_data, class_ids)

            tasks.append(task)

        package = self._build_package_from_tasks(tasks)

        return package

    def _generate_shuffled_class_ids(self) -> list[int]:
        ordered_class_ids = list(range(self.NUM_CLASSES))
        random_generator = random.Random(self.random_seed)
        shuffled_class_ids = random_generator.sample(
            ordered_class_ids, self.NUM_CLASSES
        )

        return shuffled_class_ids

    def _perform_class_ids_segmentation_by_task(
        self, shuffled_class_ids: list[int]
    ) -> list[list[int]]:
        classes_per_task = self.NUM_CLASSES // self.num_tasks
        segmented_class_ids = []

        for task_index in range(self.num_tasks):
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
        class_ids_set = set(class_ids)

        filtered_train_split = data["train"].filter(
            lambda data_point: data_point["Y"] in class_ids_set
        )
        filtered_test_split = data["test"].filter(
            lambda data_point: data_point["Y"] in class_ids_set
        )

        filtered_data = DatasetDict(
            {
                "train": filtered_train_split,
                "test": filtered_test_split,
            }
        )

        return filtered_data

    def _build_task_from_data(self, data: DatasetDict, class_ids: list[int]) -> Task:
        return CIFAR_task(
            class_ids=class_ids,
            data=data,
            random_seed=self.random_seed,
        )

    def _build_package_from_tasks(self, tasks: list[Task]) -> DataPackage:
        return DataPackage(
            tasks=tasks,
            num_classes=self.NUM_CLASSES,
        )
