from typing import Any, ClassVar

from datasets import DatasetDict, load_dataset

from reproduction_code.data.core.data_package import DataPackage
from reproduction_code.data.fetchers.fetcher import Fetcher
from reproduction_code.data.tasks.five_datasets_task import FiveDatasetsTask
from reproduction_code.data.tasks.task import Task


class FiveDatasetsFetcher(Fetcher):
    """
    Fetcher fetching data for the 5-datasets.
    Information already covered in the docstrings inside the base Fetcher class won't be specified again.
    """

    NUM_CLASSES_PER_TASK: ClassVar[int] = 10
    NUM_TASKS: ClassVar[int] = 5

    def _fetch_data(self) -> dict[str, DatasetDict]:
        return {
            "cifar10": load_dataset("uoft-cs/cifar10"),
            "mnist": load_dataset("ylecun/mnist"),
            "fashion_mnist": load_dataset("zalando-datasets/fashion_mnist"),
            "svhn": load_dataset("ufldl-stanford/svhn", "cropped_digits"),
            "notmnist": load_dataset("anubhavmaity/notMNIST"),
        }

    def _preprocess(self, data: dict[str, DatasetDict]) -> DataPackage:
        """
        Preprocesses data by generating distinct class labels for the five datasets. Each dataset corresponds to one task. The current datasets ordering is: CIFAR10, MNIST, Fashion-MNIST, SVHN, notMNIST (this order is arbitrary and was not explicitly stated in the original paper).

        Args:
            data (dict[str, DatasetDict]): Raw fetched data.

        Returns:
            DataPackage: Preprocessed data.
        """
        normalized_data = self._normalize_columns_and_splits(data)

        tasks = []
        global_offset = 0

        for dataset in normalized_data.values():
            class_ids = self._generate_class_ids(global_offset)

            task = self._build_task_from_data(dataset, class_ids, global_offset)
            tasks.append(task)

            global_offset += self.NUM_CLASSES_PER_TASK

        return self._build_package_from_tasks(tasks)

    def _normalize_columns_and_splits(
        self, data: dict[str, DatasetDict]
    ) -> dict[str, DatasetDict]:
        old_column_names_map: dict[str, tuple[str, str]] = {
            "cifar10": ("img", "label"),
            "mnist": ("image", "label"),
            "fashion_mnist": ("image", "label"),
            "svhn": ("image", "label"),
            "notmnist": ("image", "label"),
        }

        for dataset_name in data:
            X_old_name, Y_old_name = old_column_names_map[dataset_name]

            data[dataset_name] = (
                data[dataset_name]
                .rename_columns(
                    {
                        X_old_name: "X",
                        Y_old_name: "Y",
                    }
                )
                .select_columns(["X", "Y"])
            )
            if "extra" in data[dataset_name]:
                del data[dataset_name]["extra"]

        return data

    def _generate_class_ids(self, offset: int) -> list[int]:
        """
        Generates distinct class ids while keeping track of a global offset so that the labels from the different datasets don't overlap.

        Args:
            offset (int): Global offset.

        Returns:
            list[int]: Generated class labels.
        """
        return list(range(offset, offset + self.NUM_CLASSES_PER_TASK))

    def _build_task_from_data(
        self, data: DatasetDict, class_ids: list[int], offset: int
    ) -> Task:
        return FiveDatasetsTask(
            class_ids=class_ids,
            data=data,
            random_seed=self.random_seed,
            label_offset=offset,
        )

    def _build_package_from_tasks(self, tasks: list[Task]) -> DataPackage:
        return DataPackage(
            tasks=tasks,
            num_classes=self.NUM_CLASSES_PER_TASK * self.NUM_TASKS,
        )
