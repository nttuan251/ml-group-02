from typing import Any, ClassVar

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset

from reproduction_code.data.core.data_package import DataPackage
from reproduction_code.data.fetchers.fetcher import Fetcher
from reproduction_code.data.tasks.CORe50_task import CORe50Task
from reproduction_code.data.tasks.task import Task


class CORe50Fetcher(Fetcher):
    NUM_CLASSES: ClassVar[int] = 50
    TRAIN_SESSIONS: ClassVar[list[int]] = [1, 2, 4, 5, 6, 8, 9, 11]
    TEST_SESSIONS: ClassVar[list[int]] = [3, 7, 10]

    def _fetch_data(self) -> Any:
        return load_dataset("nhuy165/core50-sessions")

    def _preprocess(self, data: Any) -> DataPackage:
        normalized_data = self._normalize_columns(data)
        full_dataset = normalized_data["train"]  # HF dataset only has 1 split

        test_dataset = self._extract_test_dataset(full_dataset)

        tasks = []

        for session_id in self.TRAIN_SESSIONS:
            train_dataset = self._filter_data_by_session_id(full_dataset, session_id)

            dataset_dict = DatasetDict(
                {
                    "train": train_dataset,
                    "test": test_dataset,
                }
            )

            task = self._build_task_from_data(dataset_dict)
            tasks.append(task)

        return self._build_package_from_tasks(tasks)

    def _normalize_columns(self, data: DatasetDict) -> DatasetDict:
        return data.rename_columns({"image": "X", "label": "Y"})

    def _extract_test_dataset(self, dataset: Dataset) -> Dataset:
        sessions = np.asarray(dataset["session"])
        selected_instances_bool = np.isin(sessions, self.TEST_SESSIONS)
        selected_instances_indices = np.flatnonzero(selected_instances_bool)

        return dataset.select(selected_instances_indices)

    def _filter_data_by_session_id(self, dataset: Dataset, session_id: int) -> Dataset:
        sessions = np.asarray(dataset["session"])
        selected_instances_bool = sessions == session_id
        selected_instances_indices = np.flatnonzero(selected_instances_bool)

        return dataset.select(selected_instances_indices)

    def _build_task_from_data(self, data: DatasetDict) -> Task:
        class_ids = list(range(self.NUM_CLASSES))

        return CORe50Task(
            class_ids=class_ids,
            data=data,
            random_seed=self.random_seed,
        )

    def _build_package_from_tasks(self, tasks: list[Task]) -> DataPackage:
        return DataPackage(
            tasks=tasks,
            num_classes=self.NUM_CLASSES,
        )
