from datasets import DatasetDict

from reproduction_code.data.tasks.task import Task


class FiveDatasetsTask(Task):
    def __init__(
        self,
        class_ids: list[int],
        data: DatasetDict,
        random_seed: int,
        label_offset: int,
    ) -> None:
        super().__init__(class_ids, data, random_seed)

        self.label_offset = label_offset

    def _preprocess_label(self, label: int) -> int:
        return label + self.label_offset
