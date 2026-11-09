import numpy as np
from datasets import DatasetDict

from reproduction_code.data.tasks.task import Task


class FiveDatasetsTask(Task):
    """
    A task class designed for the 5-datasets.
    Information already covered in the docstrings inside the base Task class won't be specified again.
    """

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
        """
        Label offseting is included in the lazy transformation pipeline to prevent labels from different datasets to overlap.

        Args:
            label (int): Initial label.

        Returns:
            int: Label with offset.
        """
        return label + self.label_offset

    def _sample_indices_of_class(
        self,
        labels_column: np.ndarray,
        class_id: int,
        samples_per_class: int,
        rng: np.random.Generator,
    ) -> list[int]:
        return super()._sample_indices_of_class(
            labels_column,
            class_id - self.label_offset,
            samples_per_class,
            rng,
        )
