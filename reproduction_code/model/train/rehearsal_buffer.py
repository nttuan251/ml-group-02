import torch

from reproduction_code.data.core.batch import Batch
from reproduction_code.data.tasks.task import Task


class RehearsalBuffer:
    def __init__(
        self,
        samples_per_class: int,
        random_seed: int,
        allow_class_overlap: bool,
    ) -> None:
        self.samples_per_class = samples_per_class
        self.buffer: Batch | None = None
        self.recorded_classes: set[int] = set()
        self.random_seed = random_seed
        self.generator = torch.Generator().manual_seed(self.random_seed)
        self.allow_class_overlap = allow_class_overlap

    def get_recorded_classes(self) -> set[int]:
        return self.recorded_classes

    def is_empty(self) -> bool:
        return self.buffer is None

    def fetch_and_save_samples_from_task(self, task: Task) -> None:
        """
        Fetch a fixed number of samples per class from the task and add them to the buffer.

        Args:
            task (Task): Task.
        """
        class_ids_in_task = task.get_class_ids()

        if not self.allow_class_overlap:
            assert self.recorded_classes.isdisjoint(class_ids_in_task)

        batch_to_buffer = task.get_n_samples_per_class(self.samples_per_class)

        self.recorded_classes.update(class_ids_in_task)
        self._add_samples(batch_to_buffer)

    def get_random_samples(self, batch_size: int) -> Batch:
        """
        Gets random samples from the rehearsal buffer, packing them into a batch.

        Args:
            batch_size (int): Output batch size.

        Returns:
            Batch: Packaged random samples.
        """
        assert self.buffer is not None
        buffer_size = self.buffer.get_batch_size()
        random_indices = torch.randperm(buffer_size, generator=self.generator)[
            : min(batch_size, buffer_size)
        ]

        return self.buffer.select(random_indices)

    def _add_samples(self, batch_to_buffer: Batch) -> None:
        if self.buffer is None:
            self.buffer = batch_to_buffer
        else:
            self.buffer = self.buffer.merge(batch_to_buffer)
