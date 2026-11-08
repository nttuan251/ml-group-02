import torch

from reproduction_code.data.core.batch import Batch
from reproduction_code.data.tasks.task import Task


class RehearsalBuffer:
    def __init__(self, samples_per_class: int, random_seed: int) -> None:
        self.samples_per_class = samples_per_class
        self.buffer_X = torch.empty(0)
        self.buffer_Y = torch.empty(0, dtype=torch.long)
        self.recorded_classes: set[int] = set()
        self.random_seed = random_seed
        self.generator = torch.Generator().manual_seed(self.random_seed)

    def get_recorded_classes(self) -> set[int]:
        return self.recorded_classes

    def is_empty(self) -> bool:
        return self.buffer_Y.shape[0] == 0

    def fetch_and_save_samples_from_task(self, task: Task) -> None:
        """
        Fetch a fixed number of samples per class from the task and add them to the buffer.

        Args:
            task (Task): Task.
        """
        class_ids_in_task = task.get_class_ids()
        no_class_id_overlap = len(set(class_ids_in_task) - self.recorded_classes) == 0
        assert no_class_id_overlap

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
        buffer_size = self.buffer_Y.shape[0]
        assert buffer_size > 0

        sample_size = min(batch_size, buffer_size)

        random_indices = torch.randperm(buffer_size, generator=self.generator)[
            :sample_size
        ]

        return Batch(
            X=self.buffer_X[random_indices],
            Y=self.buffer_Y[random_indices],
        )

    def _add_samples(self, batch_to_buffer: Batch) -> None:
        X_buffer = batch_to_buffer.get_X()
        Y_buffer = batch_to_buffer.get_Y()

        if self.buffer_X.numel() == 0:
            self.buffer_X = X_buffer
            self.buffer_Y = Y_buffer
        else:
            self.buffer_X = torch.cat((self.buffer_X, X_buffer), dim=0)
            self.buffer_Y = torch.cat((self.buffer_Y, Y_buffer), dim=0)
