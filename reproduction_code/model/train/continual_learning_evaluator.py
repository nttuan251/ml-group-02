import torch
from torch import device

from reproduction_code.data.core.batch import Batch
from reproduction_code.data.tasks.task import Task
from reproduction_code.model.forward.L2P_model import L2PModel


class ContinualLearningEvaluator:
    """
    Calculates model accuracy on tasks.
    """

    def __init__(
        self,
        model: L2PModel,
        batch_size: int,
        device: device | str,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.device = device

    def calculate_accuracies_on_each_task(self, tasks: list[Task]) -> dict[int, float]:
        """
        Calculates and records average accuracies evaluated on each input tasks with the current model.

        Args:
            tasks (list[Task]): Input tasks.

        Returns:
            dict[int, float]: Recorded average accuracies.
        """
        self.model.eval()
        task_accuracies = {}

        for current_task_index, current_task in enumerate(tasks):
            current_task_accuracy = self._calculate_average_accuracy_on_task(
                current_task
            )
            task_accuracies[current_task_index] = current_task_accuracy

        return task_accuracies

    def _calculate_average_accuracy_on_task(self, task: Task) -> float:
        """
        Calculates average accuracy on a task by dividing the total correct predictions over the number of samples.

        Args:
            task (Task): Task.

        Returns:
            float: Average accuracy.
        """
        batch_iterator = task.get_test_batches(self.batch_size)

        total_correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for batch in batch_iterator:
                correct_predictions, samples = (
                    self._get_correct_predictions_and_total_samples_from_batch(batch)
                )
                total_correct_predictions += correct_predictions
                total_samples += samples

        if total_samples == 0:
            return 0.0

        task_accuracy = total_correct_predictions / total_samples

        return task_accuracy

    def _get_correct_predictions_and_total_samples_from_batch(
        self, batch: Batch
    ) -> tuple[int, int]:
        batch.to(self.device)

        X = batch.get_X()
        Y = batch.get_Y()

        logits, _ = self.model.classify_and_calculate_surrogate_loss_with_L2P(X)
        predictions = torch.argmax(logits, dim=1)

        num_correct_predictions = (predictions == Y).sum().item()
        total_samples = Y.shape[0]

        return int(num_correct_predictions), int(total_samples)
