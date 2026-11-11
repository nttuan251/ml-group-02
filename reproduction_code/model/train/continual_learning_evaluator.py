import torch
from torch import Tensor, device

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
        shared_test_set: bool,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.shared_test_set = shared_test_set

    def calculate_accuracies_on_each_task(self, tasks: list[Task]) -> dict[int, float]:
        """
        Calculates and records average accuracies evaluated on each input tasks with the current model.

        Args:
            tasks (list[Task]): Input tasks.

        Returns:
            dict[int, float]: Recorded average accuracies.
        """
        self.model.eval()

        if self.shared_test_set:
            task_accuracy = self._calculate_average_accuracy_on_task(tasks[0])
            task_accuracies = {
                task_index: task_accuracy for task_index in range(len(tasks))
            }

        else:
            task_accuracies = {}
            for current_task_index, current_task in enumerate(tasks):
                current_task_accuracy = self._calculate_average_accuracy_on_task(
                    current_task
                )
                task_accuracies[current_task_index] = current_task_accuracy

                print(f"Accuracy on task {current_task_index}: {current_task_accuracy}")

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

        total_correct_predictions = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        total_samples = 0

        with torch.inference_mode():
            for batch in batch_iterator:
                correct_predictions, samples = (
                    self._get_correct_predictions_and_total_samples_from_batch(batch)
                )
                total_correct_predictions += correct_predictions
                total_samples += samples

        if total_samples == 0:
            return 0.0

        task_accuracy = total_correct_predictions.item() / total_samples

        return task_accuracy

    def _get_correct_predictions_and_total_samples_from_batch(
        self, batch: Batch
    ) -> tuple[Tensor, int]:
        batch.to(self.device)

        X = batch.get_X()
        Y = batch.get_Y()
        queries = batch.get_queries()

        logits, _ = self.model.classify_and_calculate_surrogate_loss_with_L2P(
            X, queries
        )
        predictions = torch.argmax(logits, dim=1)

        num_correct_predictions = (predictions == Y).sum()
        total_samples = Y.shape[0]

        return num_correct_predictions, total_samples
