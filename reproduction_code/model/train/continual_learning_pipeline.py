from reproduction_code.data.core.data_package import DataPackage
from reproduction_code.data.tasks.task import Task
from reproduction_code.model.forward.L2P_model import L2PModel
from reproduction_code.model.train.continual_learning_evaluator import (
    ContinualLearningEvaluator,
)
from reproduction_code.model.train.metrics_tracker import MetricsTracker
from reproduction_code.model.train.single_task_trainer import SingleTaskTrainer


class ContinualLearningPipeline:
    """
    The complete L2P training and evaluating pipeline.
    """

    def __init__(
        self,
        model: L2PModel,
        data_package: DataPackage,
        single_task_trainer: SingleTaskTrainer,
        evaluator: ContinualLearningEvaluator,
        metrics_tracker: MetricsTracker,
    ) -> None:
        self.model = model
        self.data_package = data_package
        self.single_task_trainer = single_task_trainer
        self.evaluator = evaluator
        self.metrics_tracker = metrics_tracker

    def train_on_dataset_for_accuracy_and_forgetting(self) -> tuple[float, float]:
        """
        Performs training on all tasks of the dataset, at each task:
        - Trains on task and updates parameters.
        - Evaluates current model on all tasks seen including the current one.
        - Records accuracies obtained in the previous step.
        - If prompt frequency penalty is used, updates prompt frequency table.

        At the end, calculates final model's average accuracy on all tasks, as well as average forgetting value across all tasks.

        Returns:
            tuple[float, float]: Average accuracy and average forgetting value.
        """
        num_tasks = self.data_package.get_number_of_tasks()

        for current_task_index in range(num_tasks):
            self._train_on_current_task(current_task_index)

            accuracies_of_all_tasks_up_to_current = (
                self._calculate_accuracies_on_all_tasks_up_to_index(current_task_index)
            )

            self._record_accuracies_after_training_on_current_task(
                accuracies_of_all_tasks_up_to_current,
                current_task_index,
            )

            self._update_prompt_frequencies_if_enabled()

        final_task_index = num_tasks - 1
        final_average_accuracy = (
            self.metrics_tracker.get_average_accuracy_on_all_tasks_including_current(
                final_task_index
            )
        )
        final_average_forgetting = (
            self.metrics_tracker.get_average_forgetting_on_all_tasks_so_far(
                final_task_index
            )
        )

        return final_average_accuracy, final_average_forgetting

    def _train_on_current_task(self, current_task_index: int) -> None:
        current_task = self.data_package.get_task(current_task_index)
        num_classes = self.data_package.get_number_of_classes()

        self.single_task_trainer.train_on_task(current_task, num_classes)

    def _calculate_accuracies_on_all_tasks_up_to_index(
        self,
        current_task_index: int,
    ) -> dict[int, float]:
        task_indices_up_to_current = self._get_all_tasks_up_to_index(current_task_index)
        task_accuracies = self.evaluator.calculate_accuracies_on_each_task(
            task_indices_up_to_current
        )

        return task_accuracies

    def _record_accuracies_after_training_on_current_task(
        self,
        task_accuracies: dict[int, float],
        current_task_index: int,
    ) -> None:
        for evaluated_task_index, accuracy in task_accuracies.items():
            self.metrics_tracker.record_accuracy_on_taskB_after_training_on_taskA(
                taskA=current_task_index,
                taskB=evaluated_task_index,
                accuracy=accuracy,
            )

    def _get_all_tasks_up_to_index(self, current_task_index: int) -> list[Task]:
        seen_tasks = self.data_package.tasks[: current_task_index + 1]

        return seen_tasks

    def _update_prompt_frequencies_if_enabled(self) -> None:
        if self.model.prompt_pool.use_frequency_penalty:
            self.model.prompt_pool.update_frequency_after_task()
