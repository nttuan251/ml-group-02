import wandb
from reproduction_code.data.core.data_package import DataPackage
from reproduction_code.data.tasks.task import Task
from reproduction_code.model.forward.L2P_model import L2PModel
from reproduction_code.model.train.continual_learning_evaluator import (
    ContinualLearningEvaluator,
)
from reproduction_code.model.train.metrics_tracker import MetricsTracker
from reproduction_code.model.train.rehearsal_buffer import RehearsalBuffer
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
        rehearsal_buffer: RehearsalBuffer | None,
    ) -> None:
        self.model = model
        self.data_package = data_package
        self.single_task_trainer = single_task_trainer
        self.evaluator = evaluator
        self.metrics_tracker = metrics_tracker
        self.rehearsal_buffer = rehearsal_buffer

    def train_on_dataset_for_accuracy_and_forgetting(self) -> None:
        """
        Performs training on all tasks of the dataset, at each task:
        - Trains on task and updates parameters.
        - Evaluates current model on all tasks seen including the current one.
        - Records accuracies obtained in the previous step.
        - If prompt frequency penalty is used, updates prompt frequency table.
        - Logs all the accuracies and forgetting values evaluated on all tasks so far using W&B.

        At the end, logs the final model's average accuracy on all tasks, as well as average forgetting value across all tasks.
        """
        num_tasks = self.data_package.get_number_of_tasks()

        for current_task_index in range(num_tasks):
            print(f"----- TASK {current_task_index} -----")

            print("=== TRAINING ===")
            self._train_on_current_task_and_update_rehearsal(current_task_index)

            print("=== EVALUATING ===")
            accuracies_of_all_tasks_up_to_current = (
                self._calculate_accuracies_on_all_tasks_up_to_index(current_task_index)
            )

            self._record_accuracies_after_training_on_current_task(
                accuracies_of_all_tasks_up_to_current,
                current_task_index,
            )

            self._update_prompt_frequencies_if_enabled()

            self._log_accuracies_and_forgetting(
                current_task_index,
                accuracies_of_all_tasks_up_to_current,
                is_final_task=current_task_index == num_tasks - 1,
            )

    def _log_accuracies_and_forgetting(
        self,
        current_task_index: int,
        accuracies_of_all_tasks_up_to_current: dict[int, float],
        is_final_task: bool,
    ) -> None:
        log_data: dict[str, int | float] = {"current_task_index": current_task_index}

        for task_index, accuracy in accuracies_of_all_tasks_up_to_current.items():
            log_data[f"accuracy/task_{task_index}"] = accuracy

            if task_index < current_task_index:
                forgetting = self.metrics_tracker._get_forgetting_on_taskA_after_training_on_task_B(
                    taskA_index=task_index, taskB_index=current_task_index
                )

                print(f"Forgetting on task {task_index}: {forgetting}")
                log_data[f"forgetting/task_{task_index}"] = forgetting

        wandb.log(log_data)

        if is_final_task:
            final_average_accuracy = self.metrics_tracker.get_average_accuracy_on_all_tasks_including_current(
                current_trained_task_index=current_task_index
            )
            final_average_forgetting = (
                self.metrics_tracker.get_average_forgetting_on_all_tasks_so_far(
                    current_trained_task_index=current_task_index
                )
            )

            print("----- FINAL EVALUATION -----")
            print(f"Final average accuracy: {final_average_accuracy}")
            print(f"Final average forgetting: {final_average_forgetting}")

            wandb.summary["final_average_accuracy"] = final_average_accuracy
            wandb.summary["final_average_forgetting"] = final_average_forgetting

    def _train_on_current_task_and_update_rehearsal(
        self, current_task_index: int
    ) -> None:
        """
        Trains on current task and updates rehearsal buffer if it is being used.

        Args:
            current_task_index (int): _description_
        """
        current_task = self.data_package.get_task(current_task_index)
        num_classes = self.data_package.get_number_of_classes()

        self.single_task_trainer.train_on_task(current_task, num_classes)

        if self.rehearsal_buffer:
            self.rehearsal_buffer.fetch_and_save_samples_from_task(current_task)

    def _calculate_accuracies_on_all_tasks_up_to_index(
        self,
        current_task_index: int,
    ) -> dict[int, float]:
        print(f"** AFTER TRAINING ON TASK {current_task_index} **")

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
