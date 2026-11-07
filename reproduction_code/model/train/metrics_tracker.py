class MetricsTracker:
    """
    Records a table accuracies where accuracies[i][j] is the accuracy of the model evaluated on task j after just having trained on task i.
    """

    def __init__(self, num_tasks: int) -> None:
        self.num_tasks = num_tasks
        self.accuracies = [[0.0 for _ in range(num_tasks)] for _ in range(num_tasks)]

    def record_accuracy_on_taskB_after_training_on_taskA(
        self, taskA: int, taskB: int, accuracy: float
    ) -> None:
        self.accuracies[taskA][taskB] = accuracy

    def get_average_accuracy_on_all_tasks_including_current(
        self,
        current_trained_task_index: int,
    ) -> float:
        """
        Calculates the average accuracy evaluated on all tasks up to the current task index (inclusive) after having trained on the current task.

        Args:
            current_trained_task_index (int): Index of the most recently trained on task.

        Returns:
            float: Average accuracy.
        """
        total_accuracy = 0.0
        num_seen_tasks = current_trained_task_index + 1

        for evaluated_task_index in range(num_seen_tasks):
            total_accuracy += self.accuracies[current_trained_task_index][
                evaluated_task_index
            ]

        return total_accuracy / num_seen_tasks

    def get_average_forgetting_on_all_tasks_so_far(
        self,
        current_trained_task_index: int,
    ) -> float:
        """
        Calculates the average of all the forgetting values on all past tasks after having trained on the current task.

        Args:
            current_trained_task_index (int): Index of the most recently trained on task.

        Returns:
            float: Average forgetting value.
        """
        if current_trained_task_index == 0:
            return 0.0

        forgetting_values = self._get_forgetting_values_so_far(
            current_trained_task_index
        )

        return sum(forgetting_values) / len(forgetting_values)

    def _get_forgetting_values_so_far(
        self,
        current_trained_task_index: int,
    ) -> list[float]:
        """
        Gets all the forgetting values on all past tasks after having trained on the current task.

        Args:
            current_trained_task_index (int): Index of the most recently trained on task.

        Returns:
            list[float]: Forgetting values.
        """
        return [
            self._get_forgetting_on_taskA_after_training_on_task_B(
                taskA_index=evaluated_task_index,
                taskB_index=current_trained_task_index,
            )
            for evaluated_task_index in range(current_trained_task_index)
        ]

    def _get_forgetting_on_taskA_after_training_on_task_B(
        self,
        taskA_index: int,
        taskB_index: int,
    ) -> float:
        """
        Calculates forgetting value on task A after having trained on task B, which is the difference between the peak accuracy ever evaluated on task A and the current model's accuracy on task A.

        Args:
            taskA_index (int): Index of task A.
            taskB_index (int): Index of task B.

        Returns:
            float: Forgetting value.
        """
        peak_accuracy = self._get_peak_accuracy_on_task_A_after_training_up_to_task_B(
            taskA_index=taskA_index,
            taskB_index=taskB_index - 1,
        )
        current_accuracy = self.accuracies[taskB_index][taskA_index]

        return peak_accuracy - current_accuracy

    def _get_peak_accuracy_on_task_A_after_training_up_to_task_B(
        self, taskA_index: int, taskB_index: int
    ) -> float:
        """
        Gets the peak accuracy ever evaluated on task A up to the point of having trained on task B.

        Args:
            taskA_index (int): Index of task A.
            taskB_index (int): Index of task B.

        Returns:
            float: Peak accuracy.
        """
        peak_accuracy = 0.0

        for trained_task_index in range(taskA_index, taskB_index + 1):
            accuracy = self.accuracies[trained_task_index][taskA_index]

            if accuracy > peak_accuracy:
                peak_accuracy = accuracy

        return peak_accuracy
