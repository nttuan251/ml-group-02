from reproduction_code.data.tasks.task import Task


class DataPackage:
    """
    Contains all the training data of a dataset and its number of classes.
    Training data is divided into tasks. Each task contains its own train and test data. Tasks then yield their data through a BatchIterator object with customizable batch size.
    """

    def __init__(self, tasks: list[Task], num_classes: int) -> None:
        self.tasks = tasks
        self.num_classes = num_classes

    def get_task(self, task_index: int) -> Task:
        return self.tasks[task_index]

    def get_tasks(self) -> list[Task]:
        return self.tasks

    def get_number_of_tasks(self) -> int:
        return len(self.tasks)

    def get_number_of_classes(self) -> int:
        return self.num_classes
