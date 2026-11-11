import torch
from torch import Tensor, device

from reproduction_code.data.tasks.task import Task
from reproduction_code.model.forward.vit_feature_extractor import ViTFeatureExtractor


class QueryPrecomputer:
    """
    Runs the frozen query ViT once over the data of every task and stores the resulting queries (CLS tokens) inside the tasks.
    """

    def __init__(
        self,
        feature_extractor: ViTFeatureExtractor,
        batch_size: int,
        device: device | str,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.batch_size = batch_size
        self.device = device

    def precompute_for_tasks(self, tasks: list[Task], shared_test_set: bool) -> None:
        """
        Extracts CLS tokens from each batch in the task and caches them inside the task.

        Args:
            tasks (list[Task]): Tasks.
            shared_test_set (bool): Whether every task uses the same test split. This is only true for CORe50.
        """
        for task in tasks:
            task.set_queries(
                "train", self._extract_cls_tokens_from_batches(task, "train")
            )

        if shared_test_set:
            shared_queries = self._extract_cls_tokens_from_batches(tasks[0], "test")
            for task in tasks:
                task.set_queries("test", shared_queries)
        else:
            for task in tasks:
                task.set_queries(
                    "test", self._extract_cls_tokens_from_batches(task, "test")
                )

    def _extract_cls_tokens_from_batches(self, task: Task, split: str) -> Tensor:
        with torch.no_grad():
            batches = task.get_batches(split, self.batch_size, shuffle=False)

            return torch.cat(
                [
                    self.feature_extractor.pass_through_model_and_extract_cls_tokens(
                        batch.to(self.device).get_X()
                    ).cpu()
                    for batch in batches
                ]
            )
