import torch
from torch import Tensor, device, nn
from torch.optim import Optimizer

from reproduction_code.data.core.batch import Batch
from reproduction_code.data.core.batch_iterator import BatchIterator
from reproduction_code.data.tasks.task import Task
from reproduction_code.model.forward.L2P_model import L2PModel
from reproduction_code.model.train.optimizer_factory import OptimizerFactory


class SingleTaskTrainer:
    """
    Trains the model on one task at a time, updating the trainable components: prompt vectors, prompt keys and classification head.
    """

    def __init__(
        self,
        model: L2PModel,
        optimizer_factory: OptimizerFactory,
        epochs_per_task: int,
        batch_size: int,
        device: device | str,
        lambda_surrogate: float,
    ) -> None:
        self.model = model
        self.optimizer_factory = optimizer_factory
        self.epochs_per_task = epochs_per_task
        self.batch_size = batch_size
        self.device = device
        self.lambda_surrogate = lambda_surrogate

    def train_on_task(self, task: Task, total_num_classes: int) -> list[float]:
        """
        Performs training on one task for a predetermined number of epochs and records all the epoch losses. Initialize a new Adam optimizer for each task.

        Args:
            task (Task): Task.
            total_num_classes (int): Total number of classes in the current task.

        Returns:
            list[float]: Epoch losses.
        """
        self.model.train()

        # Optimizer created anew. Adam's internal state is reset between tasks
        optimizer = self.optimizer_factory.create_optimizer(self.model)

        current_task_class_ids = task.get_class_ids()
        class_mask = self._build_class_mask(current_task_class_ids, total_num_classes)

        batch_iterator = task.get_train_batches(self.batch_size)

        epoch_losses = self._train_on_all_epochs_and_record_losses(
            batch_iterator, class_mask, optimizer
        )

        return epoch_losses

    def _train_on_all_epochs_and_record_losses(
        self, batch_iterator: BatchIterator, class_mask: Tensor, optimizer: Optimizer
    ) -> list[float]:
        """
        Trains the model in multiple epochs and records all the epoch losses.

        Args:
            batch_iterator (BatchIterator): Iterator for all the batches in one task.
            class_mask (Tensor): Current task's class labels mask.
            optimizer (Optimizer): Optimizer (Adam).

        Returns:
            list[float]: Epoch losses.
        """
        epoch_losses = []

        for _ in range(self.epochs_per_task):
            epoch_loss = self._train_on_epoch_and_calculate_average_loss(
                batch_iterator, class_mask, optimizer
            )
            epoch_losses.append(epoch_loss)

        return epoch_losses

    def _train_on_epoch_and_calculate_average_loss(
        self, batch_iterator: BatchIterator, class_mask: Tensor, optimizer: Optimizer
    ) -> float:
        """
        Trains the model on every batch in one task in one epoch. Then calculates average loss.

        Args:
            batch_iterator (BatchIterator): Iterator for all the batches in one task.
            class_mask (Tensor): Current task's class labels mask.
            optimizer (Optimizer): Optimizer (Adam).

        Returns:
            float: Average epoch loss.
        """
        total_epoch_loss = 0.0
        num_batches = len(batch_iterator)
        assert num_batches > 0

        for batch in batch_iterator:
            batch_loss = self._train_on_batch(batch, class_mask, optimizer)
            total_epoch_loss += batch_loss

        average_epoch_loss = total_epoch_loss / num_batches

        return average_epoch_loss

    def _train_on_batch(
        self, batch: Batch, class_mask: Tensor, optimizer: Optimizer
    ) -> float:
        """
        Trains data on a batch in the following steps:
        - Moves data to desired device, clear gradients.
        - Forwards the data through the complete L2P model, receiving classification logits and surrogate loss.
        - Masks the logits with the current task's class labels mask.
        - Calculates cross entropy loss from logits and batch labels.
        - Calculates total loss, then performs backpropagation and updates parameters.

        Args:
            batch (Batch): Batch.
            class_mask (Tensor): Current task's class labels mask.
            optimizer (Optimizer): Optimizer (Adam).

        Returns:
            float: Total loss value.
        """
        batch.to(self.device)
        self._clear_optimizer_gradients(optimizer)

        X = batch.get_X()
        Y = batch.get_Y()

        logits, surrogate_loss = (
            self.model.classify_and_calculate_surrogate_loss_with_L2P(X)
        )
        masked_logits = self._mask_logits_with_class_mask(logits, class_mask)

        cross_entropy_loss = nn.functional.cross_entropy(masked_logits, Y)
        total_loss = cross_entropy_loss + (self.lambda_surrogate * surrogate_loss)

        self._backpropagation(loss_result=total_loss)
        self._update_parameters_using_optimizer(optimizer)

        total_loss_value = total_loss.item()

        return total_loss_value

    def _build_class_mask(self, class_ids: list[int], total_num_classes: int) -> Tensor:
        mask = torch.zeros(total_num_classes, dtype=torch.bool, device=self.device)
        mask[class_ids] = True

        return mask

    def _backpropagation(self, loss_result: Tensor) -> None:
        loss_result.backward()

    def _update_parameters_using_optimizer(self, optimizer: Optimizer) -> None:
        optimizer.step()

    def _clear_optimizer_gradients(self, optimizer: Optimizer) -> None:
        optimizer.zero_grad()

    def _mask_logits_with_class_mask(
        self, logits: Tensor, class_mask: Tensor
    ) -> Tensor:
        inverted_mask = ~class_mask.unsqueeze(0)
        masked_logits = logits.masked_fill(inverted_mask, float("-inf"))

        return masked_logits
