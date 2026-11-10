import torch
from torch import Tensor, device, nn
from torch.optim import Optimizer

from reproduction_code.data.core.batch import Batch
from reproduction_code.data.core.batch_iterator import BatchIterator
from reproduction_code.data.tasks.task import Task
from reproduction_code.model.forward.L2P_model import L2PModel
from reproduction_code.model.train.optimizer_factory import OptimizerFactory
from reproduction_code.model.train.rehearsal_buffer import RehearsalBuffer


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
        grad_clip_max_norm: float,
        rehearsal_buffer: RehearsalBuffer | None,
    ) -> None:
        self.model = model
        self.optimizer_factory = optimizer_factory
        self.epochs_per_task = epochs_per_task
        self.batch_size = batch_size
        self.device = device
        self.lambda_surrogate = lambda_surrogate
        self.grad_clip_max_norm = grad_clip_max_norm
        self.rehearsal_buffer = rehearsal_buffer

    def train_on_task(self, task: Task, total_num_classes: int) -> list[float]:
        """
        Performs training on one task for a predetermined number of epochs and records all the epoch losses. Initialize a new Adam optimizer for each task.

        Args:
            task (Task): Task.
            total_num_classes (int): Total number of classes in the current dataset.

        Returns:
            list[float]: Epoch losses.
        """
        self.model.train()

        # Optimizer created anew. Adam's internal state is reset between tasks
        optimizer = self.optimizer_factory.create_optimizer(self.model)

        current_task_class_mask = self._build_class_mask(
            task.get_class_ids(), total_num_classes
        )
        rehearsal_class_mask = self._build_rehearsal_class_mask(total_num_classes)

        batch_iterator = task.get_train_batches(self.batch_size)

        epoch_losses = self._train_on_all_epochs_and_record_losses(
            batch_iterator,
            current_task_class_mask,
            rehearsal_class_mask,
            optimizer,
        )

        return epoch_losses

    def _train_on_all_epochs_and_record_losses(
        self,
        batch_iterator: BatchIterator,
        current_task_class_mask: Tensor,
        rehearsal_class_mask: Tensor | None,
        optimizer: Optimizer,
    ) -> list[float]:
        """
        Trains the model in multiple epochs and records all the epoch losses.

        Args:
            batch_iterator (BatchIterator): Iterator for all the batches in one task.
            current_task_class_mask (Tensor): Current task's class labels mask.
            rehearsal_class_mask (Tensor | None): Rehearsal buffer class labels mask.
            optimizer (Optimizer): Optimizer (Adam).

        Returns:
            list[float]: Epoch losses.
        """
        epoch_losses = []

        for epoch_index in range(self.epochs_per_task):
            print(f"** EPOCH {epoch_index} **")

            epoch_loss = self._train_on_epoch_and_calculate_average_loss(
                batch_iterator,
                current_task_class_mask,
                rehearsal_class_mask,
                optimizer,
            )
            epoch_losses.append(epoch_loss)

        return epoch_losses

    def _train_on_epoch_and_calculate_average_loss(
        self,
        batch_iterator: BatchIterator,
        current_task_class_mask: Tensor,
        rehearsal_class_mask: Tensor | None,
        optimizer: Optimizer,
    ) -> float:
        """
        Trains the model on every batch in one task in one epoch. Then calculates average loss.

        Args:
            batch_iterator (BatchIterator): Iterator for all the batches in one task.
            current_task_class_mask (Tensor): Current task's class labels mask.
            rehearsal_class_mask (Tensor | None): Rehearsal buffer class labels mask.
            optimizer (Optimizer): Optimizer (Adam).

        Returns:
            float: Average epoch loss.
        """
        total_epoch_loss = 0.0
        num_batches = len(batch_iterator)
        assert num_batches > 0

        latest_batch_start_index = 0

        for batch_index, batch in enumerate(batch_iterator):
            batch_loss = self._train_on_batch(
                batch,
                current_task_class_mask,
                rehearsal_class_mask,
                optimizer,
            )
            total_epoch_loss += batch_loss

            current_batch_size = batch.get_batch_size()
            current_batch_end_index = latest_batch_start_index + current_batch_size
            print(
                f"Batch {batch_index} ({latest_batch_start_index}-{current_batch_end_index}) loss: {batch_loss}"
            )
            latest_batch_start_index += self.batch_size

        average_epoch_loss = total_epoch_loss / num_batches

        return average_epoch_loss

    def _train_on_batch(
        self,
        batch: Batch,
        current_task_class_mask: Tensor,
        rehearsal_class_mask: Tensor | None,
        optimizer: Optimizer,
    ) -> float:
        """
        Trains data on a batch in the following steps:
        - Moves data to desired device, clear gradients.
        - If rehearsal is used and not empty, samples a batch with the same batch size from it. The current batch and the rehearsal batch are kept as two separate chunks.
        - Forwards the data through the complete L2P model, receiving classification logits and surrogate loss.
        - Masks each chunk's logits with its own class mask.
        - Calculates cross entropy loss from logits and chunk labels.
        - Calculates total loss and scales it by the chunk's portion of total samples, then performs backpropagation, clips gradients and updates parameters.

        Args:
            batch (Batch): Batch.
            current_task_class_mask (Tensor): Current task's class labels mask.
            rehearsal_class_mask (Tensor | None): Rehearsal buffer class labels mask.
            optimizer (Optimizer): Optimizer (Adam).

        Returns:
            float: Total loss value.
        """
        batch.to(self.device)
        self._clear_optimizer_gradients(optimizer)

        chunks_of_batch_and_mask = [(batch, current_task_class_mask)]

        if rehearsal_class_mask is not None:
            assert self.rehearsal_buffer is not None
            rehearsal_batch = self.rehearsal_buffer.get_random_samples(
                self.batch_size
            ).to(self.device)
            chunks_of_batch_and_mask.append((rehearsal_batch, rehearsal_class_mask))

        total_samples = sum(
            chunk_batch.get_batch_size() for chunk_batch, _ in chunks_of_batch_and_mask
        )

        total_loss_value = 0.0
        for chunk_batch, chunk_class_mask in chunks_of_batch_and_mask:
            X_chunk = chunk_batch.get_X()
            Y_chunk = chunk_batch.get_Y()

            logits, surrogate_loss = (
                self.model.classify_and_calculate_surrogate_loss_with_L2P(X_chunk)
            )
            masked_logits = self._mask_logits_with_class_mask(logits, chunk_class_mask)

            cross_entropy_loss = nn.functional.cross_entropy(masked_logits, Y_chunk)
            chunk_loss = cross_entropy_loss + (self.lambda_surrogate * surrogate_loss)

            chunk_weight = X_chunk.shape[0] / total_samples
            self._backpropagation(loss_result=chunk_loss * chunk_weight)

            total_loss_value += chunk_loss.item() * chunk_weight

        self._clip_gradients()
        self._update_parameters_using_optimizer(optimizer)

        return total_loss_value

    def _clip_gradients(self) -> None:
        trainable_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        torch.nn.utils.clip_grad_norm_(
            trainable_parameters, max_norm=self.grad_clip_max_norm
        )

    def _build_rehearsal_class_mask(self, total_num_classes: int) -> Tensor | None:
        if self.rehearsal_buffer and not self.rehearsal_buffer.is_empty():
            rehearsal_class_ids = self.rehearsal_buffer.get_recorded_classes()
            rehearsal_class_mask = self._build_class_mask(
                list(rehearsal_class_ids), total_num_classes
            )
        else:
            rehearsal_class_mask = None

        return rehearsal_class_mask

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
