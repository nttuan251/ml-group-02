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
        self.scaler = torch.amp.GradScaler("cuda")

    def train_on_task(self, task: Task, total_num_classes: int) -> list[float]:
        """
        Performs training on one task for a predetermined number of epochs and records all the epoch losses. Initialize a new optimizer for each task.

        Args:
            task (Task): Task.
            total_num_classes (int): Total number of classes in the dataset.

        Returns:
            list[float]: Epoch losses.
        """
        self.model.train()

        # Optimizer created anew, internal states are reset every task.
        optimizer = self.optimizer_factory.create_optimizer(self.model)

        seen_class_ids = self._build_seen_class_ids(task)
        class_mask = self._build_class_mask(list(seen_class_ids), total_num_classes)

        batch_iterator = task.get_train_batches(self.batch_size)

        epoch_losses = self._train_on_all_epochs_and_record_losses(
            batch_iterator, class_mask, optimizer
        )

        return epoch_losses

    def _build_seen_class_ids(self, task: Task) -> list[int]:
        current_task_class_ids = task.get_class_ids()

        if self.rehearsal_buffer:
            rehearsal_buffer_class_ids: set[int] = (
                self.rehearsal_buffer.get_recorded_classes()
            )
        else:
            rehearsal_buffer_class_ids: set[int] = set()

        seen_class_ids = rehearsal_buffer_class_ids.union(set(current_task_class_ids))

        return list(seen_class_ids)

    def _train_on_all_epochs_and_record_losses(
        self, batch_iterator: BatchIterator, class_mask: Tensor, optimizer: Optimizer
    ) -> list[float]:
        """
        Trains the model in multiple epochs and records all the epoch losses.

        Args:
            batch_iterator (BatchIterator): Iterator for all the batches in one task.
            class_mask (Tensor): Current task's class labels mask.
            optimizer (Optimizer): Optimizer (Adam or SGD).

        Returns:
            list[float]: Epoch losses.
        """
        epoch_losses = []

        for epoch_index in range(self.epochs_per_task):
            print(f"** EPOCH {epoch_index} **")

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
            optimizer (Optimizer): Optimizer (Adam or SGD).

        Returns:
            float: Average epoch loss.
        """
        total_epoch_loss = torch.zeros((), device=self.device)

        for batch_index, batch in enumerate(batch_iterator):
            batch_loss = self._train_on_batch(batch, class_mask, optimizer)
            total_epoch_loss += batch_loss

            if batch_index % 10 == 0 or batch_index == len(batch_iterator) - 1:
                print(f"Batch {batch_index} loss: {batch_loss.item()}")

        num_batches = len(batch_iterator)
        assert num_batches > 0
        average_epoch_loss = (total_epoch_loss / num_batches).item()

        return average_epoch_loss

    def _train_on_batch(
        self, batch: Batch, class_mask: Tensor, optimizer: Optimizer
    ) -> Tensor:
        """
        Trains data on a batch in the following steps:
        - Moves data to desired device, clear gradients.
        - Forwards the data through the complete L2P model, receiving classification logits and surrogate loss.
        - Masks the logits with the current task's class labels mask.
        - Calculates cross entropy loss from logits and batch labels.
        - Calculates total loss, then performs backpropagation and updates parameters.

        If rehearsal buffer is used and buffer is not empty, an extra batch of past data is appended to the current batch.
        The final batch is always divided into batch-sized chunks to process at a time for fear of OOM issues when rehearsal buffer is used, in which case the batch size increases twofold.

        Args:
            batch (Batch): Batch.
            class_mask (Tensor): Current task's class labels mask.
            optimizer (Optimizer): Optimizer (Adam or SGD).

        Returns:
            Tensor: Total loss value.
        """
        batch.to(self.device)
        self._clear_optimizer_gradients(optimizer)

        if self.rehearsal_buffer and not self.rehearsal_buffer.is_empty():
            rehearsal_batch = self.rehearsal_buffer.get_random_samples(
                self.batch_size
            ).to(self.device)
            batch = batch.merge(rehearsal_batch)

        X = batch.get_X()
        Y = batch.get_Y()
        queries = batch.get_queries()
        total_samples = batch.get_batch_size()

        total_loss_value = torch.zeros((), device=self.device)
        for start_index in range(0, total_samples, self.batch_size):
            X_chunk = X[start_index : start_index + self.batch_size]
            Y_chunk = Y[start_index : start_index + self.batch_size]
            queries_chunk = queries[start_index : start_index + self.batch_size]

            with torch.autocast("cuda", dtype=torch.float16):
                logits, surrogate_loss = (
                    self.model.classify_and_calculate_surrogate_loss_with_L2P(
                        X_chunk, queries_chunk
                    )
                )
            masked_logits = self._mask_logits_with_class_mask(
                logits.float(), class_mask
            )

            cross_entropy_loss = nn.functional.cross_entropy(masked_logits, Y_chunk)
            chunk_loss = cross_entropy_loss + (self.lambda_surrogate * surrogate_loss)

            chunk_weight = X_chunk.shape[0] / total_samples
            self._backpropagation(loss_result=chunk_loss * chunk_weight)

            total_loss_value += (chunk_loss * chunk_weight).detach()

        self.scaler.unscale_(optimizer)
        self._clip_gradients()

        self._update_parameters_using_optimizer(optimizer)
        self.scaler.update()

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

    def _build_class_mask(self, class_ids: list[int], total_num_classes: int) -> Tensor:
        mask = torch.zeros(total_num_classes, dtype=torch.bool, device=self.device)
        mask[class_ids] = True

        return mask

    def _backpropagation(self, loss_result: Tensor) -> None:
        self.scaler.scale(loss_result).backward()

    def _update_parameters_using_optimizer(self, optimizer: Optimizer) -> None:
        self.scaler.step(optimizer)

    def _clear_optimizer_gradients(self, optimizer: Optimizer) -> None:
        optimizer.zero_grad()

    def _mask_logits_with_class_mask(
        self, logits: Tensor, class_mask: Tensor
    ) -> Tensor:
        inverted_mask = ~class_mask.unsqueeze(0)
        masked_logits = logits.masked_fill(inverted_mask, float("-inf"))

        return masked_logits
