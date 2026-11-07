from reproduction_code.data.core.batch_iterator import BatchIterator
from reproduction_code.data.tasks.CIFAR_task import CIFARTask


class GaussianCIFARTask(CIFARTask):
    """
    A task class designed for the CIFAR-100 dataset (Gaussian version).
    Information already covered in the docstrings inside the base Task class won't be specified again.
    """

    def get_train_batches(self, batch_size: int) -> BatchIterator:
        """
        Similar to the base class get_train_batches, except shuffling is always turned off to preserve Gaussian scheduled data stream order.

        Args:
            batch_size (int): Batch size.

        Returns:
            BatchIterator: Returned batches.
        """
        raw_train_dataset = self.data["train"]

        dataset_with_transformer = self._apply_lazy_transformation(raw_train_dataset)

        return self._build_batch_iterator(dataset_with_transformer, batch_size, False)
