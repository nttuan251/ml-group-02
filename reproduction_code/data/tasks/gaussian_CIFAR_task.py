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
        return self.get_batches("train", batch_size, shuffle=False)
