from typing import ClassVar

import torch
from torch import Tensor, device


class Batch:
    """
    A batch of data (label included). Batches are packaged and accessed via a BatchIterator object.
    """

    IMAGE_SHAPE: ClassVar[tuple[int, int, int]] = (3, 224, 224)

    def __init__(self, X: Tensor, Y: Tensor) -> None:
        self._assert_valid_X(X)
        self._assert_valid_Y(Y)
        self._assert_matching_batch_sizes(X, Y)

        self.X = X
        self.Y = Y

    def to(self, device: device | str) -> "Batch":
        """
        Moves the batch's data to the specified device, used when moving data to a GPU for computation or back to the CPU.


        Args:
            device (device | str): The target device to which data will be moved (example: "cpu", "cuda",...)

        Returns:
            Batch: Current Batch instance with data moved to the specified device.
        """
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)
        return self

    def get_X(self) -> Tensor:
        return self.X

    def get_Y(self) -> Tensor:
        return self.Y

    def _assert_valid_X(self, X: Tensor) -> None:
        assert X.ndim == 4
        assert tuple(X.shape[1:]) == self.IMAGE_SHAPE
        assert X.dtype == torch.float32

    def _assert_valid_Y(self, Y: Tensor) -> None:
        assert Y.ndim == 1
        assert Y.dtype == torch.long

    def _assert_matching_batch_sizes(self, X: Tensor, Y: Tensor) -> None:
        assert X.shape[0] == Y.shape[0]
