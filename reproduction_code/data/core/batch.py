from typing import ClassVar

import torch
from torch import Tensor, device


class Batch:
    """
    A batch of data (label included).
    Can carry queries (CLS tokens) after computed via a ViT model.
    """

    IMAGE_SHAPE: ClassVar[tuple[int, int, int]] = (3, 224, 224)

    def __init__(self, X: Tensor, Y: Tensor, queries: Tensor | None = None) -> None:
        self._assert_valid_X(X)
        self._assert_valid_Y(Y)
        self._assert_matching_batch_sizes(X, Y)
        if queries is not None:
            self._assert_valid_queries(queries, X)

        self.X = X
        self.Y = Y
        self.queries = queries

    def to(self, device: device | str) -> "Batch":
        """
        Moves the batch's data to the specified device, used when moving data to a GPU for computation or back to the CPU.


        Args:
            device (device | str): The target device to which data will be moved (example: "cpu", "cuda",...)

        Returns:
            Batch: Current Batch instance with data moved to the specified device.
        """
        self.X = self.X.to(device, non_blocking=True)
        self.Y = self.Y.to(device, non_blocking=True)
        if self.queries is not None:
            self.queries = self.queries.to(device, non_blocking=True)
        return self

    def get_X(self) -> Tensor:
        return self.X

    def get_Y(self) -> Tensor:
        return self.Y

    def get_queries(self) -> Tensor:
        assert self.queries is not None
        return self.queries

    def select(self, indices: Tensor) -> "Batch":
        return Batch(
            self.X[indices],
            self.Y[indices],
            self.get_queries()[indices],
        )

    def pin_memory(self) -> "Batch":
        self.X = self.X.pin_memory()
        self.Y = self.Y.pin_memory()
        if self.queries is not None:
            self.queries = self.queries.pin_memory()
        return self

    def get_batch_size(self) -> int:
        return self.Y.shape[0]

    def merge(self, other: "Batch") -> "Batch":
        """
        Merges the current instance with another Batch instance after moving it to the same device as the current.

        Args:
            other (Batch): Other Batch instance.

        Returns:
            Batch: Merged Batch instance.
        """

        other_X = other.get_X().to(self.X.device, non_blocking=True)
        other_Y = other.get_Y().to(self.Y.device, non_blocking=True)
        other_queries = other.get_queries().to(
            self.get_queries().device, non_blocking=True
        )

        merged_X = torch.cat((self.X, other_X), dim=0)
        merged_Y = torch.cat((self.Y, other_Y), dim=0)
        merged_queries = torch.cat((self.get_queries(), other_queries), dim=0)

        return Batch(
            X=merged_X,
            Y=merged_Y,
            queries=merged_queries,
        )

    def _assert_valid_X(self, X: Tensor) -> None:
        assert X.ndim == 4
        assert tuple(X.shape[1:]) == self.IMAGE_SHAPE
        assert X.dtype == torch.float32

    def _assert_valid_Y(self, Y: Tensor) -> None:
        assert Y.ndim == 1
        assert Y.dtype == torch.long

    def _assert_matching_batch_sizes(self, X: Tensor, Y: Tensor) -> None:
        assert X.shape[0] == Y.shape[0]

    def _assert_valid_queries(self, queries: Tensor, X: Tensor) -> None:
        assert queries.ndim == 2
        assert queries.shape[0] == X.shape[0]
        assert queries.dtype == torch.float32
