from reproduction_code.data.core.batch_iterator import BatchIterator
from reproduction_code.data.tasks.CIFAR_task import CIFARTask


class GaussianCIFARTask(CIFARTask):
    def get_train_batches(self, batch_size: int) -> BatchIterator:
        raw_train_dataset = self.data["train"]

        dataset_with_transformer = self._apply_lazy_transformer(raw_train_dataset)

        return self._make_batches(dataset_with_transformer, batch_size, False)
