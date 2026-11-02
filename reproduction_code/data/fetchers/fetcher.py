from abc import ABC, abstractmethod
from typing import Any

from reproduction_code.data.core.data_package import DataPackage


class Fetcher(ABC):
    def get_data(self) -> DataPackage:
        data = self._fetch_data()
        preprocessed_data = self._preprocess(data)

        return preprocessed_data

    @abstractmethod
    def _fetch_data(self) -> Any:
        pass

    @abstractmethod
    def _preprocess(self, data: Any) -> DataPackage:
        pass
