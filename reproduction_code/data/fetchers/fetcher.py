from abc import ABC, abstractmethod
from typing import Any

from reproduction_code.data.core.data_package import DataPackage


class Fetcher(ABC):
    """
    Classes that inherit this class are meant to be used as interfaces to fetch different datasets from different external sources.
    The data is preprocessed to ensure it follows certain contracts established inside this program (see docstring in _preprocess), before being loaded into a DataPackage object to be returned.
    """

    def __init__(self, random_seed: int) -> None:
        self.random_seed = random_seed

    def get_data(self) -> DataPackage:
        """
        Fetches and preprocesses data into a DataPackage before returning.

        Returns:
            DataPackage: Object containing all the necessary information inside the fetched dataset.
        """
        data = self._fetch_data()
        preprocessed_data = self._preprocess(data)

        return preprocessed_data

    @abstractmethod
    def _fetch_data(self) -> Any:
        """
        Fetches dataset from different sources, depending on the child class.

        Returns:
            Any: The raw data returned right after fetching from the external source.
        """
        pass

    @abstractmethod
    def _preprocess(self, data: Any) -> DataPackage:
        """
        Preprocesses data into a DataPackage. The resulting data will always have the two splits: "train" and "test". Each split always has the columns "X", "Y" corresponding to features and labels.
        Different datasets will also have different preprocessing steps.

        Args:
            data (Any): The raw data from the _fetch_data function.

        Returns:
            DataPackage: The preprocessed data packaged into a DataPackage object.
        """
        pass
