from enum import Enum


class FetcherType(str, Enum):
    CIFAR100 = "CIFAR100"
    GAUSSIAN_CIFAR100 = "GAUSSIAN_CIFAR100"
    CORE50 = "CORE50"
    FIVE_DATASETS = "FIVE_DATASETS"
