from enum import Enum


class ProjectDetail(str, Enum):
    L2P_PROJECT = "L2P_PROJECT"


class GroupDetail(str, Enum):
    OFFICIAL = "OFFICIAL"
    TEST = "TEST"


class EntityDetail(str, Enum):
    DEFAULT = "nhuy165-org"


class DatasetTagDetail(str, Enum):
    CIFAR100 = "CIFAR100"
    GAUSSIAN_CIFAR100 = "GAUSSIAN_CIFAR100"
    CORE50 = "CORE50"
    FIVE_DATASETS = "FIVE_DATASETS"


class ModelTagDetail(str, Enum):
    L2P = "L2P"


class BufferTagDetail(str, Enum):
    BUFFER = "BUFFER"
    NO_BUFFER = "NO_BUFFER"


class AblationTagDetail(str, Enum):
    PROMPT_POOL_ABLATION = "PROMPT_POOL_ABLATION"
    KEY_VALUE_PAIR_ABLATION = "KEY_VALUE_PAIR_ABLATION"
    DIVERSIFIED_SELECTION_ABLATION = "DIVERSIFIED_SELECTION_ABLATION"
    NO_ABLATION = "NO_ABLATION"
