from typing import Any

from reproduction_code.data.fetchers.fetcher_type import FetcherType
from reproduction_code.experiment.core.wandb_config_enums import (
    AblationTagDetail,
    BufferTagDetail,
    DatasetTagDetail,
    EntityDetail,
    FrequencyTableTagDetail,
    GroupDetail,
    ModelTagDetail,
    ProjectDetail,
)


def get_gaussian_CIFAR100_config(run_name: str) -> dict[str, Any]:
    wandb_log_details = {
        "project": ProjectDetail.L2P_PROJECT,
        "name": run_name,
        "group": GroupDetail.TEST,
        "entity": EntityDetail.DEFAULT,
        "tags": [
            DatasetTagDetail.CIFAR100,
            ModelTagDetail.L2P,
            BufferTagDetail.NO_BUFFER,
            AblationTagDetail.NO_ABLATION,
            FrequencyTableTagDetail.NO_FREQUENCY_TABLE,
        ],
        "save_code": True,
    }

    run_config = {
        "random_seed": 42,
        "device": "cuda",
        "data_fetcher": FetcherType.GAUSSIAN_CIFAR100,
        "prompt_pool_size": 10,
        "prompt_length": 5,
        "top_n_prompts": 5,
        "use_frequency_penalty": False,
        "batch_size": 128,
        "epochs_per_task": 1,
        "learning_rate": 0.03,
        "lambda_surrogate": 0.5,
        "grad_clip_max_norm": 1.0,
        "rehearsal_samples_per_class": 0,
        "wandb_log_details": wandb_log_details,
    }

    return run_config
