from reproduction_code.data.fetchers.fetcher_type import FetcherType
from reproduction_code.experiment.core.experiment_runner import ExperimentRunner
from reproduction_code.experiment.core.wandb_config_enums import (
    AblationTagDetail,
    BufferTagDetail,
    DatasetTagDetail,
    EntityDetail,
    GroupDetail,
    ModelTagDetail,
    ProjectDetail,
)

if __name__ == "__main__":
    RUN_NAME = "Gaussian test run 1"

    wandb_log_details = {
        "project": ProjectDetail.L2P_PROJECT,
        "name": RUN_NAME,
        "group": GroupDetail.TEST,
        "entity": EntityDetail.DEFAULT,
        "tags": [
            DatasetTagDetail.GAUSSIAN_CIFAR100,
            ModelTagDetail.L2P,
            BufferTagDetail.NO_BUFFER,
            AblationTagDetail.NO_ABLATION,
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
        "batch_size": 32,
        "epochs_per_task": 1,
        "learning_rate": 0.03,
        "lambda_surrogate": 0.5,
        "rehearsal_samples_per_class": 0,
        "wandb_log_details": wandb_log_details,
    }

    experiment_runner = ExperimentRunner(**run_config)

    print("========== STARTING RUN ==========")
    experiment_runner.execute()
    print("========== RUN SUCCESSFUL ==========")
