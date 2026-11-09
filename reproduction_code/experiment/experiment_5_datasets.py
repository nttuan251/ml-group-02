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
    RUN_NAME = ""

    wandb_log_details = {
        "project": ProjectDetail.L2P_PROJECT,
        "name": RUN_NAME,
        "group": GroupDetail.TEST,
        "entity": EntityDetail.DEFAULT,
        "tags": [
            DatasetTagDetail.FIVE_DATASETS,
            ModelTagDetail.L2P,
            BufferTagDetail.NO_BUFFER,
            AblationTagDetail.NO_ABLATION,
        ],
        "save_code": True,
    }

    run_config = {
        "random_seed": 42,
        "device": "cuda",
        "data_fetcher": FetcherType.FIVE_DATASETS,
        "prompt_pool_size": 20,
        "prompt_length": 5,
        "top_n_prompts": 4,
        "use_frequency_penalty": True,
        "batch_size": 128,
        "epochs_per_task": 5,
        "learning_rate": 0.03,
        "lambda_surrogate": 0.5,
        "rehearsal_samples_per_class": 0,
        "wandb_log_details": wandb_log_details,
    }

    experiment_runner = ExperimentRunner(**run_config)

    print("========== STARTING RUN ==========")
    experiment_runner.execute()
    print("========== RUN SUCCESSFUL ==========")
