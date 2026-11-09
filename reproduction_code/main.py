from reproduction_code.experiment.core.experiment_runner import ExperimentRunner
from reproduction_code.experiment.experiment_CIFAR100 import get_CIFAR100_config

if __name__ == "__main__":
    RUN_NAME = ""

    run_config = get_CIFAR100_config(RUN_NAME)
    # run_config = get_5_datasets_config(RUN_NAME)
    # run_config = get_CORe50_config(RUN_NAME)
    # run_config = get_gaussian_CIFAR100_config(RUN_NAME)

    experiment_runner = ExperimentRunner(**run_config)

    print("========== STARTING RUN ==========")
    experiment_runner.execute()
    print("========== RUN SUCCESSFUL ==========")
