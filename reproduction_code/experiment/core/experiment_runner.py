from typing import Any, ClassVar

from torch import device

import wandb
from reproduction_code.data.core.data_package import DataPackage
from reproduction_code.data.fetchers.CIFAR_fetcher import CIFARFetcher
from reproduction_code.data.fetchers.CORe50_fetcher import CORe50Fetcher
from reproduction_code.data.fetchers.fetcher import Fetcher
from reproduction_code.data.fetchers.fetcher_type import FetcherType
from reproduction_code.data.fetchers.five_datasets_fetcher import FiveDatasetsFetcher
from reproduction_code.data.fetchers.gaussian_CIFAR_fetcher import GaussianCIFARFetcher
from reproduction_code.experiment.core.random_seed import set_global_seed
from reproduction_code.model.forward.L2P_model import L2PModel
from reproduction_code.model.train.continual_learning_evaluator import (
    ContinualLearningEvaluator,
)
from reproduction_code.model.train.continual_learning_pipeline import (
    ContinualLearningPipeline,
)
from reproduction_code.model.train.metrics_tracker import MetricsTracker
from reproduction_code.model.train.optimizer_factory import (
    OptimizerFactory,
)
from reproduction_code.model.train.query_precomputer import QueryPrecomputer
from reproduction_code.model.train.rehearsal_buffer import RehearsalBuffer
from reproduction_code.model.train.single_task_trainer import SingleTaskTrainer


class ExperimentRunner:
    FETCHER_MAP: ClassVar[dict[FetcherType, type[Fetcher]]] = {
        FetcherType.CIFAR100: CIFARFetcher,
        FetcherType.CORE50: CORe50Fetcher,
        FetcherType.FIVE_DATASETS: FiveDatasetsFetcher,
        FetcherType.GAUSSIAN_CIFAR100: GaussianCIFARFetcher,
    }

    def __init__(
        self,
        random_seed: int,
        device: device | str,
        data_fetcher: FetcherType,
        prompt_pool_size: int,
        prompt_length: int,
        top_n_prompts: int,
        use_frequency_penalty: bool,
        batch_size: int,
        epochs_per_task: int,
        learning_rate: float,
        lambda_surrogate: float,
        grad_clip_max_norm: float,
        rehearsal_samples_per_class: int,
        wandb_log_details: dict[str, Any],
    ) -> None:
        self.random_seed = random_seed
        self.device = device
        self.data_fetcher = data_fetcher
        self.prompt_pool_size = prompt_pool_size
        self.prompt_length = prompt_length
        self.top_n_prompts = top_n_prompts
        self.use_frequency_penalty = use_frequency_penalty
        self.batch_size = batch_size
        self.epochs_per_task = epochs_per_task
        self.learning_rate = learning_rate
        self.lambda_surrogate = lambda_surrogate
        self.grad_clip_max_norm = grad_clip_max_norm
        self.rehearsal_samples_per_class = rehearsal_samples_per_class
        self.shared_test_set = self.data_fetcher == FetcherType.CORE50
        self.wandb_log_details = wandb_log_details

        self.config = {
            "random_seed": self.random_seed,
            "device": self.device,
            "data_fetcher": self.data_fetcher,
            "prompt_pool_size": self.prompt_pool_size,
            "prompt_length": self.prompt_length,
            "top_n_prompts": self.top_n_prompts,
            "use_frequency_penalty": self.use_frequency_penalty,
            "batch_size": self.batch_size,
            "epochs_per_task": self.epochs_per_task,
            "learning_rate": self.learning_rate,
            "lambda_surrogate": self.lambda_surrogate,
            "grad_clip_max_norm": self.grad_clip_max_norm,
            "rehearsal_samples_per_class": self.rehearsal_samples_per_class,
            "shared_test_set": self.shared_test_set,
        }

        self.pipeline = self._initialize()

    def execute(self) -> None:
        """
        Executes the training and logs the result to W&B, as well as printing it to the terminal.
        """
        try:
            self.pipeline.train_on_dataset_for_accuracy_and_forgetting()
        finally:
            wandb.finish()

    def _initialize(self) -> ContinualLearningPipeline:
        """
        Initializes all the necessary components before assembling the pipeline based on the passed configurations.
        W&B is initialized before the initialization.

        Returns:
            ContinualLearningPipeline: Pipeline.
        """

        self.wandb_log_details.update({"config": self.config})
        wandb.init(**self.wandb_log_details)

        wandb.define_metric("current_task_index")
        wandb.define_metric("accuracy/*", step_metric="current_task_index")
        wandb.define_metric("forgetting/*", step_metric="current_task_index")

        set_global_seed(random_seed=self.random_seed)

        data_package = self._initialize_data()

        num_classes = data_package.get_number_of_classes()
        L2P_model = self._intialize_model(num_classes=num_classes)

        self._precompute_queries(model=L2P_model, data_package=data_package)

        optimizer_factory = self._initialize_optimizer_factory()

        num_tasks = data_package.get_number_of_tasks()
        rehearsal_buffer = self._initialize_rehearsal_buffer(num_tasks=num_tasks)

        single_task_trainer = self._intialize_single_task_trainer(
            model=L2P_model,
            optimizer_factory=optimizer_factory,
            rehearsal_buffer=rehearsal_buffer,
        )

        evaluator = self._initialize_evaluator(model=L2P_model)

        metrics_tracker = self._intialize_metrics_tracker(num_tasks=num_tasks)

        pipeline = self._initialize_pipeline(
            model=L2P_model,
            data_package=data_package,
            single_task_trainer=single_task_trainer,
            evaluator=evaluator,
            metrics_tracker=metrics_tracker,
            rehearsal_buffer=rehearsal_buffer,
        )

        return pipeline

    def _precompute_queries(self, model: L2PModel, data_package: DataPackage) -> None:
        query_precomputer = QueryPrecomputer(
            feature_extractor=model.feature_extractor,
            batch_size=self.batch_size,
            device=self.device,
        )
        query_precomputer.precompute_for_tasks(
            tasks=data_package.get_tasks(),
            shared_test_set=self.shared_test_set,
        )

    def _initialize_data(self) -> DataPackage:
        data_fetcher_object = self.FETCHER_MAP[self.data_fetcher](
            random_seed=self.random_seed
        )

        data_package = data_fetcher_object.get_data()

        return data_package

    def _intialize_model(self, num_classes: int) -> L2PModel:
        L2P_model = L2PModel(
            num_classes=num_classes,
            pool_size=self.prompt_pool_size,
            prompt_length=self.prompt_length,
            top_n=self.top_n_prompts,
            use_frequency_penalty=self.use_frequency_penalty,
        ).to(self.device)

        return L2P_model

    def _initialize_optimizer_factory(self) -> OptimizerFactory:
        optimizer_factory = OptimizerFactory(
            learning_rate=self.learning_rate,
        )

        return optimizer_factory

    def _initialize_rehearsal_buffer(self, num_tasks: int) -> RehearsalBuffer | None:
        if self.rehearsal_samples_per_class > 0:
            samples_per_class = self.rehearsal_samples_per_class
            if self.shared_test_set:
                samples_per_class = max(1, samples_per_class // num_tasks)
            rehearsal_buffer = RehearsalBuffer(
                samples_per_class=samples_per_class,
                random_seed=self.random_seed,
                allow_class_overlap=self.shared_test_set,
            )
        else:
            rehearsal_buffer = None

        return rehearsal_buffer

    def _intialize_single_task_trainer(
        self,
        model: L2PModel,
        optimizer_factory: OptimizerFactory,
        rehearsal_buffer: RehearsalBuffer | None,
    ) -> SingleTaskTrainer:
        single_task_trainer = SingleTaskTrainer(
            model=model,
            optimizer_factory=optimizer_factory,
            epochs_per_task=self.epochs_per_task,
            batch_size=self.batch_size,
            device=self.device,
            lambda_surrogate=self.lambda_surrogate,
            grad_clip_max_norm=self.grad_clip_max_norm,
            rehearsal_buffer=rehearsal_buffer,
        )

        return single_task_trainer

    def _initialize_evaluator(self, model: L2PModel) -> ContinualLearningEvaluator:
        evaluator = ContinualLearningEvaluator(
            model=model,
            batch_size=self.batch_size,
            device=self.device,
            shared_test_set=self.shared_test_set,
        )

        return evaluator

    def _intialize_metrics_tracker(self, num_tasks: int) -> MetricsTracker:
        metrics_tracker = MetricsTracker(num_tasks=num_tasks)

        return metrics_tracker

    def _initialize_pipeline(
        self,
        model: L2PModel,
        data_package: DataPackage,
        single_task_trainer: SingleTaskTrainer,
        evaluator: ContinualLearningEvaluator,
        metrics_tracker: MetricsTracker,
        rehearsal_buffer: RehearsalBuffer | None,
    ) -> ContinualLearningPipeline:
        pipeline = ContinualLearningPipeline(
            model=model,
            data_package=data_package,
            single_task_trainer=single_task_trainer,
            evaluator=evaluator,
            metrics_tracker=metrics_tracker,
            rehearsal_buffer=rehearsal_buffer,
        )

        return pipeline
