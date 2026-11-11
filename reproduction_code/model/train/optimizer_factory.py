from enum import Enum

from torch.nn import Parameter
from torch.optim import SGD, Adam, Optimizer

from reproduction_code.model.forward.L2P_model import L2PModel


class OptimizerType(str, Enum):
    ADAM = "ADAM"
    SGD = "SGD"


class OptimizerFactory:
    """
    Creates an Adam or SGD optimizer for the trainable parameters of the model.
    """

    def __init__(
        self,
        learning_rate: float,
        optimizer_type: OptimizerType,
        momentum: float | None,
    ) -> None:
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.momentum = momentum

    def create_optimizer(self, model: L2PModel) -> Optimizer:
        trainable_parameters = self._get_trainable_parameters(model)

        if self.optimizer_type == OptimizerType.ADAM:
            return Adam(trainable_parameters, lr=self.learning_rate)
        else:
            assert self.momentum is not None
            return SGD(
                trainable_parameters, lr=self.learning_rate, momentum=self.momentum
            )

    def _get_trainable_parameters(self, model: L2PModel) -> list[Parameter]:
        return [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
