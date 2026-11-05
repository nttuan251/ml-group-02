from torch.nn import Parameter
from torch.optim import Adam, Optimizer

from reproduction_code.model.forward.L2P_model import L2PModel


class OptimizerFactory:
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    def create_optimizer(self, model: L2PModel) -> Optimizer:
        trainable_parameters = self._get_trainable_parameters(model)

        return Adam(trainable_parameters, lr=self.learning_rate)

    def _get_trainable_parameters(self, model: L2PModel) -> list[Parameter]:
        return [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
