import numpy as np
from raw_nn import Optimizer
from typing import List


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer implementation with momentum.
    """
    def __init__(self, parameters: List, *, learning_rate: float = 1.0, decay: float = 0.0, momentum: float = 0.0):
        """
        Initializes the SGD optimizer.

        Args:
            parameters: List of parameters (e.g., weights and bias) of the model.
            learning_rate: The learning rate for the optimization algorithm.
            decay: The decay rate for the learning rate.
            momentum: The momentum for the SGD update.
        """
        super().__init__(parameters, learning_rate=learning_rate, decay=decay)
        self.momentum = momentum

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                if layer.include_bias:
                    layer.bias_momentums = np.zeros_like(layer.bias)

            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.d_weights
            layer.weight_momentums = weight_updates
            layer.weights += weight_updates
            if layer.include_bias:
                bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.d_bias
                layer.bias_momentums = bias_updates
                layer.bias += bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.d_weights
            layer.weights += weight_updates
            if layer.include_bias:
                bias_updates = -self.current_learning_rate * layer.d_bias
                layer.bias += bias_updates

