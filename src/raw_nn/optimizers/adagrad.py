import numpy as np
from raw_nn import Optimizer


class Adagrad(Optimizer):
    """
    Adagrad optimizer implementation.
    """
    def __init__(self, parameters, *, learning_rate=0.001, decay=0., epsilon=1e-7):
        """
        Initializes the Adagrad optimizer.

        Args:
            parameters: List of parameters (e.g., weights and biases) of the model.
            learning_rate: The learning rate for the optimization algorithm.
            decay: The decay rate for the learning rate.
            epsilon: Small value to avoid division by zero.
        """
        super().__init__(parameters, learning_rate=learning_rate, decay=decay)
        self.epsilon = epsilon

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            if layer.include_biases:
                layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.d_weights ** 2
        layer.weights -= (self.current_learning_rate * layer.d_weights /
                          (np.sqrt(layer.weight_cache) + self.epsilon))

        if layer.include_biases:
            layer.bias_cache += layer.d_biases ** 2
            layer.biases -= (self.current_learning_rate * layer.d_biases /
                             (np.sqrt(layer.bias_cache) + self.epsilon))
