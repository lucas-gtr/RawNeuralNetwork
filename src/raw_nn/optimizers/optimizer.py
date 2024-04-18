from typing import List


class Optimizer:
    """
    Base class for optimization algorithms.
    """
    def __init__(self, parameters: List, *, learning_rate: float = 0.001, decay: float = 0.):
        """
        Initializes the optimizer.

        Args:
            parameters: List of parameters (e.g., weights and biases) of the model.
            learning_rate: The learning rate for the optimization algorithm.
            decay: The decay rate for the learning rate.
        """
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def step(self):
        """
        Performs one optimization step.
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1. + self.decay * self.iterations)

        for layer in self.parameters:
            self.update_params(layer)

        self.iterations += 1

    def update_params(self, layer):
        """
        Updates the parameters of the layer based on the optimization algorithm.

        Args:
            layer: The layer whose parameters are to be updated.
        """
        raise NotImplementedError("Subclasses must implement 'update_params' method.")
