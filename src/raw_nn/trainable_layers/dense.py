import numpy as np
from raw_nn import TrainableLayer


class Dense(TrainableLayer):
    """
    Fully connected layer.
    """
    def __init__(self, input_size: int, output_size: int, include_biases: bool = True,
                 weight_regularizer_l1: float = 0, weight_regularizer_l2: float = 0,
                 bias_regularizer_l1: float = 0, bias_regularizer_l2: float = 0):
        """
        Initializes the dense layer.

        Args:
            input_size: Number of input features.
            output_size: Number of output features.
            include_biases: Whether to include biases or not in the Dense layer.
            weight_regularizer_l1: L1 regularization penalty for weights.
            weight_regularizer_l2: L2 regularization penalty for weights.
            bias_regularizer_l1: L1 regularization penalty for biases.
            bias_regularizer_l2: L2 regularization penalty for biases.
        """
        super().__init__()
        self.include_biases = include_biases

        self.weights = np.random.randn(input_size, output_size)
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2

        if self.include_biases:
            self.biases = np.zeros((1, output_size))
            self.bias_regularizer_l1 = bias_regularizer_l1
            self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        if self.include_biases:
            self.outputs = np.dot(self.inputs, self.weights) + self.biases
        else:
            self.outputs = np.dot(self.inputs, self.weights)
        return self.outputs

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        self.d_weights = np.dot(self.inputs.T, output_gradient)
        if self.include_biases:
            self.d_biases = np.sum(output_gradient, axis=0, keepdims=True)

        if self.weight_regularizer_l1 > 0:
            d_l1 = np.ones_like(self.weights)
            d_l1[self.weights < 0] = -1
            self.d_weights += self.weight_regularizer_l1 * d_l1

        if self.weight_regularizer_l2 > 0:
            self.d_weights += 2 * self.weight_regularizer_l2 * self.weights

        if self.include_biases:
            if self.bias_regularizer_l1 > 0:
                d_l1 = np.ones_like(self.biases)
                d_l1[self.biases < 0] = -1
                self.d_biases += self.bias_regularizer_l1 * d_l1

            if self.bias_regularizer_l2 > 0:
                self.d_biases += 2 * self.bias_regularizer_l2 * self.biases

        d_inputs = np.dot(output_gradient, self.weights.T)
        return d_inputs
