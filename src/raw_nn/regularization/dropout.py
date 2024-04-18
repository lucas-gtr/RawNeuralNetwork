import numpy as np
from raw_nn import Layer


class Dropout(Layer):
    """
    Dropout layer for regularization during training.
    """

    def __init__(self, rate: float):
        """
        Initializes the Dropout layer.

        Args:
            rate: The dropout rate, a float between 0 and 1, where 0 means no dropout and 1 means all features are
            dropped.
        """
        super().__init__()
        if not 0.0 <= rate < 1.0:
            raise ValueError("Dropout rate must be in the range [0, 1]")
        self.dropout_rate = 1.0 - rate
        self.mask = None
        self.is_active = True

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        if self.is_active:
            self.mask = np.random.binomial(1, self.dropout_rate, size=inputs.shape) / self.dropout_rate
            self.outputs = inputs * self.mask
        else:
            self.outputs = inputs
        return self.outputs

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        if self.is_active:
            return output_gradient * self.mask
        else:
            return output_gradient
