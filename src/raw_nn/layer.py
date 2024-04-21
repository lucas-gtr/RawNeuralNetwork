import numpy as np


class Layer:
    """
    Base class for neural network trainable_layers.
    """
    def __init__(self):
        """
        Initializes the layer.
        """
        self.inputs = None
        self.outputs = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Performs forward pass through the layer.

        Args:
            inputs: Input data.

        Returns:
            Output data.
        """
        raise NotImplementedError("Subclasses of 'Layer' must implement 'forward' method")

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Performs backward pass through the layer.

        Args:
            output_gradient: Gradient of loss with respect to the layer's outputs.

        Returns:
            Gradient of loss with respect to the layer's inputs.
        """
        raise NotImplementedError("Subclasses of 'Layer' must implement 'backward' method")

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Performs forward pass when layer is called.

        Args:
            X: Input data.

        Returns:
            Output data.
        """
        return self.forward(X)
