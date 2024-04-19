import numpy as np
from raw_nn import Dense
from raw_nn import Model


class Loss:
    """
    Base class for defining loss functions.
    """
    def __init__(self, model: Model):
        """
        Initializes the loss function.

        Args:
            model: The neural network model.
        """
        self.d_inputs = None
        self.parameters = model.parameters
        self.layers = model.layers

    def regularization_loss(self) -> float:
        """
        Calculates the regularization loss.

        Returns:
            Regularization loss value.
        """
        regularization_loss = 0

        for layer in self.parameters:
            if isinstance(layer, Dense):
                if layer.weight_regularizer_l1 > 0:
                    regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

                if layer.weight_regularizer_l2 > 0:
                    regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

                if layer.include_biases:
                    if layer.bias_regularizer_l1 > 0:
                        regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

                    if layer.bias_regularizer_l2 > 0:
                        regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Computes the loss between predicted and true labels.

        Args:
            y_pred: Predicted labels.
            y_true: True labels.

        Returns:
            Loss value.
        """
        raise NotImplementedError("Subclasses must implement 'forward' method")

    def backward(self):
        """
        Computes the gradients of the loss.
        """
        d_inputs = self.d_inputs
        for layer in reversed(self.layers):
            d_inputs = layer.backward(d_inputs)

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray, *, include_regularization: bool = False):
        """
        Computes the loss.

        Args:
            y_pred: Predicted labels.
            y_true: True labels.
            include_regularization: Whether to include regularization loss.

        Returns:
            Data loss or data loss with regularization.
        """
        loss = self.forward(y_pred, y_true)
        data_loss = np.mean(loss)
        regularization_loss = self.regularization_loss()
        if include_regularization:
            return data_loss, regularization_loss
        else:
            return data_loss
