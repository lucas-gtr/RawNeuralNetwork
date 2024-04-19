import numpy as np
from raw_nn import Loss
from raw_nn import Softmax


class CategoricalCrossEntropy(Loss):
    """
    Categorical Cross Entropy loss function.
    """
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        batch_size, output_size = y_pred.shape
        y_pred_clip = np.clip(y_pred, 1e-7, 1 - 1e-7)

        y_true = y_true.flatten()

        categorical_loss = y_pred_clip[range(batch_size), y_true]

        if isinstance(self.layers[-1], Softmax):
            self.d_inputs = y_pred.copy()
            self.d_inputs[range(batch_size), y_true] -= 1
            self.d_inputs = self.d_inputs / batch_size
        else:
            y_true = np.eye(output_size)[y_true]
            self.d_inputs = -y_true / y_pred
            self.d_inputs = self.d_inputs / batch_size

        return -np.log(categorical_loss)

    def backward(self):
        if isinstance(self.layers[-1], Softmax):
            d_inputs = self.d_inputs
            for layer in reversed(self.layers[:-1]):
                d_inputs = layer.backward(d_inputs)
        else:
            super().backward()
