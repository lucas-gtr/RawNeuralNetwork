import numpy as np
from raw_nn import Loss


class MeanSquaredError(Loss):
    """
    Mean Squared Error (MSE) loss function.
    """
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        sample_losses = np.mean((y_true - y_pred) ** 2, axis=-1)

        batch_size, output_size = y_pred.shape

        self.d_inputs = - 2 * (y_true - y_pred) / output_size
        self.d_inputs = self.d_inputs / batch_size

        data_loss = np.mean(sample_losses)

        return data_loss
