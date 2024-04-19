import numpy as np
from raw_nn import Loss


class MeanAbsoluteError(Loss):
    """
    Mean Absolute Error (MAE) loss function.
    """
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        sample_losses = np.mean(abs(y_true - y_pred), axis=-1)

        batch_size, output_size = y_pred.shape

        self.d_inputs = np.sign(y_pred - y_true) / output_size
        self.d_inputs = self.d_inputs / batch_size

        return sample_losses
