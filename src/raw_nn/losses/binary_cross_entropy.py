from raw_nn import Loss
import numpy as np


class BinaryCrossEntropy(Loss):
    """
    Binary Cross Entropy loss function.
    """
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        batch_size, output_size = y_pred.shape

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        self.d_inputs = -(y_true / y_pred_clipped - (1 - y_true) / (1 - y_pred_clipped)) / output_size
        self.d_inputs = self.d_inputs / batch_size

        return sample_losses
