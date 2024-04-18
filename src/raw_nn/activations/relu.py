import numpy as np
from raw_nn import Layer


class ReLU(Layer):
    """
    Rectified Linear Unit (ReLU) activation function.
    """
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)

        return self.outputs

    def backward(self, output_gradient):
        output_gradient[self.inputs <= 0] = 0

        return output_gradient
