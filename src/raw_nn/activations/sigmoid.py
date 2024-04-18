import numpy as np
from raw_nn import Layer


class Sigmoid(Layer):
    """
    Sigmoid activation function.
    """
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = 1 / (1 + np.exp(-inputs))

        return self.outputs

    def backward(self, output_gradient):
        d_inputs = output_gradient * (1 - self.outputs) * self.outputs

        return d_inputs
