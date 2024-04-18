import numpy as np
from raw_nn import Layer


class Softmax(Layer):
    """
    Softmax activation function.
    """
    def forward(self, inputs):
        self.inputs = inputs
        inputs = inputs.astype('float64')
        tmp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.outputs = tmp / np.sum(tmp, axis=1, keepdims=True)

        return self.outputs

    def backward(self, output_gradient):
        d_inputs = np.empty_like(output_gradient)
        for index, (single_output, single_output_gradient) in enumerate(zip(self.outputs, output_gradient)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            d_inputs[index] = np.dot(jacobian_matrix, single_output_gradient)

        return d_inputs
