from raw_nn import Layer


class Linear(Layer):
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = inputs

        return self.outputs

    def backward(self, output_gradient):
        return output_gradient
