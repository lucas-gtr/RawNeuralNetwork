from raw_nn import Layer


class TrainableLayer(Layer):
    def __init__(self):
        super().__init__()
        self.include_bias = True

        self.weights = None
        self.d_weights = None
        self.weights_regularizer_l1 = None
        self.weights_regularizer_l2 = None

        self.bias = None
        self.d_bias = None
        self.bias_regularizer_l1 = None
        self.bias_regularizer_l2 = None

    def get_parameters(self) -> dict:
        """
        Return the parameters of the trainable layer.

        Returns:
            state_dict: Parameters names and values
        """
        if self.include_bias:
            return {'weights': self.weights, 'bias': self.bias}
        else:
            return {'weights': self.weights}

    def set_parameters(self, state_dict):
        """
        Set the parameters of the trainable layer.
        """
        self.weights = state_dict['weights']
        if self.include_bias:
            self.bias = state_dict['bias']
