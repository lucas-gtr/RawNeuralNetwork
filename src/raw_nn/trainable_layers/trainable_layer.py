from raw_nn import Layer


class TrainableLayer(Layer):
    def __init__(self):
        super().__init__()
        self.include_biases = False

        self.weights = None
        self.d_weights = None
        self.weight_regularizer_l1 = None
        self.weight_regularizer_l2 = None

        self.biases = None
        self.d_biases = None
        self.bias_regularizer_l1 = None
        self.bias_regularizer_l2 = None

    def get_parameters(self) -> dict:
        """
        Return the parameters of the trainable layer.

        Returns:
            state_dict: Parameters names and values
        """
        if self.include_biases:
            return {'weights': self.weights, 'biases': self.biases}
        else:
            return {'weights': self.weights}

    def set_parameters(self, params):
        """
        Set the parameters of the trainable layer.
        """
        self.weights = params['weights']
        if self.include_biases:
            self.biases = params['biases']
