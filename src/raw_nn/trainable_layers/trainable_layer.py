from raw_nn import Layer


class TrainableLayer(Layer):
    def get_parameters(self) -> dict:
        """
        Return the parameters of the trainable layer.

        Returns:
            state_dict: Parameters names and values
        """
        raise NotImplementedError("Subclasses of 'TrainableLayer' must implement 'get_parameters' method")

    def set_parameters(self, *args, **kwargs):
        """
        Set the parameters of the trainable layer.
        """
        raise NotImplementedError("Subclasses of 'TrainableLayer' must implement 'set_parameters' method")
