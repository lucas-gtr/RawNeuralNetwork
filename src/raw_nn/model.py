import pickle
import copy

import numpy as np
from typing import List

from raw_nn import Dropout
from raw_nn import Layer
from raw_nn import TrainableLayer


class Model:
    """
    Neural network model.

    Attributes:
        _training (bool): Indicates whether the model is in training mode or not.
        layers (List[Layer]): List of layers in the model.
        parameters (List[TrainableLayer]): List of trainable layers in the model.
    """

    def __init__(self):
        """
        Initializes the model.
        """
        self._training: bool = True
        self.layers: List[Layer] = []
        self.parameters: List[TrainableLayer] = []

    def add(self, layer: Layer):
        """
        Adds a layer to the model.

        Args:
            layer (Layer): Layer to be added to the model.
        """
        if isinstance(layer, TrainableLayer):
            self.parameters.append(layer)
        self.layers.append(layer)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs forward pass through the model.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Output data.
        """
        for layer in self.layers:
            X = layer(X)
        return X

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Performs forward pass when model is called.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Output data.
        """
        return self.forward(X)

    @property
    def training(self) -> bool:
        """
        Returns the current training mode.

        Returns:
            bool: Current training mode.
        """
        return self._training

    @training.setter
    def training(self, training_value: bool):
        """
        Sets the training mode for the model and trainable_layers.

        Args:
            training_value (bool): Boolean value indicating whether to set training mode.
        """
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.is_active = training_value
        self._training = training_value

    def train(self, mode: bool = True):
        """
        Sets the training mode.

        Args:
            mode (bool): Boolean value indicating whether to set training mode to True.
        """
        self.training = mode

    def eval(self, mode: bool = True):
        """
        Sets the evaluation mode.

        Args:
            mode (bool): Boolean value indicating whether to set training mode to False.
        """
        self.training = not mode

    def save_parameters(self, path):
        """
        Saves the parameters of trainable layers to a file using pickle.

        Args:
            path (str): File path where parameters will be saved.
        """
        state_dict = []
        for layer in self.parameters:
            state_dict.append(layer.get_parameters())

        with open(path, 'wb') as f:
            pickle.dump(state_dict, f)

    def load_parameters(self, path):
        """
        Loads parameters of trainable layers from a file.

        Args:
            path (str): File path from where parameters will be loaded.
        """
        with open(path, 'rb') as f:
            state_dict = pickle.load(f)

        for layer, dict_parameters in zip(self.parameters, state_dict):
            layer.set_parameters(dict_parameters)

    def save_model(self, path):
        """
        Saves the model (excluding intermediate state) to a file using pickle.

        Args:
            path (str): File path where the model will be saved.
        """
        model = copy.deepcopy(self)

        for layer in model.layers:
            for attr in ['inputs', 'output', 'd_inputs',
                             'd_weights', 'd_biases']:
                layer.__dict__.pop(attr, None)

        with open(path, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load_model(path):
        """
        Loads a model from a file.

        Args:
            path (str): File path from where the model will be loaded.

        Returns:
            Model: Loaded model.
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
