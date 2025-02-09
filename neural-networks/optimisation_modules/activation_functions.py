"""
A script to store all the activation functions and their respective derivatives.

Author: Harsh Parikh

Date: 03-02-2025
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


# This is the parent class for the activation functions


@dataclass
class ActivationFunction:
    """
    The parent class for the implementation of the activation functions.
    """

    x: NDArray

    def forward(self) -> NDArray:
        """Computes the activation function."""
        raise NotImplementedError

    def derivative(self) -> NDArray:
        """Computes the derivative of the activation function."""
        raise NotImplementedError


@dataclass
class Tanh(ActivationFunction):
    """
    The activation function for Tanh is given by:

    F(x) = (e^x + e^-x) / (e^x - e^-x)

    F(x) -> (-1, 1)
    """

    def forward(self) -> NDArray:
        """
        Feeding into the activation function.
        """
        return np.tanh(self.x)

    def derivative(self) -> NDArray:
        """
        Outputs the first derivative of the activation function.
        """
        return 1 - np.tanh(self.x) ** 2


@dataclass
class Sigmoid(ActivationFunction):
    """
    The activation function for Sigmoid is given by:

    F(x) = 1 / (1 + e^(-x))

    F(x) -> (0, 1)
    """

    def forward(self) -> NDArray:
        """
        Feeding into the activation function.
        """
        return 1 / (1 + np.exp(-self.x))

    def derivative(self) -> NDArray:
        """
        Outputs the first derivative of the activation function.
        """
        sigmoid = self.forward()
        return sigmoid * (1 - sigmoid)


@dataclass
class ReLU(ActivationFunction):
    """
    The activation function for ReLU is given by:

    F(x) = max(0, x)

    F(x) -> (0, inf)
    """

    def forward(self) -> NDArray:
        """
        Feeding into the activation function.
        """
        return np.maximum(0.0, self.x)

    def derivative(self) -> NDArray:
        """
        Calculates the first derivative of the activation function.
        """

        return np.heaviside(self.x, 0.0)


@dataclass
class Softplus(ActivationFunction):
    """
    The activation function for Softplus is given by:

    F(x) = log(1 + e^x)

    F(x) -> (0, inf)
    """

    def forward(self) -> NDArray:
        """
        Feeding into the activation function.
        """
        return np.log(1 + np.exp(self.x))

    def derivative(self) -> NDArray:
        return np.exp(self.x) / (1 + np.exp(self.x))


# This function is to predict the outputs of the neural network
def softmax(array):
    """
    Compute the softmax of an array.

    Parameters:
    vector (bool): If True, applies softmax to a 1D array. If False, applies softmax to each row of a 2D array.
    array (np.ndarray): Input array.

    Returns:
    np.ndarray: Softmax-transformed array.
    """
    exp_array = np.exp(
        array - np.max(array)
    )  # Subtracting the max for numerical stability

    return exp_array / np.sum(exp_array)
