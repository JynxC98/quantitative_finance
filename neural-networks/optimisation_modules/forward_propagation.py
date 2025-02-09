"""
Script to calculate the forward propogation of the linear models.

Author: Harsh Parikh
Date: 03-02-2025
"""

from typing import Tuple, Dict
import numpy as np
from numpy.typing import NDArray
from .activation_functions import Tanh, Sigmoid, ReLU, Softplus


def propagate_regression(
    coef: NDArray, intercept: float, feature_vector: NDArray, target_vector: NDArray
) -> Tuple[float, Dict[str, float]]:
    """
    Calculates the gradients and cost for the `LinearRegression` model.
    Args:
        - coef (numpy array): Model coefficients of shape (n,).
        - intercept (float): Model intercept.
        - feature_vector (numpy array): Training features of shape (m, n).
        - y (numpy array): Training labels of shape (m,).

    Returns:
    - gradients (dictionary): Gradients for coefficients and intercept.
    - cost (numpy array): Cost associated with the model.
    """
    num_samples = feature_vector.shape[0]  # Fetching the number of data points

    y_hat = (
        np.dot(feature_vector, coef.T) + intercept
    )  # Calculating the value based on the current coef and intercpt

    cost = (
        np.sum((y_hat - target_vector) ** 2) / num_samples
    )  # Evaluating the cost based on MSE

    coeff_grad = (
        -np.dot((target_vector - y_hat), feature_vector) / num_samples
    )  # Gradient of coefficients.

    intercept_grad = (
        -np.sum((target_vector - y_hat)) / num_samples
    )  # Gradient of intercept.

    gradients = {"dW": coeff_grad, "dB": intercept_grad}

    return gradients, cost


def propagate_classification(
    weights: NDArray,
    bias: float,
    feature_vector: NDArray,
    target_vector: NDArray,
    activation_function: str = "sigmoid",
):
    """
    Calculates the gradients and cost for the `LogisticRegression` model.
    Args:
        - weights (numpy array): Model weights of shape (n,).
        - bias (float): Model bias.
        - feature_vector (numpy array): Training features of shape (m, n).
        - target_vector (numpy array): Training labels of shape (m,).

    Returns:
    - gradients (dictionary): Gradients for weights and bias.
    - cost (numpy array): Cost associated with the model.
    """
    # Checking wheather the activation function is in the pre-determined list

    if activation_function not in ("sigmoid", "tanh", "relu", "softplus"):
        raise ValueError("Please select one from sigmoid, tanh, relu")

    if activation_function == "sigmoid":
        activation_function = Sigmoid

    elif activation_function == "tanh":
        activation_function = Tanh

    elif activation_function == "relu":
        activation_function = ReLU
    else:
        activation_function = Softplus

    num_samples = feature_vector.shape[0]  # Fetching the number of data points

    vals = (
        np.dot(feature_vector, weights) + bias
    )  # Calculating the value based on current weights and bias

    probability_vector = activation_function(
        vals
    ).forward()  # Feeding the values in the activation function

    # Calculating log cross-entropy loss
    cost = (1 / num_samples) * np.sum(
        (-np.log(probability_vector) * target_vector)
        + (-np.log(1 - probability_vector) * (1 - target_vector))
    )

    # Gradient for weights
    weights_grad = (
        np.dot(feature_vector.T, (probability_vector - target_vector)) / num_samples
    )

    # Gradient for bias
    bias_grad = np.sum(probability_vector - target_vector) / num_samples

    gradients = {"dW": weights_grad, "dB": bias_grad}
    return gradients, cost
