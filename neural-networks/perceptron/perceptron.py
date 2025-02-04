"""
An implementation of the perceptron model using NumPy extensively.

Author: Harsh Parikh

Date: 03-02-2025
"""

import warnings

import numpy as np
from numpy.typing import NDArray

from optimisation_modules.gradient_models import (
    batch_gradient_descent,
    mini_batch_gradient_descent,
    stochastic_gradient_descent,
)
from optimisation_modules.activation_functions import ReLU, Sigmoid

warnings.filterwarnings("ignore")


class CustomPerceptron:
    """
    The perceptron model is a supervised binary classification algorithm.
    The main class to simulate the perceptron model of the ANN using NumPy extensively.
    The theory has been referenced from the following sources:

    1. Deep Learning by Ian Goodfellow
    2. https://en.wikipedia.org/wiki/Perceptron
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        num_iterations: int = 1000,
        optimizer_type: str = "batch",
        activation: str = "sigmoid",
    ):
        """
        Initializes the perceptron model with given parameters.

        Args:
            - input_dim (int): Number of input features.
            - learning_rate (float, optional): Learning rate for weight updates. Default is 0.01.
            - num_iterations (int, optional): Number of training iterations. Default is 1000.
            - optimizer_type (str, optional): The type of optimization. Default is "batch"
        """
        # Checking the input for the optimizer function
        if optimizer_type not in ("batch", "mini-batch", "sgd"):
            raise ValueError("Please select one from batch, mini-batch and sgd")

        # Checking the input for the activation function
        if activation not in ("sigmoid", "relu"):
            raise ValueError("Please select one from sigmoid, relu")

        self.learning_rate_ = learning_rate
        self.num_iterations_ = num_iterations
        self.optimizer_type_ = optimizer_type
        self.activation_ = activation
        self.weights_ = None
        self.bias_ = 0.0

    def activation(self, x: NDArray) -> NDArray:
        """
        Activation function for the perceptron (Step function).

        Args:
            - x (NDArray): Input to the activation function.

        Returns:
            - NDArray: Binary output (0 or 1).
        """
        if self.activation_ == "sigmoid":
            return Sigmoid(x).forward()

        elif self.activation_ == "relu":
            return ReLU(x).derivative()  # Since we need values between 0 and 1

    def predict(self, X: NDArray, threshold: float = 0.5) -> NDArray:
        """
        Predicts the class labels for given input features.

        Args:
            - X (NDArray): Input feature matrix of shape (m, n).

        Returns:
            - NDArray: Predicted labels (0 or 1).
        """
        linear_output = np.dot(X, self.weights_) + self.bias_

        output = self.activation(linear_output)
        output = np.where(output < threshold, 0, 1)
        return output

    def fit(self, X: NDArray, y: NDArray):
        """
        Trains the perceptron using the perceptron learning rule.

        Args:
            - X (NDArray): Input feature matrix of shape (m, n).
            - y (NDArray): Target labels of shape (m,).
        """

        self.weights_ = np.zeros(X.shape[1])  # Initializing the weights

        # Evaluating the gradients based on the type of optimizer selected
        if self.optimizer_type_ == "batch":
            weights, bias, _ = batch_gradient_descent(
                weights=self.weights_,
                bias=self.bias_,
                features=X,
                target=y,
                num_iterations=self.num_iterations_,
                learning_rate=self.learning_rate_,
                is_classification=True,
            )
        elif self.optimizer_type_ == "mini-batch":
            weights, bias, _ = mini_batch_gradient_descent(
                weights=self.weights_,
                bias=self.bias_,
                features=X,
                target=y,
                num_iterations=self.num_iterations_,
                learning_rate=self.learning_rate_,
                is_classification=True,
            )

        else:
            weights, bias, _ = stochastic_gradient_descent(
                weights=self.weights_,
                bias=self.bias_,
                features=X,
                target=y,
                num_iterations=self.num_iterations_,
                learning_rate=self.learning_rate_,
                is_classification=True,
            )

        self.weights_ = weights
        self.bias_ = bias

    def evaluate(self, X: NDArray, y: NDArray) -> float:
        """
        Evaluates the model accuracy on test data.

        Args:
            - X (NDArray): Test feature matrix of shape (m, n).
            - y (NDArray): True labels of shape (m,).

        Returns:
            - float: Accuracy score of the perceptron model.
        """
        predictions = self.predict(X)
        accuracy = np.mean((predictions == y).astype(int))
        return accuracy
