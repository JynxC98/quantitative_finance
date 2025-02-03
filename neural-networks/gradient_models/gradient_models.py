"""
This script stores the gradient implementations of the following types:

1. Gradient Descent
2. Batch Gradient Descent
3. Stochastic Gradient Descent

Author: Harsh Parikh
Date: 03-02-2025
"""

from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from forward_propagation import propagate_classification, propagate_regression


def batch_gradient_descent(
    weights: NDArray,
    bias: float,
    features: NDArray,
    target: NDArray,
    num_iterations: int = 1000,
    learning_rate: float = 0.01,
    is_classification: bool = True,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Batch gradient descent implementation for both classification and regression.

    Args:
        num_iterations (int, optional): Number of iterations. Default is 1000.
        learning_rate (float, optional): Learning rate. Default is 0.01.
        is_classification (bool, optional): Whether the model is for classification or regression. Default is True.
    """
    weights_batch = weights.copy()
    bias_batch = bias
    cost_batch_gd = []  # To store cost history

    for _ in range(num_iterations):
        if is_classification:
            gradients, cost = propagate_classification(
                weights_batch, bias_batch, features, target
            )
        else:
            gradients, cost = propagate_regression(
                weights_batch, bias_batch, features, target
            )

        weights_grad = gradients["dW"]
        bias_grad = gradients["dB"]

        weights_batch -= learning_rate * weights_grad
        bias_batch -= learning_rate * bias_grad

        cost_batch_gd.append(cost)  # Append cost for each iteration

    return weights_batch, bias_batch, cost_batch_gd


def mini_batch_gradient_descent(
    weights: NDArray,
    bias: float,
    features: NDArray,
    target: NDArray,
    num_iterations: int = 1000,
    learning_rate: float = 0.01,
    batch_size: int = 32,
    is_classification: bool = True,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Vectorised mini-batch gradient descent implementation for both classification and regression.

    Args:
        num_iterations (int, optional): Number of iterations. Default is 1000.
        learning_rate (float, optional): Learning rate. Default is 0.01.
        batch_size (int, optional): Size of mini-batches. Default is 32.
        is_classification (bool, optional): Whether the model is for classification or regression. Default is True.
    """
    weights_mini_batch = weights.copy()
    bias_mini_batch = bias
    num_samples = features.shape[0]
    cost_mini_batch_gd = []  # To store cost history

    for _ in range(num_iterations):
        indices = np.random.permutation(num_samples)
        features_shuffled = features[indices]
        target_shuffled = target[indices]

        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            features_batch = features_shuffled[batch_start:batch_end]
            target_batch = target_shuffled[batch_start:batch_end]

            if is_classification:
                gradients, cost = propagate_classification(
                    weights_mini_batch, bias_mini_batch, features_batch, target_batch
                )
            else:
                gradients, cost = propagate_regression(
                    weights_mini_batch, bias_mini_batch, features_batch, target_batch
                )

            weights_grad = gradients["dW"]
            bias_grad = gradients["dB"]

            weights_mini_batch -= learning_rate * weights_grad
            bias_mini_batch -= learning_rate * bias_grad

        cost_mini_batch_gd.append(cost)  # Append cost for each iteration

    return weights_mini_batch, bias_mini_batch, cost_mini_batch_gd


def stochastic_gradient_descent(
    weights: NDArray,
    bias: float,
    features: NDArray,
    target: NDArray,
    num_iterations: int = 1000,
    learning_rate: float = 0.01,
    is_classification: bool = True,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Vectorised stochastic gradient descent implementation for both classification and regression.

    Args:
        num_iterations (int, optional): Number of iterations. Default is 1000.
        learning_rate (float, optional): Learning rate. Default is 0.01.
        is_classification (bool, optional): Whether the model is for classification or regression. Default is True.
    """
    weights_stoch = weights.copy()
    bias_stoch = bias
    num_samples = features.shape[0]
    cost_stoch_gd = []  # To store cost history

    for _ in range(num_iterations):
        indices = np.random.permutation(num_samples)
        features_shuffled = features[indices]
        target_shuffled = target[indices]

        for i in range(num_samples):
            feature_i = features_shuffled[i]
            target_i = target_shuffled[i]

            if is_classification:
                gradients, cost = propagate_classification(
                    weights_stoch,
                    bias_stoch,
                    feature_i,
                    target_i,
                )
            else:
                gradients, cost = propagate_regression(
                    weights_stoch,
                    bias_stoch,
                    feature_i,
                    target_i,
                )

            weights_grad = gradients["dW"]
            bias_grad = gradients["dB"]

            weights_stoch -= learning_rate * weights_grad
            bias_stoch -= learning_rate * bias_grad

        cost_stoch_gd.append(cost)  # Append cost for each iteration

    return weights_stoch, bias_stoch, cost_stoch_gd


if __name__ == "__main__":

    # Initialising random weights and bias for the example

    weights = np.random.randn(3)  # Example 3 features
    bias = np.random.randn(1)

    # Example data (features and targets)
    features = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    target = np.array([0, 1, 0])  # Binary target for classification

    # Batch Gradient Descent for Classification
    weights, bias, cost_batch = batch_gradient_descent(
        weights, bias, features, target, is_classification=True
    )
    print("Batch Gradient Descent Successfull")

    # Mini-Batch Gradient Descent for Regression (Assume regression targets are continuous)
    target_regression = np.array([0.1, 0.5, 0.9])  # Example regression targets
    weights, bias, cost_mini_batch = mini_batch_gradient_descent(
        weights, bias, features, target_regression, is_classification=False
    )
    print("Mini-Batch Gradient Descent for Regression successfull")

    # Stochastic Gradient Descent for Classification
    weights, bias, cost_stoch = stochastic_gradient_descent(
        weights, bias, features, target, is_classification=True
    )
    print("Stochastic Gradient Descent for Classification successfull")
