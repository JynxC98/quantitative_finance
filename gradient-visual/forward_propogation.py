"""
Script to calculate the forward propogation of the linear models.
"""

import numpy as np


def sigmoid(x):
    """
    Calculates the sigmoid of a particular vector.

    Sigmoid is calculated by:
    f(x) = 1 / (1 + e^(-x))

    Returns
    -------
        sigmoid(x)
    """
    return 1 / (1 + np.exp(-x))


def propagate_regression(coef, intercept, feature_vector, target_vector):
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
    num_samples = feature_vector.shape[0]
    y_hat = np.dot(feature_vector, coef.T) + intercept
    cost = np.sum((y_hat - target_vector) ** 2) / num_samples
    dM = (
        -np.dot((target_vector - y_hat), feature_vector) / num_samples
    )  # Gradient of coefficients.
    dC = -np.sum((target_vector - y_hat)) / num_samples  # Gradient of intercept.

    gradients = {"dM": dM, "dC": dC}

    return cost, gradients


def propagate_classification(weights, bias, feature_vector, target_vector):
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
    num_samples = feature_vector.shape[0]
    z = np.dot(feature_vector, weights) + bias
    probability_vector = sigmoid(z)
    cost = (1 / num_samples) * np.sum(
        (-np.log(probability_vector) * target_vector)
        + (-np.log(1 - probability_vector) * (1 - target_vector))
    )

    dW = np.dot(feature_vector.T, (probability_vector - target_vector)) / num_samples
    dB = np.sum(probability_vector - target_vector) / num_samples

    gradients = {"dW": dW, "dB": dB}
    return gradients, cost
