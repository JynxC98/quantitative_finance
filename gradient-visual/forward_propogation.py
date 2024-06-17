"""
Script to calculate the forward propogation of the linear models.
"""

import numpy as np


def sigmoid(z):
    """
    Calculates the sigmoid of a particular vector.

    Sigmoid is calculated by:
    f(x) = 1 / (1 + e^(-x))

    Returns
    -------
        sigmoid(x)
    """
    return 1 / (1 + np.exp(-z))


def propagate_regression(coef, intercept, X, y):
    """
    Calculates the gradients and cost for the `LinearRegression` model.
    Args:
        - coef (numpy array): Model coefficients of shape (n,).
        - intercept (float): Model intercept.
        - X (numpy array): Training features of shape (m, n).
        - y (numpy array): Training labels of shape (m,).

    Returns:
    - gradients (dictionary): Gradients for coefficients and intercept.
    - cost (numpy array): Cost associated with the model.
    """
    num_samples = X.shape[0]
    y_hat = np.dot(X, coef.T) + intercept
    cost = np.sum((y_hat - y) ** 2) / num_samples
    dM = -np.dot((y - y_hat), X) / num_samples  # Gradient of coefficients.
    dC = -np.sum((y - y_hat)) / num_samples  # Gradient of intercept.

    gradients = {"dM": dM, "dC": dC}

    return cost, gradients


def propagate_classification(weights, bias, X_train, y_train):
    """
    Calculates the gradients and cost for the `LogisticRegression` model.
    Args:
        - weights (numpy array): Model weights of shape (n,).
        - bias (float): Model bias.
        - X_train (numpy array): Training features of shape (m, n).
        - y_train (numpy array): Training labels of shape (m,).

    Returns:
    - gradients (dictionary): Gradients for weights and bias.
    - cost (numpy array): Cost associated with the model.
    """
    num_samples = X_train.shape[0]
    z = np.dot(X_train, weights) + bias
    probability_vector = sigmoid(z)
    cost = (1 / num_samples) * np.sum(
        (-np.log(probability_vector) * y_train)
        + (-np.log(1 - probability_vector) * (1 - y_train))
    )

    dW = np.dot(X_train.T, (probability_vector - y_train)) / num_samples
    dB = np.sum(probability_vector - y_train) / num_samples

    gradients = {"dW": dW, "dB": dB}
    return gradients, cost
