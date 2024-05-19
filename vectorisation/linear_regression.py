"""
Implementation of Linear Regression.
"""

import numpy as np


def propagate(coef, intercept, X, y):
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


class LinearRegression:
    """
    A simple linear regression model for fitting a straight line to data points.

    Attributes:
        coef_ (numpy.ndarray): Coefficients of the linear model.
        intercept_ (float): Intercept of the linear model.

    Methods:
        fit(X, y): Fit the linear regression model to the training data.
        predict(X): Predict output for new input data.
        score(X, y): Calculate the coefficient of determination (R^2) of the prediction.

    """

    def __init__(self):
        """
        Initialize the LinearRegression object.
        """
        self.coef_ = None
        self.intercept_ = None
        self.cost_ = []

    def fit(self, X, y, num_iterations=1000, learning_rate=0.01):
        """
        Fit the linear regression model to the training data.

        Args:
            X (numpy.ndarray): Input features of shape (n_samples, n_features).
            y (numpy.ndarray): Target values of shape (n_samples,).

        Returns:
            LinearRegression: Self.

        Raises:
            ValueError: If the dimensions of X and y do not match.
        """
        self.coef_ = np.zeros(X.shape[1])
        self.intercept_ = 0

        if X.shape[1] != len(y):
            raise ValueError(
                "The shape of the feature vector and target vector don't match."
            )
        for _ in range(num_iterations):
            cost, gradients = propagate(self.coef_, self.intercept_, X, y)
            dM = gradients["dM"]
            dC = gradients["dC"]
            self.coef_ -= learning_rate * dM
            self.intercept_ -= learning_rate * dC
            self.cost_.append(cost)

    def predict(self, X):
        """
        Predict output for new input data.

        Args:
            X (numpy.ndarray): Input features of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Predicted target values of shape (n_samples,).
        """
        y_hat = np.dot(X, self.coef_) + self.intercept_
        return y_hat

    def score(self, X, y):
        """
        Calculate the coefficient of determination (R^2) for the linear regression model.

        Parameters:
        -----------
        X : numpy.ndarray
            Input features of shape (n_samples, n_features).
        y : numpy.ndarray
            Target values of shape (n_samples,).

        Returns:
        --------
        float
            The R^2 score, a measure of the proportion of the variance in the dependent variable that is predictable from the independent variables.
        """
        y_hat = np.dot(X, self.coef_) + self.intercept_
        SSR = np.sum((y_hat - y) ** 2)  # Sum of squared residuals

        baseline_pred = np.mean(y) * np.ones(len(y))

        SST = np.sum((y - baseline_pred) ** 2)  # Sum of squared total
        r_squared = 1 - (SSR / SST)
        return r_squared
