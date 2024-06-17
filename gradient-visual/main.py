import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_regression, make_classification
from sklearn.preprocessing import StandardScaler

from forward_propogation import propagate_classification


class GradientVisual:
    """
    A function that visualises the iterative process of gradient descent for LogisticRegression.
    """

    def __init__(self, X, y):
        """
        Initialisation class of the `GradientVisual` module.
        """

        self.X = X
        self.y = y
        self.cost_batch_gd = []  # Storing the loss of batch gradient descent.
        self.cost_mini_batch_gd = []  # Storing the loss of mini batch gradient descent.
        self.cost_stoch_gd = []  # # Storing the loss of stochastic gradient descent.

    def batch_gradient_descent(self, num_iterations=1000, learning_rate=0.01):
        """
        Batch gradient descent implimentation.
        """
        X, y = self.X, self.y
        weights_batch = np.zeros(
            X.shape[1]
        )  # Weights associated to batch gradient descent
        bias = 0

        for _ in num_iterations:
            gradients, cost = propagate_classification(weights_batch, bias, X, y)
            dW = gradients["dW"]
            db = gradients["db"]

            weights_batch -= learning_rate * dW
            bias -= learning_rate * db

            self.cost_batch_gd.append(cost)

    def mini_batch_gradient_descent(
        self, num_iterations=1000, learning_rate=0.01, num_batches=10
    ):
        """
        Mini batch gradient descent implementation.
        """
        X, y = self.X, self.y
        weights_mini_batch = np.zeros(X.shape[1])
