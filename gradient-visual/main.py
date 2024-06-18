"""
A script that visualises the path taken by different gradient descent methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from forward_propogation import propagate_classification


class GradientVisual:
    """
    A class that visualizes the iterative process of gradient descent for Logistic Regression.

    Attributes:
        X (np.ndarray): Input features.
        y (np.ndarray): Target labels.
        cost_batch_gd (list): List to store the loss of batch gradient descent.
        cost_mini_batch_gd (list): List to store the loss of mini-batch gradient descent.
        cost_stoch_gd (list): List to store the loss of stochastic gradient descent.
    """

    def __init__(self, features, target):
        """
        Initialize the GradientVisual class.

        Args:
            features (np.ndarray): Input features.
            y (np.ndarray): Target labels.
        """
        self.features = features
        self.target = target
        self.cost_batch_gd = []  # Storing the loss of batch gradient descent.
        self.cost_mini_batch_gd = []  # Storing the loss of mini batch gradient descent.
        self.cost_stoch_gd = []  # Storing the loss of stochastic gradient descent.
        self.weights = np.zeros(features.shape[1])
        self.bias = 0

        # Storing paths to visualise the gradient descent algorithm
        self.batch = []
        self.mini_batch = []
        self.stochastic = []

    def batch_gradient_descent(self, num_iterations=1000, learning_rate=0.01):
        """
        Batch gradient descent implementation.

        Args:
            num_iterations (int, optional): Number of iterations. Default is 1000.
            learning_rate (float, optional): Learning rate. Default is 0.01.
        """
        weights_batch = self.weights.copy()
        bias_batch = self.bias

        for _ in range(num_iterations):
            gradients, cost = propagate_classification(
                weights_batch, bias_batch, self.features, self.target
            )
            weights_grad = gradients["dW"]
            bias_grad = gradients["dB"]

            weights_batch -= learning_rate * weights_grad
            bias_batch -= learning_rate * bias_grad

            self.cost_batch_gd.append(cost)
            self.batch.append(weights_batch.copy())

    def mini_batch_gradient_descent(
        self, num_iterations=1000, learning_rate=0.01, batch_size=32
    ):
        """
        Vectorized mini-batch gradient descent implementation.

        Args:
            num_iterations (int, optional): Number of iterations. Default is 1000.
            learning_rate (float, optional): Learning rate. Default is 0.01.
            batch_size (int, optional): Size of mini-batches. Default is 32.
        """
        weights_mini_batch = self.weights.copy()
        bias_mini_batch = self.bias
        m = self.features.shape[0]

        for _ in range(num_iterations):
            indices = np.random.permutation(m)
            features_shuffled = self.features[indices]
            y_shuffled = self.target[indices]
            _, cost = propagate_classification(
                weights_mini_batch, bias_mini_batch, self.features, self.target
            )
            self.cost_mini_batch_gd.append(cost)
            self.mini_batch.append(weights_mini_batch.copy())
            for batch_start in range(0, m, batch_size):
                batch_end = min(
                    batch_start + batch_size, m
                )  # Ensuring there is no index overflow.
                features_batch = features_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]

                gradients, cost = propagate_classification(
                    weights_mini_batch, bias_mini_batch, features_batch, y_batch
                )
                weights_grad = gradients["dW"]
                bias_grad = gradients["dB"]

                # Update weights and biases
                weights_mini_batch -= learning_rate * weights_grad
                bias_mini_batch -= learning_rate * bias_grad

    def stochastic_gradient_descent(self, num_iterations=1000, learning_rate=0.01):
        """
        Vectorized stochastic gradient descent implementation.

        Args:
            num_iterations (int, optional): Number of iterations. Default is 1000.
            learning_rate (float, optional): Learning rate. Default is 0.01.
        """
        weights_stoch = self.weights.copy()
        bias_stoch = self.bias
        m = self.features.shape[0]

        for _ in range(num_iterations):
            indices = np.random.permutation(m)
            features_shuffled = self.features[indices]
            y_shuffled = self.target[indices]
            _, cost = propagate_classification(
                weights_stoch, bias_stoch, self.features, self.target
            )

            self.cost_stoch_gd.append(cost)
            self.stochastic.append(weights_stoch.copy())
            for i in range(0, m):
                feature_i = features_shuffled[i]
                target_i = y_shuffled[i]

                gradients, cost = propagate_classification(
                    weights_stoch, bias_stoch, feature_i, target_i
                )
                weights_grad = gradients["dW"]
                bias_grad = gradients["dB"]

                weights_stoch -= learning_rate * weights_grad
                bias_stoch -= learning_rate * bias_grad

    def activate(self, learning_rate=0.01, num_iterations=1000):
        """
        Run all the gradient descent methods.
        """
        self.batch_gradient_descent(
            learning_rate=learning_rate, num_iterations=num_iterations
        )
        self.mini_batch_gradient_descent(
            learning_rate=learning_rate, num_iterations=num_iterations
        )
        self.stochastic_gradient_descent(
            learning_rate=learning_rate, num_iterations=num_iterations
        )

    def plot_costs(self):
        """
        Plots the costs of each gradient descent method with respect to every iteration.
        """
        self.activate()
        plt.plot(self.cost_batch_gd, label="Batch Gradient Descent")
        plt.plot(self.cost_mini_batch_gd, label="Mini-Batch Gradient Descent")
        plt.plot(self.cost_stoch_gd, label="Stochastic Gradient Descent")
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.legend()
        plt.show()

    def plot_contours_and_paths(self):
        """
        Plot the contour of the cost function and the paths taken by the gradient descent methods.
        """
        # Create a grid of points
        self.activate(learning_rate=0.05, num_iterations=100)  # Activating the function
        theta0_vals = np.linspace(-7, 7, 100)
        theta1_vals = np.linspace(-7, 7, 100)
        theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

        # Compute cost for each point in the grid
        z = np.array(
            [
                propagate_classification(
                    np.array([t0, t1]), self.bias, self.features, self.target
                )[1]
                for (t0, t1) in zip(theta0_vals.ravel(), theta1_vals.ravel())
            ]
        )
        Z = z.reshape(theta0_vals.shape)

        _, ax = plt.subplots(figsize=(10, 8))

        # Plot contours
        CS = ax.contourf(
            theta0_vals, theta1_vals, Z, levels=50, cmap="viridis", alpha=0.7
        )
        plt.colorbar(CS)

        # Get paths of gradient descent methods
        paths = {
            "batch": self.batch,
            "mini_batch": self.mini_batch,
            "stochastic": self.stochastic,
        }
        # Plot paths
        ax.plot(
            [p[0] for p in paths["batch"]],
            [p[1] for p in paths["batch"]],
            "r-",
            label="Batch GD",
            linewidth=2,
        )
        ax.plot(
            [p[0] for p in paths["stochastic"]],
            [p[1] for p in paths["stochastic"]],
            "g-",
            label="Stochastic GD",
            linewidth=2,
        )
        ax.plot(
            [p[0] for p in paths["mini_batch"]],
            [p[1] for p in paths["mini_batch"]],
            "b-",
            label="Mini-Batch GD",
            linewidth=2,
        )

        # Plot start points
        ax.plot(
            paths["batch"][0][0],
            paths["batch"][0][1],
            "ro",
            label="Start",
            markersize=10,
        )

        # Plot end points
        ax.plot(
            paths["batch"][-1][0],
            paths["batch"][-1][1],
            "r*",
            label="Batch GD End",
            markersize=10,
        )
        ax.plot(
            paths["stochastic"][-1][0],
            paths["stochastic"][-1][1],
            "g*",
            label="Stochastic GD End",
            markersize=10,
        )
        ax.plot(
            paths["mini_batch"][-1][0],
            paths["mini_batch"][-1][1],
            "b*",
            label="Mini-Batch GD End",
            markersize=10,
        )

        ax.set_xlabel(r"$\theta_0$")
        ax.set_ylabel(r"$\theta_1$")
        ax.set_title("Gradient Descent Paths")
        ax.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    feature_matrix, target_matrix = make_classification(
        n_samples=100,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42,
    )

    # Standardising the features
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)
    gv = GradientVisual(feature_matrix, target_matrix)
    gv.plot_costs()
