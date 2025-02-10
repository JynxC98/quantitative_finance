"""
A script to simulate the artificial neural networks for classification.

Author: Harsh Parikh

Date: 04-02-2024
"""

import sys
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import f1_score, mean_squared_error, log_loss

from optimisation_modules.gradient_models import (
    batch_gradient_descent,
    mini_batch_gradient_descent,
    stochastic_gradient_descent,
)
from optimisation_modules.activation_functions import Sigmoid, ReLU, Tanh, Softplus


class NeuralNetworks:
    """
    The main class to initialise the multilayer perceptron model for classification.

    References:
    1. Deep Learning by Ian Goodfellow, Yoshua Bengio and Aaron Courville.
    2. https://math.uchicago.edu/~may/REU2018/REUPapers/Guilhoto.pdf
    3. https://www.youtube.com/watch?v=w8yWXqWQYmU
    4. http://neuralnetworksanddeeplearning.com/index.html
    5. https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=1

    Input parameters:
    -----------------
    - num_layers: The number of hidden layers.
    - layer_sizes: The number of neurons in each layer.
    - activation_function: The activation function for each layer.
    - optimiser: The choice of optimiser (batch, mini-batch, sgd)
    - num_epochs: The number of epochs for the training.
    - num_outputs: The number of outcomes for the problem.
    - verbose: Whether to print the process during learning
    """

    def __init__(
        self,
        num_layers: int,
        layer_sizes: int,
        activation_function: str = "sigmoid",
        optimiser: str = "batch",
        num_epochs: int = 1000,
        num_outputs: int = 1,
        learning_rate: float = 0.01,
        verbose: bool = True,
    ) -> None:
        """
        The initialisation class for the Neural Networks
        """
        self.num_layers_ = num_layers
        self.layer_size_ = layer_sizes
        self.activation_function_ = activation_function
        self.optimiser_ = optimiser
        self.num_epochs_ = num_epochs
        self.num_outputs_ = num_outputs
        self.learning_rate_ = learning_rate
        self.verbose_ = verbose

        # Checking the input parameters

        if activation_function not in ("sigmoid", "relu", "tanh", "softplus"):
            raise ValueError("Please select one from `sigmoid, relu, tanh, softplus`.")

        if optimiser not in ("batch", "mini-batch", "sgd"):
            raise ValueError("Please select one from `batch, mini-batch, sgd.")

        # These parameters will be calculated later.

        self.weights_ = None
        self.bias_ = None
        self.activations_ = None
        self.loss_history = None

        # Assigning the activation function
        activation_functions = {
            "sigmoid": Sigmoid,
            "relu": ReLU,
            "tanh": Tanh,
            "softplus": Softplus,
        }
        self.activation_function_ = activation_functions[activation_function]

        # Assigning the optimisation method
        if self.optimiser_ == "batch":
            self.optimiser_ = batch_gradient_descent
        elif self.optimiser_ == "mini-batch":
            self.optimiser_ = mini_batch_gradient_descent
        else:
            self.optimiser_ = stochastic_gradient_descent

    def fit(self, X: NDArray, y: NDArray) -> None:
        """
        The method to fit the neural network.
        """
        # Initializing the weights and bias
        num_elements = X.shape[1]  # Number of features.

        # Calculating the layer size
        layer_size = (
            [num_elements] + [self.layer_size_] * self.num_layers_ + [self.num_outputs_]
        )
        # Calculating the scaling factor for the Xavier Initialisation

        # The sigma term for Xavier initialisation is given as
        # sigma = sqrt(2 / (num_input + num_output))
        sigma = np.sqrt(2 / (np.array(layer_size[:-1]) + np.array(layer_size[1:])))

        # Determine problem type based on y vector
        problem_type = (
            "Classification"
            if np.issubdtype(y.dtype, np.integer) and np.unique(y).size < 20
            else "Regression"
        )

        # Initialize weights and biases
        weights = [
            np.random.standard_normal((layer_size[i + 1], layer_size[i])) * s
            for (i, s) in zip(range(len(layer_size) - 1), sigma)
        ]

        bias = [np.zeros((weight.shape[0], 1)) for weight in weights]

        for epoch in range(self.num_epochs_):

            # Storing the activations
            activations = [X.T]  # The first activation is the input vectors itself

            # Storing the z values
            z_values = []

            # Storing the loss for each iteration
            loss_history = []

            for itr, (weights, bias) in enumerate(zip(weights, bias)):

                print(f"Iteration number {itr}")

                # Calculating the dot product
                z = np.dot(weights, activations[-1]) + bias
                z_values.append(z)

                # Calculating the activation
                activation = self.activation_function_(z).forward()

                # Appending the activation functions
                activations.append(activation)

            # These calculations are for the back propagation
            y_pred = self.activation_function_(activations[-1]).forward().flatten()

            # Computing the loss
            if problem_type == "Classification":
                y_pred = np.where(y_pred >= 0.5, 1, 0)
                loss = log_loss(y, y_pred)
            else:
                loss = mean_squared_error(y, y_pred)

            loss_history.append(loss)

            # Backpropagation
            for i in reversed(range(len(weights))):

                if (
                    i == len(weights) - 1
                ):  # Calculating the gradients at the final node (Output)

                    difference = (2 / num_elements) * (
                        activations[-1] - y.reshape(-1, 1)
                    )

                    # `delta` is the gradient of the loss with respect to z
                    delta = np.dot(
                        difference, self.activation_function_(z[i]).derivative()
                    )
                else:

                    delta = (
                        np.dot(weights[i + 1].T, delta)
                        * self.activation_function_(z_values[i]).derivative()
                    )

                # Compute gradients for weights and biases
                gradient_weight = np.dot(delta, activations[i].T)
                gradient_bias = (
                    np.sum(delta, axis=1, keepdims=True)
                    if delta.ndim > 1
                    else np.sum(delta)
                )

                # Update weights and biases
                weights[i] -= self.learning_rate_ * gradient_weight
                bias[i] -= self.learning_rate_ * gradient_bias

        if self.verbose_ and problem_type == "Classification":
            print(f"The f1 score for epoch {epoch} is {f1_score(y, y_pred)}")

        elif self.verbose_ and problem_type == "Regression":
            print(f"The MSE for epoch {epoch} is {mean_squared_error(y, y_pred)}")

        # Storing the weights, bias, activations
        self.weights_ = weights
        self.bias_ = bias
        self.activations_ = activations

    def predict(self, X: NDArray) -> NDArray:
        """
        The main method for predicting the output.
        """
        activations = X
        for w, b in zip(self.weights_, self.bias_):
            z = np.dot(activations, w) + b
            activations = self.activation_function_.forward(z)
        return (activations > 0.5).astype(int)


if __name__ == "__main__":
    # These modules will be used for validating the current implementation
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler

    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    feature_matrix, target_matrix = make_classification(
        n_samples=1000,
        n_features=3,
        n_informative=2,
        n_redundant=1,
        n_clusters_per_class=1,
        random_state=42,
    )

    # Standardising the features
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)

    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, target_matrix, random_state=42
    )

    # Initialising the neural network model

    network = NeuralNetworks(
        num_layers=2, layer_sizes=4, activation_function="sigmoid", optimiser="batch"
    )

    network.fit(X_train, y_train)

    y_pred = network.predict(X_test)

    print(classification_report(y_test, y_pred))
