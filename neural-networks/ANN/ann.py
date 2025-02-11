"""
A script to simulate the artificial neural networks for classification.

Author: Harsh Parikh

Date: 04-02-2024
"""

import sys
from typing import Tuple, List
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
        self.loss_history_ = []

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

    def _forward_pass(self, X: NDArray) -> Tuple[List[NDArray], List[NDArray]]:
        """
        The method to perform the forward pass in the neural network
        """
        activations = [X.T]  # The first activation is the input vectors itself
        z_values = []
        for weight, bias in zip(self.weights_, self.bias_):

            # Computing the dot product
            z = np.dot(weight, activations[-1]) + bias
            z_values.append(z)

            # Activation = f(z)
            activation = self.activation_function_(z).forward()
            activations.append(activation)
        return z_values, activations

    def _backpropogation(
        self,
        z_values: List[NDArray],
        activations: List[NDArray],
        y: NDArray,
    ) -> None:
        """
        Implements backpropagation using the chain rule.

        Input
        -----
        z_values: List of z values from the forward pass.
        activations: List of activations from the forward pass.
        y: The target values.
        """
        # Calculating the number of input vectors
        num_elements = activations[0].shape[
            0
        ]  # The first activation is the input vectors.

        # Initialize delta for the output layer
        delta = None

        for i in reversed(range(len(self.weights_))):
            # At terminal node, the cost is calculated for backpropagation
            if i == (len(self.weights_) - 1):
                # Calculating the loss at the final node
                difference = (2 / num_elements) * (activations[-1] - y.T)

                # `delta` is the gradient of the loss with respect to z
                delta = difference * self.activation_function_(z_values[i]).derivative()
            else:
                # Backpropagate the error
                delta = (
                    np.dot(self.weights_[i + 1].T, delta)
                    * self.activation_function_(z_values[i]).derivative()
                )

            # Computing the weight gradients based on the chain rule
            gradient_weight = np.dot(delta, activations[i].T)

            # Computing the bias gradient based on the chain rule
            gradient_bias = np.sum(delta, axis=1, keepdims=True)

            # Updating the weights and the bias
            self.weights_[i] -= self.learning_rate_ * gradient_weight
            self.bias_[i] -= self.learning_rate_ * gradient_bias

    def fit(self, X: NDArray, y: NDArray) -> None:
        """
        The method to fit the neural network.

        Input
        -----
        X: The feature matrix.
        y: The target values.
        """
        # Initializing the weights and bias
        num_elements = X.shape[1]  # Number of features.

        # Calculating the layer size
        layer_size = (
            [num_elements] + [self.layer_size_] * self.num_layers_ + [self.num_outputs_]
        )

        # Calculating the scaling factor for the Xavier Initialisation
        sigma = np.sqrt(2 / (np.array(layer_size[:-1]) + np.array(layer_size[1:])))

        # Determine problem type based on y vector
        self.problem_type_ = (
            "Classification"
            if np.issubdtype(y.dtype, np.integer) and np.unique(y).size < 20
            else "Regression"
        )

        # Initialize weights and biases
        self.weights_ = [
            np.random.standard_normal((layer_size[i + 1], layer_size[i])) * s
            for (i, s) in zip(range(len(layer_size) - 1), sigma)
        ]
        self.bias_ = [np.zeros((weight.shape[0], 1)) for weight in self.weights_]

        for epoch in range(self.num_epochs_):
            # Implementing forward pass in the neural network
            z_values, activations = self._forward_pass(X)

            # Implementing backpropagation
            self._backpropogation(z_values=z_values, activations=activations, y=y)

            # Predicting the output for the current epoch
            y_pred = self.predict(X)

            # Computing the loss
            if self.problem_type_ == "Classification":
                y_pred = np.where(y_pred >= 0.5, 1, 0)
                loss = log_loss(y, y_pred)
            else:
                loss = mean_squared_error(y, y_pred)

            # Storing the loss history
            self.loss_history_.append(loss)

            # Printing the progress if verbose is True
            if self.verbose_ and self.problem_type_ == "Classification":
                print(f"The f1 score for epoch {epoch} is {f1_score(y, y_pred)}")
            elif self.verbose_ and self.problem_type_ == "Regression":
                print(f"The MSE for epoch {epoch} is {mean_squared_error(y, y_pred)}")

    def predict(self, X: NDArray) -> NDArray:
        """
        The main method for predicting the output.

        Input
        -----
        X: The feature matrix.

        Output
        ------
        NDArray: The predicted output.
        """
        activations = X.T
        for w, b in zip(self.weights_, self.bias_):
            z = np.dot(w, activations) + b
            activations = self.activation_function_(z).forward()
        return np.where(activations > 0.5, 1, 0).T


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
