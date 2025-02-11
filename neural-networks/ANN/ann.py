"""
A script to simulate the artificial neural networks.

Author: Harsh Parikh

Date: 04-02-2024
"""

from typing import Tuple, List
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import f1_score, mean_squared_error, log_loss, r2_score

from optimisation_modules.activation_functions import Sigmoid, ReLU, Tanh, Softplus


class NeuralNetworks:
    """
    The main class to initialise the multilayer perceptron model.

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

    def _compute_loss(self, y_true: NDArray, y_pred: NDArray) -> float:
        """
        Compute appropriate loss based on problem type.
        """
        if self.problem_type_ == "Classification":
            y_pred_binary = np.where(y_pred >= 0.5, 1, 0)
            return log_loss(y_true, y_pred_binary)

        # Loss for classification
        return mean_squared_error(y_true, y_pred)

    def _compute_metric(self, y_true: NDArray, y_pred: NDArray) -> float:
        """
        This method evaluates the goodness of the fit for the neural network.
        """
        if self.problem_type_ == "Classification":
            y_pred_binary = np.where(y_pred >= 0.5, 1, 0)
            return f1_score(y_true, y_pred_binary)

        # Loss for regression
        return r2_score(y_true, y_pred)

    def _batch_gradient_descent(self, X: NDArray, y: NDArray) -> None:
        """
        This method performs the batch gradient descent
        """

        # Calculating the z values and the activation
        z_values, activations = self._forward_pass(X)

        # Updating the gradients based on backpropogation
        graidents = self._backpropogation(z_values, activations, y)

        for j in range(len(self.weights_)):
            self.weights_[j] -= self.learning_rate_ * graidents["weights"][j]
            self.bias_[j] -= self.learning_rate_ * graidents["bias"][j]

    def _mini_batch(self, X: NDArray, y: NDArray, batch_size: int = 32) -> None:
        """
        This method performs mini-batch gradient descent.
        """

        num_samples = y.shape[0]

        # Processing the mini-batch
        indices = np.random.permutation(num_samples)

        # Shuffling the feature and target values based on index
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for batch_start in range(0, X.shape[0], batch_size):

            # This LOC prevents the index overflow
            batch_end = min(batch_start + batch_size, num_samples)

            X_batch = X_shuffled[batch_start:batch_end]
            y_batch = y_shuffled[batch_start:batch_end]

            # Calculating the z values and the activation
            z_values, activations = self._forward_pass(X_batch)

            # Updating the gradients based on backpropogation
            graidents = self._backpropogation(z_values, activations, y_batch)

            # Updating the weights and bias
            for j in range(len(self.weights_)):

                self.weights_[j] -= self.learning_rate_ * graidents["weights"][j]
                self.bias_[j] -= self.learning_rate_ * graidents["bias"][j]

    def _stochastic_gradient_descent(
        self, X: NDArray, y: NDArray
    ) -> None:  # Placeholder, to be implemented later.
        """ """

    def _forward_pass(self, X: NDArray) -> Tuple[List[NDArray], List[NDArray]]:
        """
        The method to perform the forward pass in the neural network
        """
        activations = [X.T]  # The first activation is the input vectors itself
        z_values = []
        for itr, (weight, bias) in enumerate(zip(self.weights_, self.bias_)):

            # Computing the dot product
            z = np.dot(weight, activations[-1]) + bias
            z_values.append(z)

            # Activation = f(z)
            if (itr == len(self.weights_) - 1) and self.problem_type_ == "Regression":
                activations.append(z)
            else:
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

        gradients = {
            "weights": [np.zeros_like(weight) for weight in self.weights_],
            "bias": [np.zeros_like(bias) for bias in self.bias_],
        }
        # Initialize delta for the output layer
        delta = None

        for i in reversed(range(len(self.weights_))):
            # At terminal node, the cost is calculated for backpropagation
            if i == (len(self.weights_) - 1):
                # Loss is given by :
                # Loss = (a[L] - y)^2

                # Calculating the derivative of the loss at the final node
                difference = (2 / num_elements) * (activations[-1] - y.T)

                # `delta` is the gradient of the loss with respect to z

                if self.problem_type_ == "Regression":
                    delta = difference  # Since the output node for regression does not have an activation function.
                else:
                    delta = (
                        difference * self.activation_function_(z_values[i]).derivative()
                    )
            else:
                # Backpropagate the error in the hidden layers.
                delta = (
                    np.dot(self.weights_[i + 1].T, delta)
                    * self.activation_function_(z_values[i]).derivative()
                )

            # Computing the weight gradients based on the chain rule
            gradients["weights"][i] = np.dot(delta, activations[i].T)

            # Computing the bias gradient based on the chain rule
            gradients["bias"][i] = np.sum(delta, axis=1, keepdims=True)

        return gradients

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
        if self.activation_function_ == "relu":  # He initialisation for ReLU
            self.weights_ = [
                np.random.randn(layer_size[i + 1], layer_size[i])
                * np.sqrt(2 / layer_size[i])
                for i in range(len(layer_size) - 1)
            ]
        else:

            # Xavier initialisation for other activation functions.
            self.weights_ = [
                np.random.standard_normal((layer_size[i + 1], layer_size[i])) * s
                for (i, s) in zip(range(len(layer_size) - 1), sigma)
            ]

        self.bias_ = [np.zeros((weight.shape[0], 1)) for weight in self.weights_]

        for epoch in range(self.num_epochs_):
            # Implementing forward pass in the neural network

            if self.optimiser_ == "mini-batch":
                _ = self._mini_batch(X=X, y=y)

            elif self.optimiser_ == "batch":
                _ = self._batch_gradient_descent(X=X, y=y)

            # Predicting the output for the current epoch
            y_pred = self.predict(X)

            # Computing the loss
            if self.problem_type_ == "Classification":
                y_pred = np.where(y_pred >= 0.5, 1, 0)
                loss = log_loss(y, y_pred)
                # Printing the progress if verbose is True
                if self.verbose_:
                    print(f"The f1 score for epoch {epoch} is {f1_score(y, y_pred)}")
            else:
                loss = mean_squared_error(y, y_pred)
                if self.verbose_:
                    print(f"The r2 for epoch {epoch} is {r2_score(y, y_pred)}")

            # Storing the loss history
            self.loss_history_.append(loss)

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
        for itr, (w, b) in enumerate(zip(self.weights_, self.bias_)):

            z = np.dot(w, activations) + b
            # For regression, the output node needs to be a linear function.
            if itr == (len(self.weights_) - 1) and self.problem_type_ == "Regression":
                return z.flatten()

            # For classification, all nodes require an activation function.
            activations = self.activation_function_(z).forward()

        return activations.T
