"""
Implementation of Logistic Regression.
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

def propagate(weights, bias, X, y):
    """
    Calculates the gradients and cost for the `LogisticRegression` model.
    Args:
        - weights (numpy array): Model weights of shape (n,).
        - bias (float): Model bias.
        - X (numpy array): Training features of shape (m, n).
        - y (numpy array): Training labels of shape (m,).

    Returns:
    - gradients (dictionary): Gradients for weights and bias.
    - cost (numpy array): Cost associated with the model.
    """
    num_samples = X.shape[0]
    z = np.dot(X, weights) + bias
    probability_vector = sigmoid(z)
    
    epsilon = 1e-10  # Prevent log(0)
    cost = (1 / num_samples) * np.sum(
        (-np.log(probability_vector + epsilon) * y)
        + (-np.log(1 - probability_vector + epsilon) * (1 - y))
    )

    dW = np.dot(X.T, (probability_vector - y)) / num_samples
    dB = np.sum(probability_vector - y) / num_samples

    gradients = {"dW": dW, "dB": dB}
    return gradients, cost

class LogisticRegression:
    """
    Logistic Regression model for binary classification.

    Attributes:
    - X_train (numpy array): Training features of shape (m, n), where m is the number of samples and n is the number of features.
    - y_train (numpy array): Training labels of shape (m,).
    - weights_ (numpy array): Model weights of shape (n,).
    - bias_ (float): Model bias.
    - cost_ (list): List to store the cost during training.

    Methods:
    - fit(): Fit the logistic regression model to the training data.
    - predict(): Predict class labels for input data.
    """

    def __init__(self):
        """
        Initialise logistic regression model with training data.
        """
        self.X_train_ = None
        self.y_train_ = None
        self.weights_ = None
        self.bias_ = 0
        self.cost_ = []

    def fit(self, X_train, y_train, learning_rate=0.01, num_iterations=1000):
        """
        Fit the logistic regression model to the training data.

        Args:
        - learning_rate (float): Learning rate for gradient descent.
        - num_iterations (int): Number of iterations for gradient descent.
        """
        self.X_train_ = X_train
        self.y_train_ = y_train
        self.weights_ = np.zeros(X_train.shape[1])

        for _ in range(num_iterations):
            gradients, cost = propagate(
                self.weights_, self.bias_, self.X_train_, self.y_train_
            )
            dW = gradients["dW"]
            dB = gradients["dB"]

            self.weights_ -= learning_rate * dW
            self.bias_ -= learning_rate * dB

            self.cost_.append(cost)

    def predict(self, X, threshold=0.5):
        """
        Predict class labels for input data.

        Args:
        - X (numpy array): Input features of shape (m, n).

        Returns:
        - predictions (numpy array): Predicted class labels of shape (m,).
        """
        z = np.dot(X, self.weights_) + self.bias_
        probabilities = sigmoid(z)
        predictions = (probabilities > threshold).astype(int)
        return predictions
