"""
This script is to test the neural network implementation.

Date: 11-02-2025
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ANN.ann import NeuralNetworks


def main_classification():
    """
    The main function for classification.
    """
    feature_matrix, target_matrix = make_classification(
        n_samples=10000,
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
        num_layers=2,
        layer_sizes=16,
        activation_function="tanh",
        optimiser="mini-batch",  # Don't use the `sgd` method, not yet functional
    )

    network.fit(X_train, y_train)

    y_pred = network.predict(X_test)
    y_pred = np.where(y_pred >= 0.5, 1, 0)

    print(classification_report(y_test, y_pred))


def main_regression():
    """
    The main function for regression.
    """
    feature_matrix, target_matrix = make_regression(
        n_samples=10000, n_features=3, n_informative=2, noise=0.1, random_state=42
    )

    # Standardizing the features
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)

    # Standardizing the target variable for better training
    target_scaler = StandardScaler()
    target_matrix = target_scaler.fit_transform(target_matrix.reshape(-1, 1))

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, target_matrix, random_state=42
    )

    # Initializing the neural network model
    network = NeuralNetworks(
        num_layers=2,
        layer_sizes=32,
        activation_function="relu",  # ReLU often works better for regression
        optimiser="mini-batch",
        learning_rate=0.01,  # Lower learning rate for stability
        num_epochs=1000,
        verbose=True,
    )

    # Training the model
    network.fit(X_train, y_train)

    # Making predictions
    y_pred = network.predict(X_test).reshape(-1, 1)

    # Converting predictions back to original scale
    y_pred_original = target_scaler.inverse_transform(y_pred)
    y_test_original = target_scaler.inverse_transform(y_test)

    # Calculating metrics
    mse = mean_squared_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, y_pred_original)

    print("\nRegression Metrics:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")


if __name__ == "__main__":
    main_regression()
