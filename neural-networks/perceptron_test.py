"""
This is the main function to execute the `CustomPerceptron` class from the
`perceptron` directory.

Author: Harsh Parikh
Date: 04-02-2025
"""

# These modules will be used for validating the current implementation
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import (
    Perceptron,
)
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Importing the custom perceptron library
from perceptron import CustomPerceptron


def main():
    """
    The main function for comparing the two methodologies.
    """
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

    # Initializing the `sklearn` implementation.
    perceptron = Perceptron()

    perceptron.fit(X_train, y_train)
    y_pred_perceptron = perceptron.predict(X_test)
    print("Classification report for sklearn implementation")
    print("*" * 100)
    print(classification_report(y_test, y_pred_perceptron))

    # Initializing the custom perceptron
    custom_percp = CustomPerceptron(optimizer_type="mini-batch", activation="relu")
    custom_percp.fit(X_train, y_train)

    y_pred_custom = custom_percp.predict(X_test)
    print("Classification report for custom implementation")
    print("*" * 100)

    print(classification_report(y_test, y_pred_custom))


if __name__ == "__main__":
    main()
