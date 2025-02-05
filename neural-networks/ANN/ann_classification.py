"""
A script to simulate the artificial neural networks for classification.

Author: Harsh Parikh

Date: 04-02-2024
"""

import numpy as np
from optimisation_modules.gradient_models import (
    batch_gradient_descent,
    mini_batch_gradient_descent,
    stochastic_gradient_descent,
)
from optimisation_modules.activation_functions import Sigmoid, ReLU, Tanh

from optimisation_modules.forward_propagation import propagate_classification


class NeuralNetworksClassification:
    """
    The main class to initialise the multilayer perceptron model for classification.

    Input parameters:
    num_layers: int

    """
