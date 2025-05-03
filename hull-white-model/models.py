"""
A script to calculate the Hull-White model using the affine term structure
formula.

Author: Harsh Parikh 
Date: 18-02-2025
"""

import numpy as np
from numba import jit


class HullWhite:
    """
    Hull-White short-rate model using the affine term structure formulation.

    The Hull-White model is defined as:
        dr(t) = [θ(t) - a * r(t)] dt + σ dW(t)

    This class supports fitting the model parameters (a, σ) using observed
    zero-coupon bond prices or yields, and computing the θ(t) term needed
    for calibration.

    Attributes:
    -----------
    a : float
        Mean reversion rate.
    sigma : float
        Volatility of the short rate.
    curve_times : ndarray
        Array of time points corresponding to the yield curve.
    curve_rates : ndarray
        Observed zero rates or discount factors at the specified times.
    """

    def __init__(self, a, sigma, curve_times, curve_rates):
        """
        Initialize the Hull-White model with required parameters.

        Parameters:
        -----------
        a : float
            Mean reversion rate.
        sigma : float
            Volatility of the short rate.
        curve_times : ndarray
            Array of time points for the input yield curve.
        curve_rates : ndarray
            Array of observed market zero-coupon bond yields or rates.
        """

    def _calculate_theta(self):
        """
        Calculate the θ(t) term needed to exactly fit the initial term structure.

        This method uses numerical differentiation to approximate the first
        and second derivatives of the forward rate curve.

        Returns:
        --------
        theta : ndarray
            Array of θ(t) values corresponding to curve_times.
        """

    def fit(self):
        """
        Calibrate the Hull-White model using the affine term structure.

        This method uses the initial yield curve to determine the θ(t) term
        such that the model matches the observed curve exactly.

        Returns:
        --------
        theta : ndarray
            Calibrated θ(t) function over the curve times.
        """
        return self._calculate_theta()


if __name__ == "__main__":

    pass
