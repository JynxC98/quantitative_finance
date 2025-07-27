"""
Characteristic Function Library for Option Pricing Models.

This module defines characteristic functions for various stochastic models,
such as Black-Scholes, which are compatible with Fourier-based pricing 
techniques like the Carr–Madan framework.

These functions return the characteristic function φ(u) = E[e^{iu log(S_T)}]
under the risk-neutral measure and are designed to be passed directly to 
Fourier-based pricing engines.

Author: Harsh Parikh  
Date: July 27, 2025
"""
