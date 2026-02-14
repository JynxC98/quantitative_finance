"""
Implementation of European option pricing under the Mixed-Exponential 
Jump-Diffusion (MEJD) model.

This script implements the framework proposed in:

    Kou, S. G. (2011). "Option Pricing Under a Mixed-Exponential Jump Diffusion
    Model"

    Management Science.
    http://www.columbia.edu/~sk75/mixedExpManagementSci.pdf

The Mixed-Exponential Jump-Diffusion model extends the classical 
Black-Scholes framework by incorporating jumps with a mixed-exponential 
distribution. This structure preserves analytical tractability while 
allowing flexible approximation of arbitrary jump size distributions.

Author: Harsh Parikh
"""

import numpy as np
from numba import jit
