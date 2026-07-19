"""
noncentral_chi2.py
------------------
Non-central chi-squared distribution: PDF evaluation and exact sampling.

The non-central chi-squared distribution χ²(k, λ) arises naturally in the
Broadie-Kaya exact simulation scheme as the transition distribution of the
CIR variance process:

    V_u | V_t ~ (σ²(1 - e^{-κΔt}) / 4κ) * χ²(d, λ)

where:
    d = 4κθ/σ²          (degrees of freedom)
    λ = 4κ e^{-κΔt} V_t / (σ²(1 - e^{-κΔt}))   (non-centrality parameter)

Two functions are provided:
  - noncentral_chi2_pdf    : exact PDF via modified Bessel function
  - sample_noncentral_chi2 : exact sampler via Poisson-Gamma mixture

Reference:
    https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution

Author: Harsh Parikh (Python port)
"""

import numpy as np
from .bessel import modified_bessel


def noncentral_chi2_pdf(
    z: float,
    dof: float,
    lambda_: float,
    num_iterations: int = 100,
    tolerance: float = 1e-8,
) -> float:
    """
    Evaluate the PDF of the non-central chi-squared distribution χ²(dof, λ) at z.

    The PDF is:
        f(z) = ½ * exp(-½(z+λ)) * (z/λ)^(d/4 - ½) * I_{d/2-1}(√(λz))

    where I_α is the modified Bessel function of the first kind, evaluated
    in log-space to avoid overflow for large arguments.

    Parameters
    ----------
    z : float
        Evaluation point. Returns 0.0 for z ≤ 0.
    dof : float
        Degrees of freedom k > 0. In Heston: d = 4κθ/σ².
    lambda_ : float
        Non-centrality parameter λ ≥ 0.
        In Heston: λ = 4κ e^{-κΔt} V_t / (σ²(1-e^{-κΔt})).
    num_iterations : int
        Maximum iterations for the Bessel function series.
    tolerance : float
        Convergence tolerance for the Bessel function.

    Returns
    -------
    float
        PDF value f(z; dof, λ). Returns 0.0 for z ≤ 0.

    Raises
    ------
    ValueError
        If dof ≤ 0 or lambda_ < 0.

    Examples
    --------
    >>> from scipy.stats import ncx2
    >>> z, dof, lam = 3.0, 4.0, 2.0
    >>> abs(noncentral_chi2_pdf(z, dof, lam) - ncx2.pdf(z, dof, lam)) < 1e-8
    True
    """
    if dof <= 0.0:
        raise ValueError(f"Degrees of freedom must be positive, got {dof}")
    if lambda_ < 0.0:
        raise ValueError(
            f"Non-centrality parameter must be non-negative, got {lambda_}"
        )

    # PDF is zero on the non-positive reals
    if z <= 0.0:
        return 0.0

    # --- Exponential term: ½ exp(-½(z + λ)) ---
    exp_term = 0.5 * np.exp(-0.5 * (z + lambda_))

    # --- Centrality term: (z/λ)^(d/4 - ½) ---
    # Computed in log-space: exp((d/4 - 0.5) * log(z/λ))
    cent_term = np.exp((dof / 4.0 - 0.5) * np.log(z / lambda_))

    # --- Bessel term: I_{d/2-1}(√(λz)) ---
    sqrt_arg = np.sqrt(lambda_ * z)
    alpha = 0.5 * dof - 1.0

    # Threshold scales with order to stay in power-series regime when needed
    threshold = abs(alpha) + 10.0

    log_bessel = modified_bessel(
        complex(sqrt_arg),
        alpha,
        num_iterations=num_iterations,
        tolerance=tolerance,
        threshold=threshold,
        log_space=True,
    )

    # Warn if imaginary part is unexpectedly large (should be ~0 for real input)
    if abs(log_bessel.imag) > 1e-10:
        import warnings

        warnings.warn(
            f"Non-zero imaginary part in Bessel evaluation: {log_bessel.imag:.2e}. "
            "Check input parameters.",
            RuntimeWarning,
        )

    bessel_term = np.exp(log_bessel.real)

    return exp_term * cent_term * bessel_term


def sample_noncentral_chi2(
    dof: float,
    lambda_: float,
    size: int = 1,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Draw exact samples from the non-central chi-squared distribution χ²(dof, λ).

    Uses the Poisson-Gamma mixture representation, which avoids direct
    evaluation of Bessel functions and is numerically stable for all
    valid parameter values:

        1. Draw N ~ Poisson(λ/2)
        2. Draw X ~ Gamma(shape=(dof + 2N)/2, scale=2)   [= χ²(dof + 2N)]
        3. Return X

    This works because χ²(k, λ) = χ²(k + 2N) where N ~ Poisson(λ/2),
    and a central χ²(k) is Gamma(k/2, 2).

    Parameters
    ----------
    dof : float
        Degrees of freedom k > 0. In Heston: d = 4κθ/σ².
    lambda_ : float
        Non-centrality parameter λ ≥ 0.
        In Heston: λ = 4κ e^{-κΔt} V_t / (σ²(1-e^{-κΔt})).
    size : int
        Number of samples to draw.
    rng : np.random.Generator, optional
        Random number generator. If None, uses np.random.default_rng().
        Pass a seeded generator for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (size,) containing i.i.d. samples.

    Raises
    ------
    ValueError
        If dof ≤ 0 or lambda_ < 0.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> samples = sample_noncentral_chi2(4.0, 2.0, size=100_000, rng=rng)
    >>> abs(samples.mean() - (4.0 + 2.0)) < 0.05   # E[X] = dof + lambda
    True
    """
    if dof <= 0.0:
        raise ValueError(f"Degrees of freedom must be positive, got {dof}")
    if lambda_ < 0.0:
        raise ValueError(
            f"Non-centrality parameter must be non-negative, got {lambda_}"
        )

    if rng is None:
        rng = np.random.default_rng()

    # Step 1: N ~ Poisson(λ/2)
    n_poisson = rng.poisson(lam=lambda_ / 2.0, size=size)

    # Step 2: X ~ Gamma(shape=(dof + 2N)/2, scale=2)
    # Vectorised: each sample has its own shape parameter
    shapes = (dof + 2.0 * n_poisson) / 2.0
    samples = rng.gamma(shape=shapes, scale=2.0)

    return samples
