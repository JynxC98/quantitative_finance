/**
 * @brief This script is used to calculate the value of the Bessel's function.
 * More details on the same can be found below:
 *
 * https://en.wikipedia.org/wiki/Bessel_function
 *
 * @author Harsh Parikh
 */
#if !defined(BESSEL_HPP)
#define BESSEL_HPP

#include <iostream>
#include <cmath>
#include <complex>
#include "gamma.hpp"

/**
 * @brief This function is the implementation of the power scheme for the
 * Bessel's function when |z| <= threshold.
 *
 * @param z: The point at which the function needs to be evaluated.
 * @param alpha: Order of the Bessel function
 * @param tolerance: The error term used to evaluate the convergence.
 * @param num_iterations: The number of terms for the expansion.
 *
 * @returns The logarithm of the Bessel value function at point z.
 */
std::complex<double> PowerScheme(std::complex<double> z,
                                 double alpha,
                                 double tolerance = 1e-8,
                                 int num_iterations = 100)
{
    // First term: k = 0
    std::complex<double> term = 1.0 / GammaFunction(alpha + 1.0);
    std::complex<double> sum = term;

    // The following formula is calculated using the recurrence method.
    /*
    term_k/term_{k-1} = 0.25 * z^2 * 1/(k(alpha + k))

    More details can be found in `CALCULATIONS.md` file.
    */
    for (int k = 1; k < num_iterations; ++k)
    {
        // term_k = term_{k-1} * (z^2/4) / (k * (alpha + k))
        term *= (0.25 * z * z) / (static_cast<double>(k) * (alpha + static_cast<double>(k)));
        sum += term;

        /*
        Here, the overall logic is as follows:

        If the value of the term is `x`, we need to ensure that the value
        is contributing a decent proportion to the overall sum. Hence, the
        proportion of x/sum should be greater than the tolerance in order to
        ensure a significant contribution.
        */
        if (std::abs(term) < tolerance * std::abs(sum))
        {
            // Returning the logarithmic variant
            return alpha * std::log(0.5 * z) + sum;
            // return std::pow(0.5 * z, alpha) * sum;
        }
    }
    throw std::runtime_error("PowerScheme: Convergence not achieved within iteration limit");
}

/**
 * @brief This function is the implementation of the asymptotic expansion for
 * the Bessel's function when |z| > threshold.
 *
 * @param z: The point at which the function needs to be evaluated.
 * @param alpha: Order of the Bessel function
 * @param tolerance: The error term used to evaluate the convergence.
 * @param num_iterations: The number of terms for the expansion.
 *
 * @returns The logarithm of the Bessel value function at point z.
 *
 * @note The asymptotic series is NOT convergent —
 * it is an asymptotic expansion that eventually diverges for any fixed z.
 *
 * The fix is a minimum-term stopping rule: halt as soon as the magnitude of
 * the current term exceeds that of the previous term, keeping only the
 * partial sum up to the smallest term.
 */
std::complex<double> AsymptoticExpansion(std::complex<double> z,
                                         double alpha,
                                         double tolerance = 1e-8,
                                         int num_iterations = 100)
{
    // First term, k=0
    std::complex<double> term(1.0, 0.0);
    std::complex<double> sum = term;
    double prev_abs = std::abs(term);

    for (int k = 1; k < num_iterations; ++k)
    {
        /*
        The asymptotic series for I_alpha(z) is an alternating series:

            I_alpha(z) ~ e^z / sqrt(2*pi*z) * sum_{k=0}^inf (-1)^k * a_k(alpha) / z^k

        where:

            a_k(alpha) = prod_{j=1}^{k} (4*alpha^2 - (2j-1)^2) / (k! * 8^k)

        The recurrence for the term ratio is:

            term_k / term_{k-1} = -(4*alpha^2 - (2k-1)^2) / (8 * k * z)

        Expanding with mk = (2k - 1):

            -(4*alpha^2 - mk^2) / (8 * k * z)

        The negation produces the required alternating signs: without it the
        series accumulates same-sign terms and converges to the wrong value.
        */
        double mk = 2.0 * static_cast<double>(k) - 1.0;
        term *= -(4.0 * alpha * alpha - mk * mk) /
                (8.0 * static_cast<double>(k) * z);

        double curr_abs = std::abs(term);

        /*
        Minimum-term stopping rule for asymptotic series:

        Unlike a convergent power series, this asymptotic expansion eventually
        diverges for any fixed z. Once the term magnitude starts growing again,
        adding further terms only increases the error. We therefore stop as
        soon as the current term is larger than the previous one, keeping only
        the partial sum accumulated so far (up to and not including the
        diverging term).

        This rule is applied BEFORE adding the current term to the sum so that
        only the well-behaved portion of the series is retained.
        */
        if (curr_abs > prev_abs)
        {
            break;
        }

        sum += term;

        /*
        Here, the overall logic is as follows:

        If the value of the term is `x`, we need to ensure that the value
        is contributing a decent proportion to the overall sum. Hence, the
        proportion of x/sum should be greater than the tolerance in order to
        ensure a significant contribution.
        */
        if (curr_abs < tolerance * std::abs(sum))
        {
            break;
        }

        prev_abs = curr_abs;
    }

    // For I_alpha(z), the leading factor is e^z / sqrt(2*pi*z)

    // Returning the logarithmic value to avoid overflow for large z
    return z - 0.5 * std::log(2.0 * M_PI * z) + std::log(sum);
}

/**
 * @brief This function stores calculates the value of the modified Bessel's
 * function for the value of `z`. The function uses power series for the small
 * value of |z| and asymptotic expansion for the large value of |z| depending
 * on the threshold. Mathematically, the Bessel function can have a complex
 * number as its order. However, in the implementation presented by
 * Brodie & Kaya, the order is a real number. Hence, for simplicity,
 * the order of the function in the current implementation is a real number.
 *
 * @param z: The point at which the function needs to be evaluated.
 * @param alpha: Order of the Bessel function
 * @param num_iterations: The upper value for the summation.
 * @param tolerance: The error term used to evaluate the convergence.
 * @param threshold: The threshold at which the method is decided.
 *
 * @returns: The logarithm of the Bessel function at `z` to avoid overflow.
 */
std::complex<double> ModifiedBessel(std::complex<double> z,
                                    double alpha,
                                    int num_iterations = 100,
                                    double tolerance = 1e-10,
                                    double threshold = 10.0)
{
    // NOTE: Negative integer orders (alpha < 0, integer) currently return
    // inf/nan due to a known bug in the symmetry redirect.
    // This does not affect Broadie-Kaya usage where alpha = d/2 - 1 > 0
    // for valid Heston parameters. Deferred for future fix.

    // Handling negative integer orders using symmetry
    if (alpha < 0.0 && std::abs(std::round(alpha) - alpha) < 1e-12)
    {
        // For negative integers, I_{-n}(z) = I_n(z)
        return ModifiedBessel(z, -alpha, num_iterations, tolerance, threshold);
    }

    // Base cases for zero values
    // For real order α:
    // I_0(0) = 1 (by analytic continuation)
    // I_α(0) = 0 for α > 0
    // I_α(0) diverges (→ ∞) for α < 0

    if (z == std::complex<double>(0.0, 0.0))
    {
        if (alpha < 0.0)
        {
            throw std::domain_error("I_alpha(0) diverges to infinity for alpha < 0");
        }
        if (alpha == 0.0)
        {
            return std::complex<double>(1.0, 0.0); // I_0(0) = 1
        }
        // alpha > 0
        return std::complex<double>(0.0, 0.0);
    }

    // Choose method based on threshold (not tolerance!)
    if (std::abs(z) <= threshold)
    {
        return PowerScheme(z, alpha, tolerance, num_iterations);
    }
    else
    {
        return AsymptoticExpansion(z, alpha, tolerance, num_iterations);
    }
}

#endif