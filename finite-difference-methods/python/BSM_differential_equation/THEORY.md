# Black-Scholes PDE Solver

The following is an implementation of the Black-Scholes PDE solver using finite-difference methods. The underlying process is assumed to be a geometric Brownian motion and is given by:
$$
dS_t = S_t \cdot (rdt + \sigma dW_t)
$$
The process is assumed to follow an equivalent martingale measure, where the final asset price is given as follows:
$$
S_T = S_t \cdot e^{rT}
$$
The differential form of the Black-Scholes process is given as follows:
$$
\frac{\partial V}{\partial t} + rS \frac{\partial V}{\partial S} + \frac{1}{2} \sigma^2 \frac{\partial^2 V}{\partial S^2} - rV = 0
$$