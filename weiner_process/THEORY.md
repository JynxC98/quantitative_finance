# Weiner process and its properties

## Brownian motion

Brownian motion is a continuous stochastic process.

It has the following properties:

1. $W_0 = 0$
2. $W_0$ has independent increments: every future value is independent of the past values.
3. $W$ has [Guassian increments](https://math.stackexchange.com/questions/3563037/independent-increment-and-gaussian-increments-process-is-a-gaussian-process): $W_{t+du} - W_t$ is normally distributed with mean zero and variance $du$. 


Consider the differential equation 

$dx(t) = \underbrace{f(t, x(t))dt}_{drift} + \underbrace{\sigma(t, x(t))dW(t)}_{diffusion}$

Here, the first part is the deterministic parameter and the second part is the stochastic parameter.


#### Stock price process under stochastic differential equation

Since stock price cannot be negative, we cannot use a standard Weiner process.
Hence, we use geometric brownian motion to model stock price changes.

Stock price $S$ under risk-neutral probability measure is given by:
$$
dS_t = rS_tdt + \sigma S_tdW_t
$$
The stochastic differential equation can be solved, explicitly leading to the result:
$$
S_t = S_0\exp(\sigma W_t + (r - \frac{\sigma^{2}}{2})t)
$$
