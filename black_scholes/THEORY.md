### Black-Scholes option pricing.


Black scholes call option pricing formula is given by:

$$
C\left(S_t, t\right) = S\phi\left(d_1\right) - K e^{-rt} \phi(d_2) \\
$$

Black scholes put option pricing formula is given by:
$$
P\left(S_t, t\right) = K e^{-rt} \phi(-d_2) - S\phi\left(-d_1\right)  \\
$$
where,
$$
d1 = \frac{\ln{\frac{S_t}{K}} + (r + \frac{\sigma^2}{2})t}{\sigma \sqrt{t}}
$$
$$
d2 = d1 - \sigma \sqrt{t}
$$
and 
$S_t$: Stock price
$t$: Time to maturity
$\phi$: Cumulative distribution function.
$K$: Strike Price
$r$: Risk free interest rate

### Monte-Carlo simulation

For simulation of stock price, we use the principle of [Markov Process](https://en.wikipedia.org/wiki/Markov_chain). This means that the stock price follows a random walk and is consistent with the weak form of efficient market hypothesis. 

The formula for monte-carlo process for GBM is given by:

$$
\frac{\delta S}{S} = \mu \delta{t} + \sigma \epsilon \sqrt{\delta{t}}
$$

