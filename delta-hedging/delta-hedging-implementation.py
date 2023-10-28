""" Script for simulating delta hedging
"""
import numpy as np
from black_scholes import call_option_price, put_option_price


def delta_hedge(S0, r, T, n, N, mu, sigma, K, option_type="call"):
    """
    Input parameters:
    -----------------
    S0: Initial stock price.
    r: Risk free rate.
    T: Time to maturity.
    n: Exponent in the number of time steps.
    N: Number of Monte-Carlo steps.
    mu: Asset return.
    sigma: Volatility.
    K: Strike Price.
    option_type: Call/Put type.

    """
    option_params = (S0, K, T, r, sigma)
    n_steps = pow(2, n)
    if option_type == "call":
        option_price = call_option_price(*option_params)
    else:
        option_price = put_option_price(*option_params)

    time_steps = T / n_steps
    time_discetisation = np.linspace(0, T, n_steps)
    dW = np.zeros([n_steps, N])
    dW[2:n_steps, :] = np.random.normal(0, 1, size=(n_steps - 1, N)) * np.sqrt(
        time_steps
    )
    W = np.cumsum(dW, axis=1)


# nstep = power(2,n); %number of time steps
# tstep = T/nstep; %time step size
# tdisc = linspace(0,T,nstep+1);
# dW = zeros(nstep+1,N);
# dW(2:nstep+1,1:N) = normrnd(0,1,[nstep,N])*sqrt(tstep); %Brownian motion increments
# W = cumsum(dW,1); %Brownian motion paths
# tgrid = (tdisc.').*ones(nstep+1,N); %discretised time grid
# incr = sigma*W + (mu-0.5*sigma*sigma)*tgrid; %expression in the Black-Scholes price formula
# ST = S0*exp(incr); %Black-Scholes price path
# tt = ((T - tdisc).').*ones(nstep+1,N); %time to maturity
# dfmat = exp(-r*tgrid); %discount factor

# d1 = (log(ST(1:nstep,1:N)./(K*exp(-r*tt(1:nstep,1:N)))) + 0.5*sigma*sigma*tt(1:nstep,1:N))./(sigma*sqrt(tt(1:nstep,1:N)));
# delta = normcdf(d1); %delta at different time steps excluding the final maturity time
# pricediff = ST(2:nstep+1,1:N).*dfmat(2:nstep+1,1:N) - ST(1:nstep,1:N).*dfmat(1:nstep,1:N); %difference between two consecutive values of discounted asset price
# sumholding = sum(delta.*pricediff,1); %final value of the holding in the underlying asset
# X = exp(r*T)*(cprice + sumholding); %final value of the self-financing portfolio
# upayoff = (ST(end,1:N)-K);
# payoff = upayoff.*(upayoff>0); %payoff from option exercise
# PNL = X - payoff; %Profit and loss over Monte Carlo sample paths
