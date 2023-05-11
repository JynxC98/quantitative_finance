"""
Script to simulate variance reduction methods. 
"""


class VarianceReduction:
    """
    Initialisation class of variance reduction technique using control variate method.

    Control variate method for call options is defined by:

    C(S, T) = Payoff(S(t)) - beta * (X - E[X])

    Where,

    Payoff(S(t)) = exp(-r * T) * max(S(T) - K, 0)

    X: exp(-r * T) S(t)

    E[X]: S_0(Initial stock price)
    """

    def __init__(self, **kwargs):
        """
        Initialisation class of the model:

        Input Parameters:
        -----------------

        S_0: Inital Stock Price
        T: Time to maturity
        """
