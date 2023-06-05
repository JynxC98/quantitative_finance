""" Script that plots interest paid, total amount paid and principal
"""

from collections import defaultdict
import warnings
import numpy as np
import pandas as pd
import plotly.express as px

warnings.filterwarnings("ignore")


class Amortization:
    """
    Calculates Amortization
    """

    def __init__(self, interest: float, principal: float, n_years: int) -> None:
        """
        Initialisation class of Amortization.
        ------------------------------------
        Parameters:
        1. Interest: Annual interst rate
        2. Principal: Total loan value
        3. n_years: Loan period
        """

        self.interest = interest
        self.principal = principal
        self.n_years = n_years

    def calculate_periodic_payments(self) -> float:
        """
        Calculation of monthy emi
        """

        numerator = (
            (self.interest / 12)
            * (pow(1 + (self.interest / 12), self.n_years * 12))
            * self.principal
        )
        denominator = pow((self.interest / 12 + 1), self.n_years * 12) - 1

        periodic_payments = numerator / denominator

        return periodic_payments

    def monthly_interest(self, remaining_amount: float) -> float:
        """
        Calculates the amount of montly interest.
        """
        return remaining_amount * self.interest / 12

    def amortization_schedule(self) -> pd.DataFrame:
        """
        Generates monthly schedule of payments
        """

        schedule = defaultdict(list)
        monthly_installment_ = self.calculate_periodic_payments()

        period_ = 1
        schedule["Period"].append(period_)
        schedule["Total Payment"].append(np.nan)
        schedule["Computed Interest"].append(np.nan)
        schedule["Principal Due"].append(np.nan)
        schedule["Principal Balance"].append(self.principal)

        while period_ <= self.n_years * 12:
            schedule["Period"].append(period_)
            schedule["Total Payment"].append(monthly_installment_)
            previous_balance_ = list(schedule["Principal Balance"])[-1]

            monthly_interest_ = self.monthly_interest(previous_balance_)

            amount_towards_principal_ = monthly_installment_ - monthly_interest_

            schedule["Principal Balance"].append(
                previous_balance_ - amount_towards_principal_
            )
            schedule["Principal Due"].append(amount_towards_principal_)
            schedule["Computed Interest"].append(monthly_interest_)

            period_ += 1

        return pd.DataFrame(schedule)

    def plot_payment_and_interest(self) -> None:
        """
        Plots the overall principal left and cumulative amount paid over the period
        """
        data = self.amortization_schedule()
        data["Interest Paid"] = np.cumsum(data["Computed Interest"])
        data["Amount Paid"] = np.cumsum(data["Total Payment"])
        fig = px.line(
            data[["Amount Paid", "Principal Balance", "Interest Paid"]][1:],
            labels={"value": "Amount", "index": "Period"},
        )
        fig.show()


if __name__ == "__main__":
    dummy_data = Amortization(interest=0.145, principal=3e6, n_years=13)

    dummy_data.plot_payment_and_interest()
