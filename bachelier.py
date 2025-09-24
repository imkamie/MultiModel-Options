from numpy import exp, sqrt
from scipy.stats import norm


class Bachelier:
    def __init__(
        self,
        time_to_maturity: float,
        current_price: float,
        strike_price: float,
        interest_rate: float,
        volatility: float,
    ):
        self.time_to_maturity = time_to_maturity
        self.current_price = current_price
        self.strike_price = strike_price
        self.interest_rate = interest_rate
        self.volatility = volatility

    def run(self):
        time_to_maturity = self.time_to_maturity
        current_price = self.current_price
        strike_price = self.strike_price
        interest_rate = self.interest_rate
        volatility = self.volatility

        F = current_price * exp(interest_rate * time_to_maturity)
        d = (F - strike_price) / (volatility * sqrt(time_to_maturity))

        self.call_price = exp(-interest_rate * time_to_maturity) * (
            (F - strike_price) * norm.cdf(d)
            + volatility * sqrt(time_to_maturity) * norm.pdf(d)
        )

        self.put_price = exp(-interest_rate * time_to_maturity) * (
            (strike_price - F) * norm.cdf(-d)
            + volatility * sqrt(time_to_maturity) * norm.pdf(d)
        )

        # self.put_price = (
        #     self.call_price
        #     - current_price
        #     + exp(-interest_rate * time_to_maturity) * strike_price
        # )
