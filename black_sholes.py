from numpy import exp, sqrt, log
from scipy.stats import norm


class BlackSholes:
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

        d1 = (
            log(current_price / strike_price)
            + (interest_rate + 0.5 * volatility**2) * time_to_maturity
        ) / (volatility * sqrt(time_to_maturity))

        d2 = d1 - volatility * sqrt(time_to_maturity)

        call_price = current_price * norm.cdf(d1) - (
            strike_price * exp(-(interest_rate * time_to_maturity)) * norm.cdf(d2)
        )

        put_price = (
            strike_price * exp(-(interest_rate * time_to_maturity)) * norm.cdf(-d2)
        ) - (current_price * norm.cdf(-d1))

        self.call_price = call_price
        self.put_price = put_price

        self.call_delta = norm.cdf(d1)
        self.put_delta = norm.cdf(d1) - 1

        self.call_gamma = norm.pdf(d1) / (
            current_price * volatility * sqrt(time_to_maturity)
        )
        self.put_gamma = self.call_gamma
