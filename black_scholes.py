import numpy as np

from scipy.stats import norm


class BlackScholes:
    """
    European Black-Scholes (no dividends).
    Assumes:
        - time_to_maturity > 0
        - volatility > 0
        - current_price > 0, strike_price > 0
    Outputs (after run()):
        - call_price, put_price
        - call_delta, put_delta
        - call_gamma, put_gamma  (gamma is identical for call/put)
    """

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
        """
        Compute European call/put prices and (Delta, Gamma) using Black-Scholes.
        No dividend yield term (q=0). Greeks are with respect to S.
        """
        time_to_maturity = self.time_to_maturity
        current_price = self.current_price
        strike_price = self.strike_price
        interest_rate = self.interest_rate
        volatility = self.volatility

        # d1 and d2: standard BS terms
        # d1 = [ln(S/K) + (r + 0.5*sigma^2) T] / (sigma * sqrt(T))
        # d2 = d1 - sigma * sqrt(T)
        d1 = (
            np.log(current_price / strike_price)
            + (interest_rate + 0.5 * volatility**2) * time_to_maturity
        ) / (volatility * np.sqrt(time_to_maturity))

        d2 = d1 - volatility * np.sqrt(time_to_maturity)

        # European call/put prices (q=0):
        # C = S * N(d1) - K * e^(-r*T) * N(d2)
        # P = K * e^(-r*T) * N(-d2) - S * N(-d1)
        call_price = current_price * norm.cdf(d1) - (
            strike_price * np.exp(-(interest_rate * time_to_maturity)) * norm.cdf(d2)
        )

        put_price = (
            strike_price * np.exp(-(interest_rate * time_to_maturity)) * norm.cdf(-d2)
        ) - (current_price * norm.cdf(-d1))

        self.call_price = call_price
        self.put_price = put_price

        # Deltas:
        # Δ_call = N(d1),  Δ_put = N(d1) - 1   (no dividends)
        self.call_delta = norm.cdf(d1)
        self.put_delta = norm.cdf(d1) - 1

        # Gamma (same for call/put):
        # Γ = φ(d1) / (S * sigma * sqrt(T))
        # where φ is the standard normal pdf
        self.call_gamma = norm.pdf(d1) / (
            current_price * volatility * np.sqrt(time_to_maturity)
        )
        self.put_gamma = self.call_gamma
