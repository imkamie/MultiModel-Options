import numpy as np

from scipy.stats import norm


class Bachelier:
    """
    European option pricing under the Bachelier (normal) model, no dividends (q = 0).

    Assumptions:
        - time_to_maturity > 0
        - volatility > 0 and is a *normal* (absolute) vol in price units per sqrt(year),
          not a lognormal vol (i.e., sigma_N with the same units as the underlying).
        - current_price > 0, strike_price > 0
        - interest_rate is the continuously compounded risk-free rate.

    Notation (inside run()):
        F   : forward price with q=0, F = S * e^{rT}
        disc: discount factor, e^{-rT}
        d   : (F - K) / (sigma_N * sqrt(T))
        Call = disc * [ (F - K) N(d) + sigma_N sqrt(T) φ(d) ]
        Put  = disc * [ (K - F) N(-d) + sigma_N sqrt(T) φ(d) ]

    Notes:
        - Put-call parity holds: Call - Put = disc * (F - K).
        - With continuous dividend yield q, replace F = S * e^{(r-q)T}.
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
        time_to_maturity = self.time_to_maturity
        current_price = self.current_price
        strike_price = self.strike_price
        interest_rate = self.interest_rate
        volatility = self.volatility

        # Discount factor e^(-r*T)
        disc = np.exp(-interest_rate * time_to_maturity)

        # Forward price under q=0: F = S * e^(r*T)
        # (If you later add a dividend yield q, use F = S * e^((r - q)T).)
        F = current_price * np.exp(interest_rate * time_to_maturity)

        # Normalized moneyness under the normal model
        # (volatility is absolute; denominator has units of price)
        d = (F - strike_price) / (volatility * np.sqrt(time_to_maturity))

        # Bachelier (normal) call price:
        # C = disc * [ (F - K) N(d) + sigma_N sqrt(T) φ(d) ]
        self.call_price = disc * (
            (F - strike_price) * norm.cdf(d)
            + volatility * np.sqrt(time_to_maturity) * norm.pdf(d)
        )

        # Bachelier (normal) put price:
        # P = disc * [ (K - F) N(-d) + sigma_N sqrt(T) φ(d) ]
        self.put_price = disc * (
            (strike_price - F) * norm.cdf(-d)
            + volatility * np.sqrt(time_to_maturity) * norm.pdf(d)
        )
