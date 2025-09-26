import numpy as np


class BinomialTree:
    """
    Cox-Ross-Rubinstein (CRR) binomial tree with continuous dividend yield,
    supporting European and American valuation for calls and puts.

    After calling run():
        - call_P, put_P: upper-triangular value trees (V[i, j] valid for 0 <= j <= i)
        - call_exercise, put_exercise: early-exercise masks for American; None for European
        - S_tree: underlying price tree
        - Prices at t=0 are call_P[0, 0] and put_P[0, 0]
    """

    def __init__(
        self,
        steps: int,
        time_to_maturity: float,
        strike_price: float,
        current_price: float,
        volatility: float,
        interest_rate: float,
        dividend_yield: float,
        is_american: bool,
    ):
        self.steps = steps
        self.time_to_maturity = time_to_maturity
        self.strike_price = strike_price
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate
        self.dividend_yield = dividend_yield
        self.is_american = is_american

        # Per-step time increment
        self.delta_time = time_to_maturity / steps

        # If T=0 (=> dt=0), we build a degenerate (flat) tree: u=d=1
        # Otherwise standard CRR up/down multipliers u=exp(sigma*sqrt(dt)), d=1/u
        if self.delta_time == 0:
            self.up = 1.0
            self.down = 1.0
        else:
            self.up = np.exp(volatility * np.sqrt(self.delta_time))
            self.down = np.exp(-(volatility * np.sqrt(self.delta_time)))

        # One-step discount factor e^(-r*dt)
        self.disc = np.exp(-(interest_rate * self.delta_time))

        # Risk-neutral probability with dividend yield q:
        # p = (e^{(r - q)*dt} - d) / (u - d)
        # If u == d (dt=0 here), p is irrelevant to pricing; set to 1.0 to avoid division by zero.
        if self.up == self.down:
            self.p = 1.0
        else:
            self.p = (
                np.exp((interest_rate - dividend_yield) * self.delta_time) - self.down
            ) / (self.up - self.down)

        if not (0.0 <= self.p <= 1.0):
            raise ValueError(f"Risk-neutral probability out of bounds: p={self.p}")

    def build_price_tree(self):
        """
        Build recombining upper-triangular price tree S where:
        S[i, j] = price after i steps with j up-moves (0 <= j <= i).

        Vectorized construction:
            - left edge (j=0): S[i,0] = S[i-1,0] * d
            - interior (j=1..i): S[i,j] = S[i-1, j-1] * u
        """
        n = self.steps
        S = np.zeros((n + 1, n + 1), dtype=float)
        S[0, 0] = self.current_price

        if self.up == 1.0 and self.down == 1.0:
            # Degenerate tree when dt=0: keep all nodes equal to S0
            # (Backward induction will collapse to intrinsic value at t=0.)
            for i in range(1, n + 1):
                S[i, : i + 1] = self.current_price
            return S

        for i in range(1, n + 1):
            # Lowest node at step i (all downs)
            S[i, 0] = S[i - 1, 0] * self.down
            # Vectorized fill of the remaining nodes (one up from previous row’s nodes)
            S[i, 1 : i + 1] = S[i - 1, 0:i] * self.up

        return S

    def backward_value(self, S, payoff_fn):
        """
        Vectorized backward induction.

        At maturity (row n): V[n, j] = payoff(S[n, j]).
        For i from n-1 down to 0:
            continuation = e^(-r*dt) * ( p * V[i+1, j+1] + (1-p) * V[i+1, j] )
            American: V = max(continuation, intrinsic)
            European: V = continuation

        Slices:
            - V[i+1, 1:i+2] are the j+1 "up" children
            - V[i+1, 0:i+1] are the j   "down" children
            - We compute all valid j (0..i) in one vectorized operation.
        """
        n = self.steps
        P = np.zeros_like(S)
        exercise = None

        # Terminal payoffs
        P[n, : n + 1] = payoff_fn(S[n, : n + 1])

        if self.is_american:
            # Marks where immediate exercise is (strictly) better than continuation
            exercise = np.zeros_like(S, dtype=bool)

        # Backward induction
        for i in range(n - 1, -1, -1):
            cont = self.disc * (
                self.p * P[i + 1, 1 : i + 2] + (1 - self.p) * P[i + 1, 0 : i + 1]
            )

            if self.is_american:
                intrinsic = payoff_fn(S[i, : i + 1])
                P[i, : i + 1] = np.maximum(cont, intrinsic)
                # Mark an exercise if the chosen value came from intrinsic.
                # Using '>' means ties are NOT marked as exercise; switch to '>=' if you prefer tie-as-exercise.
                exercise[i, : i + 1] = P[i, : i + 1] > cont
            else:
                P[i, : i + 1] = cont

        return P, exercise

    def run(self):
        """
        Builds S, prices both call and put via backward induction.
        Prices at t=0 are call_P[0,0] and put_P[0,0].
        """
        S = self.build_price_tree()

        # Payoff functions
        def call_payoff(x):
            return np.maximum(x - self.strike_price, 0.0)

        def put_payoff(x):
            return np.maximum(self.strike_price - x, 0.0)

        # Price trees (and exercise masks if American)
        self.call_P, self.call_exercise = self.backward_value(S, call_payoff)
        self.put_P, self.put_exercise = self.backward_value(S, put_payoff)
