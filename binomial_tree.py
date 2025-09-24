from numpy import exp, sqrt


class BinomialTree:
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

        self.delta_time = time_to_maturity / steps

        self.up, self.down = self.up_down(volatility, self.delta_time)

        self.probability = self.risk_neutral_probability(
            interest_rate, dividend_yield, self.delta_time, self.up, self.down
        )

    def up_down(self, volatility, delta_time):
        up = exp(volatility * sqrt(delta_time))
        down = exp(-(volatility * sqrt(delta_time)))

        return up, down

    def risk_neutral_probability(
        self, interest_rate, dividend_yield, delta_time, up, down
    ):
        p = (exp((interest_rate - dividend_yield) * delta_time) - down) / (up - down)

        return p

    def calc_all_nodes_for_puts(
        self,
        steps,
        current_price,
        up,
        down,
        strike_price,
        probability,
        interest_rate,
        delta_time,
    ):
        # Price
        S = {}

        # S[i, j] is defined as S[time,(step - amount of ups)]
        S[0, 0] = current_price

        for i in range(1, steps + 1):
            for j in range(0, i + 1):
                if j > 0:
                    S[i, j] = S[i - 1, j - 1] * up
                else:
                    S[i, 0] = S[i - 1, 0] * down

        # P-alive vs P-exercise
        P = {}
        exercise = {}

        for i in range(steps, -1, -1):
            for j in range(0, i + 1):
                if i == steps:
                    P[i, j] = max(strike_price - S[i, j], 0)
                else:
                    P[i, j] = (
                        P[i + 1, j + 1] * probability + P[i + 1, j] * (1 - probability)
                    ) * exp(-(interest_rate * delta_time))

                P[i, j] = max(P[i, j], 0)

                if P[i, j] < (strike_price - S[i, j]):
                    exercise[i, j] = True
                else:
                    exercise[i, j] = False

                if self.is_american:
                    P[i, j] = max(P[i, j], (strike_price - S[i, j]))

        return S, P, exercise

    def calc_all_nodes_for_calls(
        self,
        steps,
        current_price,
        up,
        down,
        strike_price,
        probability,
        interest_rate,
        delta_time,
    ):
        # Price
        S = {}

        # S[i, j] is defined as S[time,(step - amount of ups)]
        S[0, 0] = current_price

        for i in range(1, steps + 1):
            for j in range(0, i + 1):
                if j > 0:
                    S[i, j] = S[i - 1, j - 1] * up
                else:
                    S[i, 0] = S[i - 1, 0] * down

        # P-alive vs P-exercise
        P = {}
        exercise = {}

        for i in range(steps, -1, -1):
            for j in range(0, i + 1):
                if i == steps:
                    P[i, j] = max(S[i, j] - strike_price, 0)
                else:
                    P[i, j] = (
                        P[i + 1, j + 1] * probability + P[i + 1, j] * (1 - probability)
                    ) * exp(-(interest_rate * delta_time))

                P[i, j] = max(P[i, j], 0)

                if P[i, j] < (S[i, j] - strike_price):
                    exercise[i, j] = True
                else:
                    exercise[i, j] = False

                if self.is_american:
                    P[i, j] = max(P[i, j], (S[i, j] - strike_price))

        return S, P, exercise

    def run(self):
        self.put_s, self.put_P, self.put_exercise = self.calc_all_nodes_for_puts(
            self.steps,
            self.current_price,
            self.up,
            self.down,
            self.strike_price,
            self.probability,
            self.interest_rate,
            self.delta_time,
        )

        self.call_s, self.call_P, self.call_exercise = self.calc_all_nodes_for_calls(
            self.steps,
            self.current_price,
            self.up,
            self.down,
            self.strike_price,
            self.probability,
            self.interest_rate,
            self.delta_time,
        )
