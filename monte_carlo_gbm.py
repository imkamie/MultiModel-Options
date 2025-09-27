import numpy as np


class MonteCarloGBM:
    """
    European / American option pricing under risk-neutral GBM with optional dividend yield.

    Model (risk-neutral):
        dS_t = (r - q) S_t dt + sigma S_t dW_t
        S_{t+delta} = S_t * exp((r - q - 0.5*sigma^2) delta + sigma * sqrt(delta) * Z),  Z ~ N(0,1)

    Assumptions:
        - time_to_maturity > 0, volatility > 0
        - current_price > 0, strike_price > 0
        - interest_rate, dividend_yield are continuously compounded

    Features:
        - is_american: if True, price via Longstaff-Schwartz regression (basis [1, S, S^2])
        - antithetic: antithetic variates (Z, -Z) for variance reduction (paired by path)
        - control_variate: optional CV with X=S_T (European only), using payoffs (discount after)

    Outputs (after run()):
        - call_price, put_price
    """

    def __init__(
        self,
        time_to_maturity: float,
        current_price: float,
        strike_price: float,
        interest_rate: float,
        volatility: float,
        dividend_yield: float,
        is_american: bool,
        n_paths: int,
        steps: int,
        antithetic: bool,
        control_variate: bool,
        seed: int,
    ):
        self.time_to_maturity = time_to_maturity
        self.current_price = current_price
        self.strike_price = strike_price
        self.interest_rate = interest_rate
        self.volatility = volatility
        self.dividend_yield = dividend_yield
        self.is_american = is_american
        self.n_paths = n_paths
        self.steps = steps
        self.antithetic = antithetic
        self.control_variate = control_variate
        self.seed = seed

        self.rng = np.random.default_rng(seed)

    def _draw_Z(self, size):
        """
        Draw standard normals with optional antithetics.
        - If size is (n_steps, n_paths), we pair columns (path-wise antithetics).
        - If size is an int, we pair within the vector.
        """

        if not self.antithetic:
            return self.rng.standard_normal(size)

        # Build antithetic pairs along last axis
        if isinstance(size, tuple):
            n_steps, n_paths = size
            half_cols = (n_paths + 1) // 2
            Z_half = self.rng.standard_normal((n_steps, half_cols))
            # Concatenate antithetic partners along path axis, then trim
            Z = np.concatenate([Z_half, -Z_half], axis=1)[:, :n_paths]
            return Z
        else:
            half = (size + 1) // 2
            Z_half = self.rng.standard_normal(half)
            Z = np.concatenate([Z_half, -Z_half])[:size]
            return Z

    def _simulate_terminal(self, n_paths):
        """
        Single-step simulation of S_T for European pricing:
        S_T = S0 * exp((r - q - 0.5*sigma^2)T + sigma*sqrt(T)*Z).
        """
        T = self.time_to_maturity
        S0 = self.current_price
        r = self.interest_rate
        q = self.dividend_yield
        sigma = self.volatility

        Z = self._draw_Z(n_paths)
        drift = (r - q - 0.5 * sigma**2) * T
        volT = sigma * np.sqrt(T)
        ST = S0 * np.exp(drift + volT * Z)
        return ST

    def _simulate_paths_matrix(self, n_paths, n_steps):
        """
        Full path matrix for LSM, shape (n_steps+1, n_paths).
        Row 0 = S0, row n_steps = maturity. Exact GBM step per dt.
        """
        T = self.time_to_maturity
        S0 = self.current_price
        r = self.interest_rate
        q = self.dividend_yield
        sigma = self.volatility

        dt = T / n_steps
        nudt = (r - q - 0.5 * sigma**2) * dt
        sigsdt = sigma * np.sqrt(dt)

        Z = self._draw_Z((n_steps, n_paths))
        S = np.empty((n_steps + 1, n_paths), dtype=float)
        S[0, :] = S0
        for t in range(1, n_steps + 1):
            S[t, :] = S[t - 1, :] * np.exp(nudt + sigsdt * Z[t - 1, :])
        return S

    # ---------- LSM core ----------

    def _lsm_price(self, S, payoff_fn, disc_step):
        """
        Longstaff-Schwartz for a single payoff (call OR put).
        Basis: [1, S, S^2]. Regress only on in-the-money paths at each time.
        CF[t, :] holds the value at time t along each path after decisions for times > t.
        """
        n_steps = S.shape[0] - 1
        intrinsic = payoff_fn(S)  # (n_steps+1, n_paths)
        CF = (
            intrinsic.copy()
        )  # initialize with terminal payoffs; will be overwritten backward

        # Backward induction: t = n_steps-1, ..., 1  (t=0 handled separately)
        for t in range(n_steps - 1, 0, -1):
            itm = intrinsic[t, :] > 0.0  # ITM paths only for regression
            if np.any(itm):
                X = S[t, itm]
                # One-step discount of next-step value (already optimal from later times)
                Y = disc_step * CF[t + 1, itm]

                # Regress Y on [1, S, S^2] to estimate continuation value at time t
                A = np.vstack([np.ones_like(X), X, X * X]).T
                beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
                continuation = beta[0] + beta[1] * S[t, :] + beta[2] * (S[t, :] ** 2)
            else:
                # No ITM paths: continuation is just "carry forward" discounted value
                continuation = disc_step * CF[t + 1, :]

            # Exercise decision: exercise if intrinsic strictly better than continuation
            exercise = (
                intrinsic[t, :] > continuation
            )  # use >= if you prefer tie-break to exercise
            CF[t, exercise] = intrinsic[t, exercise]
            CF[t, ~exercise] = disc_step * CF[t + 1, ~exercise]
            CF[t + 1 :, exercise] = 0.0  # stop future CF on exercised paths

        # Time 0: choose between immediate intrinsic (same across paths) and discounted next-step expectation
        intrinsic0 = intrinsic[0, 0]
        cont0 = disc_step * np.mean(CF[1, :])
        return max(intrinsic0, cont0)

    def run(self):
        T = self.time_to_maturity
        K = self.strike_price
        r = self.interest_rate

        if not self.is_american:
            # European pricing: simulate S_T and discount expected payoff
            ST = self._simulate_terminal(self.n_paths)
            disc_T = np.exp(-r * T)

            call_payoff = np.maximum(ST - K, 0.0)
            put_payoff = np.maximum(K - ST, 0.0)

            if self.control_variate:
                # Control variate: X = S_T with E[X] = S0 * e^((r - q)T)
                S0 = self.current_price
                q = self.dividend_yield
                EX = S0 * np.exp((r - q) * T)
                Xc = ST - EX  # zero-mean control
                eps = 1e-16

                def apply_cv(vals):
                    varX = np.var(Xc, ddof=0)
                    if varX < eps:
                        return vals
                    # Use population covariance to match var(ddof=0)
                    cov = np.cov(vals, Xc, bias=True)[0, 1]
                    beta = cov / varX
                    return vals - beta * Xc

                call_vals = disc_T * apply_cv(call_payoff)
                put_vals = disc_T * apply_cv(put_payoff)
            else:
                call_vals = disc_T * call_payoff
                put_vals = disc_T * put_payoff

            self.call_price = float(call_vals.mean())
            self.put_price = float(put_vals.mean())
            return

        # American pricing via Longstaff–Schwartz
        n_steps = self.steps
        S = self._simulate_paths_matrix(self.n_paths, n_steps)
        dt = T / n_steps
        disc_step = np.exp(-r * dt)

        def call_payoff(x):
            return np.maximum(x - K, 0.0)

        def put_payoff(x):
            return np.maximum(K - x, 0.0)

        self.call_price = float(self._lsm_price(S, call_payoff, disc_step))
        self.put_price = float(self._lsm_price(S, put_payoff, disc_step))
