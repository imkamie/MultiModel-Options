import streamlit as st
import numpy as np
import seaborn as sns

from black_scholes import BlackScholes
from binomial_tree import BinomialTree
from bachelier import Bachelier
from monte_carlo_gbm import MonteCarloGBM


def _num(label, value, minv, maxv, step, *, cast=float):
    v = st.sidebar.number_input(
        label,
        value=float(value),
        min_value=float(minv),
        max_value=float(maxv),
        step=float(step),
    )
    return cast(v)


def _toggle(label, default_yes=True):
    return (
        st.sidebar.selectbox(label, ["No", "Yes"], index=1 if default_yes else 0)
        == "Yes"
    )


def _exercise(label="Exercise style", default="European"):
    return (
        st.sidebar.selectbox(
            label, ["European", "American"], index=0 if default == "European" else 1
        )
        == "American"
    )


def render_params_mc():
    p = {}
    # common finance inputs
    p["time_to_maturity"] = _num("Time to maturity (years)", 0.5, 1e-6, 50.0, 0.01)
    p["current_price"] = _num("Spot price S", 100.0, 0.0001, 1e7, 0.1)
    p["strike_price"] = _num("Strike K", 100.0, 0.0001, 1e7, 0.1)
    p["interest_rate"] = _num("Risk-free r (cont.)", 0.02, -1.0, 1.0, 0.01)
    p["volatility"] = _num("Volatility σ (lognormal)", 0.25, 1e-6, 5.0, 0.01)
    p["dividend_yield"] = _num("Dividend yield q", 0.00, -1.0, 1.0, 0.01)

    # exercise + MC common
    p["is_american"] = _exercise(default="European")
    p["n_paths"] = _num("MC paths (integer)", 20000, 1000, 2_000_000, 1000, cast=int)
    p["antithetic"] = _toggle("Antithetic variates", default_yes=True)
    p["seed"] = _num("Random seed (integer)", 42, 0, 2**31 - 1, 1, cast=int)

    # exercise-specific
    if p["is_american"]:
        p["steps"] = _num("LSM steps (American only)", 50, 1, 2000, 1, cast=int)
        p["control_variate"] = False  # ignored for American
        st.sidebar.caption(
            "Longstaff–Schwartz uses 'LSM steps'. Control variate is European-only."
        )
    else:
        p["control_variate"] = _toggle(
            "Control variate (European only)", default_yes=True
        )
        p["steps"] = 1  # harmless placeholder

    return p


def render_params_generic(cfg):
    p = {}
    for key, (label, default, minv, maxv, step) in cfg["params"].items():
        if key == "is_american":
            p[key] = _exercise(label, default="American")
        elif key in {"steps"}:
            p[key] = _num(label, default, minv, maxv, step, cast=int)
        else:
            p[key] = _num(label, default, minv, maxv, step)
    return p


def metric_box_html(label: str, value: float, css_class: str) -> str:
    return f"""
        <div class="metric-container {css_class}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">${value:.2f}</div>
        </div>
    """


def draw_heatmap(
    ax, data, S_grid, V_grid, title: str, cmap, center: float = 0.0, annot: bool = True
):
    sns.heatmap(
        data,
        xticklabels=np.round(S_grid, 2),
        yticklabels=np.round(V_grid, 2),
        annot=annot,
        fmt=".2f",
        cmap=cmap,
        center=center,
        ax=ax,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Spot S (scenario)")
    ax.set_ylabel("Volatility (scenario)")
    ax.set_title(title)


def _price_with_current_model(
    model_name: str, base_params: dict, S: float, sigma: float, tau: float, which: str
) -> float:
    p = dict(base_params)
    p["current_price"] = S
    p["time_to_maturity"] = tau

    if model_name.startswith("Bachelier"):
        p["volatility"] = sigma  # normal vol (price units)
        m = Bachelier(
            time_to_maturity=p["time_to_maturity"],
            current_price=p["current_price"],
            strike_price=p["strike_price"],
            interest_rate=p["interest_rate"],
            volatility=p["volatility"],
        )
        m.run()
        return m.call_price if which == "Call" else m.put_price

    elif model_name.startswith("Black-Scholes"):
        p["volatility"] = sigma
        m = BlackScholes(
            time_to_maturity=p["time_to_maturity"],
            current_price=p["current_price"],
            strike_price=p["strike_price"],
            interest_rate=p["interest_rate"],
            volatility=p["volatility"],
        )
        m.run()
        return m.call_price if which == "Call" else m.put_price

    elif model_name.startswith("Monte Carlo"):
        p["volatility"] = sigma
        m = MonteCarloGBM(
            time_to_maturity=p["time_to_maturity"],
            current_price=p["current_price"],
            strike_price=p["strike_price"],
            interest_rate=p["interest_rate"],
            volatility=p["volatility"],
            dividend_yield=p.get("dividend_yield", 0.0),
            is_american=bool(p.get("is_american", False)),
            n_paths=int(p.get("n_paths", 20000)),
            steps=int(p.get("steps", 50)),
            antithetic=bool(p.get("antithetic", True)),
            control_variate=bool(p.get("control_variate", True)),
            seed=int(p.get("seed", 42)),
        )
        m.run()
        return m.call_price if which == "Call" else m.put_price

    else:  # Binomial (CRR)
        p["volatility"] = sigma
        m = BinomialTree(
            steps=int(p["steps"]),
            time_to_maturity=p["time_to_maturity"],
            strike_price=p["strike_price"],
            current_price=p["current_price"],
            volatility=p["volatility"],
            interest_rate=p["interest_rate"],
            dividend_yield=p["dividend_yield"],
            is_american=bool(p["is_american"]),
        )
        m.run()
        return m.call_P[(0, 0)] if which == "Call" else m.put_P[(0, 0)]


def build_pnl_surfaces(
    model_name: str,
    base_params: dict,
    S_grid,
    V_grid,
    tau: float,
    qty: float,
    price_paid_call: float,
    price_paid_put: float,
):
    nV, nS = len(V_grid), len(S_grid)
    PNL_CALL = np.zeros((nV, nS))
    PNL_PUT = np.zeros((nV, nS))

    for i, sigma in enumerate(V_grid):
        for j, S in enumerate(S_grid):
            mtm_call = _price_with_current_model(
                model_name, base_params, S, sigma, tau, "Call"
            )
            mtm_put = _price_with_current_model(
                model_name, base_params, S, sigma, tau, "Put"
            )
            PNL_CALL[i, j] = qty * (mtm_call - price_paid_call)
            PNL_PUT[i, j] = qty * (mtm_put - price_paid_put)

    return PNL_CALL, PNL_PUT
