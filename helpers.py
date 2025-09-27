import streamlit as st


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
