import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from black_scholes import BlackScholes
from binomial_tree import BinomialTree
from bachelier import Bachelier
from monte_carlo_gbm import MonteCarloGBM
from helpers import (
    render_params_mc,
    render_params_generic,
    metric_box_html,
    draw_heatmap,
    build_pnl_surfaces,
)

st.set_page_config(
    page_title="Options Pricing",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

st.markdown(
    """
<style>
/* Adjust the size and alignment of the CALL and PUT value containers */
.metric-container {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    padding: 8px; /* Adjust the padding to control height */
    color: black; /* Black font color */
    border-radius: 12px; /* Rounded corners */
}

/* Custom classes for CALL and PUT values */
.metric-call {
    background-color: #A8F1A3; /* Light green background */
}

.metric-put {
    background-color: #FFBEBC; /* Light red background */
}

.metric-pnl {
    background-color: #D7EDFA; /* Light blue background */
}

/* Style for the value text */
.metric-value {
    font-size: 1.8rem; /* Adjust font size */
    font-weight: bold;
}

/* Style for the label text */
.metric-label {
    font-size: 1.2rem; /* Adjust font size */
}

</style>
""",
    unsafe_allow_html=True,
)


st.title("MultiModel Options")


# Each param is: (label, default, min, max, step)
MODELS = {
    "Black-Scholes-Merton": {
        "constructor": BlackScholes,
        "params": {
            "time_to_maturity": ("Time to maturity (years)", 0.5, 0.01, 50.0, 0.01),
            "current_price": ("Spot price S", 100.0, 0.0001, 1e7, 0.1),
            "strike_price": ("Strike K", 100.0, 0.0001, 1e7, 0.1),
            "interest_rate": ("Risk-free r (cont.)", 0.02, -1.0, 1.0, 0.01),
            "volatility": ("Volatility σ (lognormal)", 0.25, 1e-6, 5.0, 0.01),
        },
        "run": lambda inst: (inst.run(), inst.call_price, inst.put_price),
        "show_prices": lambda inst: (inst.call_price, inst.put_price),
    },
    "Binomial (CRR)": {
        "constructor": BinomialTree,
        "params": {
            "steps": ("Tree steps (integer)", 200, 1, 5000, 1),
            "time_to_maturity": ("Time to maturity (years)", 0.5, 0.0, 50.0, 0.01),
            "strike_price": ("Strike K", 100.0, 0.0001, 1e7, 0.1),
            "current_price": ("Spot price S", 100.0, 0.0001, 1e7, 0.1),
            "volatility": ("Volatility σ (lognormal)", 0.25, 1e-6, 5.0, 0.01),
            "interest_rate": ("Risk-free r (cont.)", 0.02, -1.0, 1.0, 0.01),
            "dividend_yield": ("Dividend yield q", 0.00, -1.0, 1.0, 0.01),
            "is_american": (
                "Exercise style",
                0,
                0,
                1,
                1,
            ),
        },
        "run": lambda inst: (inst.run(), inst.call_P[(0, 0)], inst.put_P[(0, 0)]),
        "show_prices": lambda inst: (inst.call_P[(0, 0)], inst.put_P[(0, 0)]),
    },
    "Bachelier (Normal)": {
        "constructor": Bachelier,
        "params": {
            "time_to_maturity": ("Time to maturity (years)", 0.5, 0.01, 50.0, 0.01),
            "current_price": ("Spot price S", 100.0, 0.0001, 1e7, 0.1),
            "strike_price": ("Strike K", 100.0, 0.0001, 1e7, 0.1),
            "interest_rate": ("Risk-free r (cont.)", 0.02, -1.0, 1.0, 0.01),
            "volatility": ("Normal vol σₙ (price units)", 5.0, 1e-9, 1e6, 0.01),
        },
        "run": lambda inst: (inst.run(), inst.call_price, inst.put_price),
        "show_prices": lambda inst: (inst.call_price, inst.put_price),
    },
    "Monte Carlo (GBM)": {
        "constructor": MonteCarloGBM,
        "params": {
            "time_to_maturity": ("Time to maturity (years)", 0.5, 0.01, 50.0, 0.01),
            "current_price": ("Spot price S", 100.0, 0.0001, 1e7, 0.1),
            "strike_price": ("Strike K", 100.0, 0.0001, 1e7, 0.1),
            "interest_rate": ("Risk-free r (cont.)", 0.02, -1.0, 1.0, 0.01),
            "volatility": ("Volatility σ (lognormal)", 0.25, 1e-6, 5.0, 0.01),
            "dividend_yield": ("Dividend yield q", 0.00, -1.0, 1.0, 0.01),
            "is_american": (
                "Exercise style",
                0,
                0,
                1,
                1,
            ),
            "n_paths": ("MC paths (integer)", 20000, 1000, 2_000_000, 1000),
            "steps": ("LSM steps (American only)", 50, 1, 2000, 1),
            "antithetic": ("Antithetic variates", 1, 0, 1, 1),  # 0/1 for your UI loop
            "control_variate": ("Control variate (European only)", 1, 0, 1, 1),
            "seed": ("Random seed (integer)", 42, 0, 2**31 - 1, 1),
        },
        "run": lambda inst: (inst.run(), inst.call_price, inst.put_price),
        "show_prices": lambda inst: (inst.call_price, inst.put_price),
    },
}

st.sidebar.title(":chart_with_upwards_trend: MultiModel Options")

st.sidebar.write("`Created by:`")
linkedin_url = "https://www.linkedin.com/in/kamila-nurkhametova/"
st.sidebar.markdown(
    f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="24" height="24" style="vertical-align: middle; margin-right: 10px;">`Kamila Nurkhametova`</a>',
    unsafe_allow_html=True,
)


st.sidebar.markdown("## Choose a Model")
model_name = st.sidebar.selectbox("Model", options=list(MODELS.keys()))
cfg = MODELS[model_name]

st.sidebar.markdown("## Parameters")
user_params = {}

# Render number inputs according to the schema
if model_name.startswith("Monte Carlo"):
    user_params = render_params_mc()
else:
    user_params = render_params_generic(cfg)

# Instantiate model with only the args it expects (by name)
ModelClass = cfg["constructor"]
model = ModelClass(**user_params)  # names in schema match constructor signatures
_, call_price, put_price = cfg["run"](model)


st.markdown("## " + model_name)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        metric_box_html("CALL price", call_price, "metric-call"), unsafe_allow_html=True
    )

with col2:
    st.markdown(
        metric_box_html("PUT price", put_price, "metric-put"), unsafe_allow_html=True
    )

# Tiny note for Bachelier users
if model_name.startswith("Bachelier"):
    st.caption(
        "Note: Bachelier uses normal volatility (σₙ) in price units, not percent."
    )

st.sidebar.markdown("---")
st.sidebar.markdown("## P&L Inputs")
qty = st.sidebar.number_input("Quantity (long + / short -)", value=1.0, step=1.0)

purchase_price_call = st.sidebar.number_input(
    "Purchase price (Call)", value=float(call_price), min_value=0.0, step=0.01
)
purchase_price_put = st.sidebar.number_input(
    "Purchase price (Put)", value=float(put_price), min_value=0.0, step=0.01
)

st.sidebar.markdown("---")
st.sidebar.markdown("## Heatmap Settings (Spot \u00d7 Vol)")
S0 = float(user_params.get("current_price", 100.0))
sig0 = float(user_params.get("volatility", 0.25))
T0 = float(user_params.get("time_to_maturity", 0.5))

S_min = st.sidebar.number_input("Spot min", value=max(0.0, 0.5 * S0))
S_max = st.sidebar.number_input("Spot max", value=1.5 * S0)

V_min = st.sidebar.number_input(
    "Vol min", value=0.5 * sig0, help="For Bachelier this is normal vol (price units)"
)
V_max = st.sidebar.number_input("Vol max", value=1.5 * sig0)

tau = st.sidebar.number_input(
    "Time to expiry for MTM (years)", value=T0, min_value=0.0, max_value=T0
)

# CURRENT P&L METRICS
pnl_now_call = qty * (float(call_price) - purchase_price_call)
pnl_now_put = qty * (float(put_price) - purchase_price_put)

st.subheader("Current P&L (Mark-to-Market)")
pc, pp = st.columns(2)
with pc:
    st.markdown(
        metric_box_html("CALL P&L", pnl_now_call, "metric-pnl"), unsafe_allow_html=True
    )

with pp:
    st.markdown(
        metric_box_html("PUT P&L", pnl_now_put, "metric-pnl"), unsafe_allow_html=True
    )


# BUILD HEATMAPS
S_grid = np.linspace(S_min, S_max, 10)
V_grid = np.linspace(V_min, V_max, 10)

PNL_CALL, PNL_PUT = build_pnl_surfaces(
    model_name,
    user_params,
    S_grid,
    V_grid,
    tau,
    qty,
    purchase_price_call,
    purchase_price_put,
)


# PLOT HEATMAPS
st.subheader("P&L Heatmaps (Spot \u00d7 Volatility)")

cmap = mcolors.LinearSegmentedColormap.from_list(
    "red_white_green", ["red", "white", "green"]
)

h1, h2 = st.columns(2)
with h1:
    fig_c, ax_c = plt.subplots(figsize=(10, 8))
    draw_heatmap(ax_c, PNL_CALL, S_grid, V_grid, "CALL P&L Heatmap", cmap)
    st.pyplot(fig_c)

with h2:
    fig_p, ax_p = plt.subplots(figsize=(10, 8))
    draw_heatmap(ax_p, PNL_PUT, S_grid, V_grid, "PUT P&L Heatmap", cmap)
    st.pyplot(fig_p)

st.caption(
    "P&L = quantity \u00d7 (new model price - purchase price). Positive values indicate profit for a long position."
)
