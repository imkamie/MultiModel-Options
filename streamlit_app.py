import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from models_registry import make_models_dict
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

/* Custom class for P&L values */
.metric-pnl {
    background-color: #D7EDFA; /* Light blue background */
}

/* Style for the value text */
.metric-value {
    font-size: 1.6rem; /* Adjust font size */
    font-weight: bold;
}

/* Style for the label text */
.metric-label {
    font-size: 1rem; /* Adjust font size */
}

.contact-container {
    background-color: #FFFFFF;
    padding: 0px 6px 6px;
    border-radius: 8px;
}

.contact-container a {
    font-style: normal;
    text-decoration: none;
}

.contact-container .label {
    font-style: italic;
}

</style>
""",
    unsafe_allow_html=True,
)

st.title("MultiModel Options")

st.markdown(
    """
Here you can experiment with different option pricing models — Black-Scholes, Binomial Trees, Bachelier, and Monte Carlo — and see how option prices and P&L change under various market scenarios.

#### How to use this app:
- **Choose a model** in the sidebar. Each model has its own configurable parameters.
- **Parameters section** lets you set the option contract details (spot, strike, maturity, volatility, etc.).
- **P&L Inputs** allow you to specify your position size and purchase prices to calculate mark-to-market profit & loss.
- **Heatmap Settings** control the scenario ranges for Spot and Volatility, so you can visualize how P&L changes across different market conditions.

#### What you see on the page:
- **CALL and PUT Prices**: Theoretical values computed from the selected model.
- **Current P&L (Mark-to-Market)**: Shows your profit or loss if you bought at the given purchase price.
- **P&L Heatmaps (Spot \u00d7 Volatility)**: Scenario analysis grids.  
"""
)


MODELS = make_models_dict()

st.sidebar.title(":chart_with_upwards_trend: MultiModel Options")

# st.sidebar.write("Created by:")
linkedin_url = "https://www.linkedin.com/in/kamila-nurkhametova/"
st.sidebar.markdown(
    f"""
        <div class="contact-container">
            <div class="label">Created by:</div>
            <a href="{linkedin_url}" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="24" height="24" style="margin-right: 6px;">
                Kamila Nurkhametova
            </a>
        <div>
    """,
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


S_min, S_max = st.sidebar.slider(
    "Spot price range (S)", 0.0, 5_000.0, (max(0.0, 0.5 * S0), 1.5 * S0)
)

V_min, V_max = st.sidebar.slider(
    "Volatility range (σ)", 0.0, 5.0, (0.5 * sig0, 1.5 * sig0)
)


tau = st.sidebar.number_input(
    "Time to expiry for MTM (years)", value=T0, min_value=0.0, max_value=T0
)

# CURRENT P&L METRICS
pnl_now_call = qty * (float(call_price) - purchase_price_call)
pnl_now_put = qty * (float(put_price) - purchase_price_put)

st.markdown("## Current P&L (Mark-to-Market)")
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
st.markdown("## P&L Heatmaps (Spot \u00d7 Volatility)")

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
