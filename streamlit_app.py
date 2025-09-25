import streamlit as st

from black_scholes import BlackScholes
from binomial_tree import BinomialTree
from bachelier import Bachelier

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
}

/* Custom classes for CALL and PUT values */
.metric-call {
    background-color: #A8F1A3; /* Light green background */
    color: black; /* Black font color */
    border-radius: 12px; /* Rounded corners */
}

.metric-put {
    background-color: #FFBEBC; /* Light red background */
    color: black; /* Black font color */
    border-radius: 12px; /* Rounded corners */
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
            "time_to_maturity": ("Time to maturity (years)", 0.5, 0.0, 50.0, 0.01),
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
            "time_to_maturity": ("Time to maturity (years)", 0.5, 0.0, 50.0, 0.01),
            "current_price": ("Spot price S", 100.0, 0.0001, 1e7, 0.1),
            "strike_price": ("Strike K", 100.0, 0.0001, 1e7, 0.1),
            "interest_rate": ("Risk-free r (cont.)", 0.02, -1.0, 1.0, 0.01),
            "volatility": ("Normal vol σₙ (price units)", 5.0, 1e-9, 1e6, 0.01),
        },
        "run": lambda inst: (inst.run(), inst.call_price, inst.put_price),
        "show_prices": lambda inst: (inst.call_price, inst.put_price),
    },
}


st.sidebar.markdown("## Choose a Model")
model_name = st.sidebar.selectbox("Model", options=list(MODELS.keys()))
cfg = MODELS[model_name]

st.sidebar.markdown("## Parameters")
user_params = {}

# Render number inputs according to the schema
for key, (label, default, minv, maxv, step) in cfg["params"].items():
    if key == "is_american":
        choice = st.sidebar.selectbox(
            label, ["European", "American"], index=1
        )  # default = American (index 1) or 0 for European
        user_params[key] = choice == "American"
    elif key == "steps":
        val = st.sidebar.number_input(
            label,
            value=float(default),
            min_value=float(minv),
            max_value=float(maxv),
            step=float(step),
        )
        user_params[key] = int(val)
    else:
        user_params[key] = st.sidebar.number_input(
            label,
            value=float(default),
            min_value=float(minv),
            max_value=float(maxv),
            step=float(step),
        )


# Instantiate model with only the args it expects (by name)
ModelClass = cfg["constructor"]
model = ModelClass(**user_params)  # names in schema match constructor signatures
_, call_price, put_price = cfg["run"](model)


st.markdown("## Selected model: " + model_name)

# st.subheader("Parameters")
# st.json(user_params)

st.subheader("Results")
col1, col2 = st.columns(2)

with col1:
    # Using the custom class for CALL value
    st.markdown(
        f"""
        <div class="metric-container metric-call">
            <div class="metric-label">CALL price</div>
            <div class="metric-value">${call_price:.2f}</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    # Using the custom class for PUT value
    st.markdown(
        f"""
        <div class="metric-container metric-put">
            <div class="metric-label">PUT price</div>
            <div class="metric-value">${put_price:.2f}</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

# Tiny note for Bachelier users
if model_name.startswith("Bachelier"):
    st.caption(
        "Note: Bachelier uses normal volatility (σₙ) in price units, not percent."
    )
