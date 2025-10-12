# MultiModel Options

Interactive Streamlit app for pricing European and American options across multiple models and visualizing **P&L surfaces** over Spot × Volatility scenarios.

https://multimodel-options.streamlit.app/

## ✨ Features

- **Models**
  - **Black-Scholes-Merton** (lognormal)
  - **Binomial (CRR)** — European/American
  - **Bachelier (Normal)** — normal volatility
  - **Monte Carlo (GBM)** — European or American (LSM)
- **Interactive sidebar**
  - Model-specific parameters
  - Context-aware **Monte Carlo** inputs (European vs American)
  - **P&L inputs** (quantity, purchase prices)
  - **Range sliders** for Spot and Volatility scenario ranges
- **At-a-glance metrics**
  - CALL / PUT model prices
  - Current **Mark-to-Market P&L**
- **P&L Heatmaps (Spot × Volatility)**
  - Diverging colormap
  - Per-cell values (annotated)

## 🔧 Dependencies:

`streamlit`, `numpy`, `matplotlib`, `seaborn`, `scipy`
