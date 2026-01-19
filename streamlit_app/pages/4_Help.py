"""Help and documentation page for the Streamlit application."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Help - Portfolio Simulator", page_icon="📖", layout="wide")


def render_help_page() -> None:
    st.title("📖 Configuration Reference")
    st.markdown("""
This guide explains all configuration parameters available in the Model page.
Use the sections below to understand how each setting affects your analysis.
        """)

    # Table of contents
    st.markdown("---")
    st.markdown("""
**Jump to section:**
- [Fund Selection Settings](#fund-selection-settings)
- [Portfolio Settings](#portfolio-settings)
- [Metric Weights](#metric-weights)
- [Risk Settings](#risk-settings)
- [Weighting Schemes](#weighting-schemes)
- [Tips for Common Scenarios](#tips-for-common-scenarios)
        """)

    # Fund Selection Settings (formerly Trend Signal)
    st.markdown("---")
    st.header("Fund Selection Settings")
    st.markdown("""
These settings control how funds are evaluated and filtered for inclusion in the portfolio.
        """)

    st.subheader("Preset")
    st.markdown("""
**What it does:** Selects a pre-configured set of parameters optimized for different investment styles.

| Preset | Description |
|--------|-------------|
| **Baseline** | Default settings suitable for most analyses. Balanced approach. |
| **Conservative** | Longer lookback periods, more data required. Reduces noise but may lag market turns. |
| **Aggressive** | Shorter lookbacks, faster response. More responsive but potentially noisier. |
| **Custom** | Manually configure all parameters below. |
        """)

    st.subheader("Lookback Window (months)")
    st.markdown("""
**What it does:** The number of months of historical returns used to calculate fund performance metrics
(Sharpe ratio, returns, drawdowns) for ranking and selection.

**Default:** 36 months (3 years)

**Trade-offs:**
- **Longer windows (48–60 months):** More statistical significance, but may include stale data from different market regimes
- **Shorter windows (12–24 months):** More relevant to current market conditions, but higher estimation error

**Example:** A 36-month window bases fund metrics on the past 3 years of performance.
        """)

    st.subheader("Minimum History Required")
    st.markdown("""
**What it does:** The minimum number of months of return data a fund must have to be considered
for portfolio inclusion. Funds with less history are excluded from selection.

**Default:** Same as Lookback Window

**Trade-offs:**
- **Set equal to Lookback Window:** Strictest; only funds with complete history are considered
- **Set lower (e.g., Lookback/2):** Allow newer funds with partial history to be considered

**Example:** With Lookback=36 and Min History=18, a fund with only 20 months of data
will still be evaluated for inclusion.
        """)

    # Portfolio Settings
    st.markdown("---")
    st.header("Portfolio Settings")
    st.markdown("""
These settings control how the portfolio is constructed from the selected funds.
        """)

    st.subheader("Evaluation Window (months)")
    st.markdown("""
**What it does:** Defines the out-of-sample period over which portfolio performance is measured
after fund selection. This is the "test" period.

**Default:** 12 months

**How it works:** Funds are selected based on their in-sample metrics (Lookback period),
then held for the evaluation period to measure realized performance.

**Minimum:** 3 months (shorter periods have high variance)
        """)

    st.subheader("Selection Count")
    st.markdown("""
**What it does:** The number of top-ranked funds to include in the portfolio.

**Default:** 10 funds

**Trade-offs:**
- **Fewer funds (3–5):** Concentrated portfolio, higher tracking to best performers, more sensitive to individual fund outcomes
- **More funds (15–20):** Diversified portfolio, lower individual fund impact, more index-like behavior

**Constraint:** Cannot exceed the number of available funds in your dataset.
        """)

    st.subheader("Weighting Scheme")
    st.markdown("""
**What it does:** Determines how capital is allocated across the selected funds.

See the [Weighting Schemes](#weighting-schemes) section below for detailed descriptions of all available options.
        """)

    # Metric Weights
    st.markdown("---")
    st.header("Metric Weights")
    st.markdown("""
These weights control the relative importance of each performance metric when ranking funds for selection.
The final ranking score is a weighted combination of z-scored metrics.

**How it works:**
1. Each metric is calculated for all funds over the lookback period
2. Metrics are z-scored (normalized) across all funds
3. Z-scores are combined using these weights
4. Funds are ranked by their combined score
5. Top N funds (Selection Count) are included in the portfolio
        """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Available Metrics:**

| Metric | Description | Higher is Better |
|--------|-------------|------------------|
| **Sharpe Ratio** | Risk-adjusted return (return / volatility) | ✓ |
| **Annual Return** | Annualized total return | ✓ |
| **Sortino Ratio** | Return / downside volatility | ✓ |
| **Info Ratio** | Excess return over benchmark / tracking error | ✓ |
| **Max Drawdown** | Largest peak-to-trough decline | ✗ (penalized) |
| **Volatility** | Annualized standard deviation | ✗ (penalized) |
            """)

    with col2:
        st.markdown("""
**Example Weight Configurations:**

**Risk-Focused (Conservative):**
- Sharpe: 2.0, Return: 0.5, Drawdown: 2.0, Sortino: 1.0

**Return-Focused (Aggressive):**
- Sharpe: 1.0, Return: 2.0, Drawdown: 0.5

**Balanced:**
- All weights equal (1.0 each)

Weights are normalized internally, so only relative values matter.
            """)

    # Risk Settings
    st.markdown("---")
    st.header("Risk Settings")

    st.subheader("Target Volatility (Portfolio)")
    st.markdown("""
**What it does:** The target annualized volatility for the overall portfolio.
The system scales portfolio weights to achieve approximately this volatility level.

**Default:** 0.10 (10% annualized)

**How it works:** If the unscaled portfolio has 15% volatility and the target is 10%,
weights are reduced by 1/3 (with the remainder allocated to cash or not invested).

**Trade-offs:**
- **Lower targets (5–8%):** Conservative, more cash buffer, smaller drawdowns
- **Higher targets (12–20%):** Aggressive, fully invested, larger potential gains and losses
        """)

    # Weighting Schemes
    st.markdown("---")
    st.header("Weighting Schemes")
    st.markdown("""
The weighting scheme determines how capital is distributed across selected funds.
All schemes normalize final weights to sum to 100%.
        """)

    st.markdown("""
| Scheme | Description | Best For |
|--------|-------------|----------|
| **equal** | Each fund receives equal weight (1/N). Simple, robust, transparent. | Most users; no assumptions about fund characteristics |
| **risk_parity** | Weights inversely proportional to volatility. Higher-vol funds get lower weights. | Balancing risk across assets with different volatilities |
| **hrp** | Hierarchical Risk Parity. Uses clustering to build a diversified allocation. | Complex portfolios with correlated assets |
| **erc** | Equal Risk Contribution. Optimizes for equal marginal risk from each fund. | Formal risk targeting |
| **robust_mv** | Robust Mean-Variance. Shrinkage-based optimization resistant to estimation error. | When you trust return forecasts but want stability |
| **robust_risk_parity** | Robust Risk Parity with shrinkage. Combines risk parity with robust estimation. | Large portfolios with estimation uncertainty |

**Fallback behavior:** If an advanced weighting scheme fails (e.g., due to insufficient data or
numerical issues), the system automatically falls back to equal weighting and logs a warning.
        """)

    # Tips
    st.markdown("---")
    st.header("Tips for Common Scenarios")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
**🛡️ Conservative Long-Term**
- Lookback: 48–60 months
- Selection: 15–20 funds
- Weights: Emphasize Sharpe, Drawdown
- Target Vol: 6–8%
- Scheme: risk_parity or hrp
            """)

    with col2:
        st.markdown("""
**⚡ Aggressive Momentum**
- Lookback: 12–24 months
- Selection: 5–8 funds
- Weights: Emphasize Return
- Target Vol: 15–20%
- Scheme: equal
            """)

    with col3:
        st.markdown("""
**⚖️ Balanced Portfolio**
- Lookback: 36 months
- Selection: 10–12 funds
- Weights: Equal (1.0 each)
- Target Vol: 10%
- Scheme: equal or risk_parity
            """)

    st.markdown("---")
    st.caption(
        "For more details, see the [full documentation](https://github.com/stranske/Trend_Model_Project/blob/main/docs/UserGuide.md)."
    )


render_help_page()
