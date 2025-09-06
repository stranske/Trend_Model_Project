"""Chart helpers for Trend Analysis results pages.

This module exposes small utilities that transform simulation outputs
into dataframes ready for visualisation.  The helpers are intentionally
lightweight so they can be reused outside of Streamlit and make it easy
to export the underlying data.
"""

from .charts import (drawdown_curve, equity_curve, rolling_information_ratio,
                     turnover_series, weights_heatmap, weights_heatmap_data)

__all__ = [
    "equity_curve",
    "drawdown_curve",
    "rolling_information_ratio",
    "turnover_series",
    "weights_heatmap",
    "weights_heatmap_data",
]
