"""Chart helpers for Trend Analysis results pages.

Keep imports lightweight so `trend_analysis.viz` can be imported without
pulling in optional UI dependencies.
"""

__all__ = [
    "equity_curve",
    "drawdown_curve",
    "rolling_information_ratio",
    "turnover_series",
    "weights_heatmap",
    "weights_heatmap_data",
]


def __getattr__(name: str):
    if name in __all__:
        from . import charts

        return getattr(charts, name)
    raise AttributeError(f"module 'trend_analysis.viz' has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
