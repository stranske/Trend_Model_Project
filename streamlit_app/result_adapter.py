"""Adapters to align pipeline results with the Streamlit UI expectations."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping

import pandas as pd


class RunResultAdapter:
    """Wrapper that adds Streamlit-friendly helpers to ``RunResult`` objects."""

    def __init__(self, run_result: Any) -> None:
        self._raw = run_result
        self.metrics = getattr(run_result, "metrics", pd.DataFrame())
        self.details = getattr(run_result, "details", {})
        self.seed = getattr(run_result, "seed", None)
        self.environment = getattr(run_result, "environment", {})
        self.fallback_info = getattr(run_result, "fallback_info", None)
        self._portfolio_returns = self._compute_portfolio_returns()
        # ``export_bundle`` expects an attribute named ``portfolio``.
        self.portfolio = self._portfolio_returns.copy()
        self._weights_by_date = self._build_weights_by_date()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _out_frame(self) -> pd.DataFrame | None:
        raw = self.details.get("out_sample_scaled")
        if isinstance(raw, pd.DataFrame):
            return raw.copy()
        if raw is None:
            return None
        try:
            return pd.DataFrame(raw)
        except Exception:
            return None

    def _weight_series(self) -> pd.Series | None:
        raw = self.details.get("fund_weights")
        if isinstance(raw, pd.Series):
            series = raw.astype(float)
        elif isinstance(raw, Mapping):
            try:
                series = pd.Series(raw, dtype=float)
            except Exception:
                return None
        else:
            return None
        if series.empty:
            return None
        return series

    def _compute_portfolio_returns(self) -> pd.Series:
        frame = self._out_frame()
        weights = self._weight_series()
        if frame is None or weights is None or frame.empty:
            return pd.Series(dtype=float)
        aligned = frame.reindex(columns=weights.index).fillna(0.0)
        returns = aligned.mul(weights, axis=1).sum(axis=1)
        try:
            idx = pd.to_datetime(aligned.index)
        except Exception:
            idx = pd.Index(aligned.index)
        returns = pd.Series(returns.astype(float).values, index=idx, dtype=float)
        return returns

    def _build_weights_by_date(self) -> dict[pd.Timestamp, pd.Series]:
        frame = self._out_frame()
        weights = self._weight_series()
        if frame is None or weights is None or frame.empty:
            return {}
        mapping: OrderedDict[pd.Timestamp, pd.Series] = OrderedDict()
        for label in frame.index:
            try:
                ts = pd.Timestamp(label)
            except Exception:
                continue
            mapping[ts] = weights.copy()
        return mapping

    # ------------------------------------------------------------------
    # Public helpers consumed by the Streamlit UI
    # ------------------------------------------------------------------
    def portfolio_curve(self) -> pd.Series:
        returns = self._portfolio_returns
        if returns.empty:
            return pd.Series(dtype=float)
        curve = (1.0 + returns).cumprod()
        return curve.astype(float)

    def drawdown_curve(self) -> pd.Series:
        curve = self.portfolio_curve()
        if curve.empty:
            return pd.Series(dtype=float)
        peak = curve.cummax()
        dd = curve / peak - 1.0
        return dd.astype(float)

    def event_log_df(self) -> pd.DataFrame:
        weights = self._weight_series()
        if weights is None or weights.empty:
            return pd.DataFrame(columns=["Fund", "Weight"])
        df = pd.DataFrame({"Fund": weights.index, "Weight": weights.values})
        df["Weight"] = df["Weight"].astype(float)
        return df

    def summary(self) -> dict[str, float]:
        curve = self.portfolio_curve()
        if curve.empty:
            return {}
        total = float(curve.iloc[-1] - 1.0)
        drawdown = float((curve / curve.cummax() - 1.0).min())
        periods = max(len(curve), 1)
        try:
            annualised = float(curve.iloc[-1] ** (12.0 / periods) - 1.0)
        except Exception:
            annualised = total
        return {
            "total_return": total,
            "max_drawdown": drawdown,
            "ann_return_approx": annualised,
        }

    @property
    def weights(self) -> dict[pd.Timestamp, pd.Series]:
        return dict(self._weights_by_date)

    @property
    def raw_result(self) -> Any:
        return self._raw

    # ------------------------------------------------------------------
    # Fallback attribute access
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


def adapt_run_result(result: Any) -> Any:
    """Return an object that provides the interface expected by the UI."""

    if hasattr(result, "portfolio_curve") and callable(result.portfolio_curve):
        return result
    return RunResultAdapter(result)


__all__ = ["RunResultAdapter", "adapt_run_result"]
