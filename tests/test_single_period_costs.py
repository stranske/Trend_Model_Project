"""Tests for single-period transaction-cost handling (issue #5394 / A14).

The single-period analysis path charges ``run.monthly_cost`` and still supports
``portfolio.max_turnover`` / ``portfolio.lambda_tc`` turnover controls; it never
applies ``portfolio.transaction_cost_bps``, which is a multi-period-only turnover
lever. Setting it on a single-period run is therefore a silent no-op.
``run_simulation`` must warn loudly instead.
"""

from __future__ import annotations

import warnings

import pandas as pd

from trend_analysis import api
from trend_analysis.config import Config


def _make_df() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame({"Date": dates, "RF": 0.0, "A": 0.01, "B": 0.012})


def _make_single_period_cfg(transaction_cost_bps: float | None) -> Config:
    portfolio: dict[str, object] = {}
    if transaction_cost_bps is not None:
        portfolio["transaction_cost_bps"] = transaction_cost_bps
    return Config(
        version="1",
        data={
            "risk_free_column": "RF",
            "allow_risk_free_fallback": False,
            "date_column": "Date",
            "frequency": "M",
        },
        preprocessing={},
        vol_adjust={"target_vol": 1.0},
        sample_split={
            "in_start": "2020-01",
            "in_end": "2020-03",
            "out_start": "2020-04",
            "out_end": "2020-06",
        },
        portfolio=portfolio,
        metrics={},
        export={},
        run={},
    )


def _tc_warnings(records: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [
        r
        for r in records
        if issubclass(r.category, UserWarning)
        and "transaction_cost_bps" in str(r.message)
        and "single-period" in str(r.message)
    ]


def test_single_period_warns_when_transaction_cost_bps_set() -> None:
    """A non-zero ``transaction_cost_bps`` on a single-period run must warn."""
    df = _make_df()
    cfg = _make_single_period_cfg(transaction_cost_bps=25.0)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        api.run_simulation(cfg, df)

    matches = _tc_warnings(records)
    assert matches, (
        "expected a UserWarning that single-period transaction_cost_bps is ignored; "
        f"got categories/messages: {[(type(r.message).__name__, str(r.message)) for r in records]}"
    )
    assert "25.0" in str(matches[0].message)
    assert "portfolio.lambda_tc" in str(matches[0].message)


def test_single_period_silent_when_transaction_cost_bps_unset() -> None:
    """No transaction-cost warning when the key is absent (avoid noise)."""
    df = _make_df()
    cfg = _make_single_period_cfg(transaction_cost_bps=None)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        api.run_simulation(cfg, df)

    assert not _tc_warnings(records)
