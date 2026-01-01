from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from trend_analysis import api


def _make_returns(periods: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2010-01-31", periods=periods, freq="ME")
    base = np.linspace(0.01, 0.05, periods)
    data = {
        "Date": dates,
        "FundA": base,
        "FundB": base[::-1] * 0.9,
        "FundC": np.sin(np.linspace(0.0, 6.0, periods)) * 0.02,
    }
    return pd.DataFrame(data)


def _make_config(returns: pd.DataFrame, signals: dict[str, object]) -> SimpleNamespace:
    dates = returns["Date"]
    split = {
        "in_start": dates.iloc[0].strftime("%Y-%m"),
        "in_end": dates.iloc[79].strftime("%Y-%m"),
        "out_start": dates.iloc[80].strftime("%Y-%m"),
        "out_end": dates.iloc[-1].strftime("%Y-%m"),
    }
    return SimpleNamespace(
        seed=11,
        sample_split=split,
        metrics={},
        vol_adjust={"target_vol": 1.0},
        run={"monthly_cost": 0.0},
        data={"allow_risk_free_fallback": True},
        portfolio={
            "selection_mode": "all",
            "random_n": 3,
            "custom_weights": None,
            "rank": {},
            "manual_list": None,
            "indices_list": None,
            "weighting_scheme": "equal",
        },
        benchmarks={},
        signals=signals,
    )


def _signal_frames_differ(frame_a: pd.DataFrame, frame_b: pd.DataFrame) -> bool:
    return frame_a.fillna(0.0).ne(frame_b.fillna(0.0)).to_numpy().any()


def test_run_simulation_trend_window_changes_output() -> None:
    returns = _make_returns()
    config_short = _make_config(
        returns,
        {"window": 20, "lag": 1, "zscore": False, "vol_adjust": False},
    )
    config_long = _make_config(
        returns,
        {"window": 60, "lag": 1, "zscore": False, "vol_adjust": False},
    )

    short_result = api.run_simulation(config_short, returns)
    long_result = api.run_simulation(config_long, returns)

    short_signals = short_result.details["signal_frame"]
    long_signals = long_result.details["signal_frame"]

    assert _signal_frames_differ(short_signals, long_signals)


def test_run_simulation_trend_zscore_changes_output() -> None:
    returns = _make_returns()
    config_base = _make_config(
        returns,
        {"window": 20, "lag": 1, "zscore": False, "vol_adjust": False},
    )
    config_zscore = _make_config(
        returns,
        {"window": 20, "lag": 1, "zscore": True, "vol_adjust": False},
    )

    base_result = api.run_simulation(config_base, returns)
    zscore_result = api.run_simulation(config_zscore, returns)

    base_signals = base_result.details["signal_frame"]
    zscore_signals = zscore_result.details["signal_frame"]

    assert _signal_frames_differ(base_signals, zscore_signals)
