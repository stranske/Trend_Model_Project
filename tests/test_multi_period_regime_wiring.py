from __future__ import annotations

import pandas as pd

from trend_analysis.config import load_config
from trend_analysis.multi_period import run as run_multi_period


def _regime_returns_frame() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    return pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.02, 0.021, 0.019, 0.02],
            "FundB": [0.015, 0.014, 0.016, 0.015],
            "FundC": [0.025, 0.026, 0.024, 0.025],
            "FundD": [0.01, 0.011, 0.009, 0.01],
            "SPX": [-0.02, -0.03, -0.01, 0.02],
            "ACWI": [0.03, 0.02, 0.01, 0.0],
            "RF": [0.001] * 4,
        }
    )


def _make_config(regime_cfg: dict[str, object]) -> object:
    return load_config(
        {
            "version": "1",
            "data": {
                "allow_risk_free_fallback": False,
                "csv_path": "Trend Universe Data.csv",
                "date_column": "Date",
                "frequency": "ME",
                "risk_free_column": "RF",
            },
            "preprocessing": {},
            "vol_adjust": {
                "enabled": True,
                "target_vol": 1.0,
                "floor_vol": 0.0,
                "warmup_periods": 0,
                "window": {"length": 1, "decay": "simple", "lambda": 0.94},
            },
            "sample_split": {},
            "portfolio": {
                "rebalance_calendar": "NYSE",
                "max_turnover": 1.0,
                "transaction_cost_bps": 0.0,
                "selection_mode": "rank",
                "rank": {
                    "inclusion_approach": "top_n",
                    "n": 3,
                    "score_by": "Sharpe",
                    "transform": "raw",
                },
            },
            "benchmarks": {},
            "metrics": {},
            "export": {},
            "run": {},
            "multi_period": {
                "frequency": "M",
                "in_sample_len": 3,
                "out_sample_len": 1,
                "start": "2020-01",
                "end": "2020-04",
            },
            "regime": regime_cfg,
        }
    )


def _regime_settings(enabled: bool, proxy: str) -> dict[str, object]:
    return {
        "enabled": enabled,
        "proxy": proxy,
        "lookback": 1,
        "smoothing": 1,
        "threshold": 0.0,
        "neutral_band": 0.0,
        "min_observations": 1,
    }


def test_multi_period_regime_enabled_changes_selection_count() -> None:
    df = _regime_returns_frame()
    disabled_cfg = _make_config(_regime_settings(False, "SPX"))
    enabled_cfg = _make_config(_regime_settings(True, "SPX"))

    disabled = run_multi_period(disabled_cfg, df=df)
    enabled = run_multi_period(enabled_cfg, df=df)

    assert disabled and enabled
    assert len(disabled[0]["selected_funds"]) == 3
    assert len(enabled[0]["selected_funds"]) == 2


def test_multi_period_regime_proxy_changes_selection_count() -> None:
    df = _regime_returns_frame()
    spx_cfg = _make_config(_regime_settings(True, "SPX"))
    acwi_cfg = _make_config(_regime_settings(True, "ACWI"))

    spx = run_multi_period(spx_cfg, df=df)
    acwi = run_multi_period(acwi_cfg, df=df)

    assert spx and acwi
    assert len(spx[0]["selected_funds"]) != len(acwi[0]["selected_funds"])
