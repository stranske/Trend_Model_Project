from __future__ import annotations

from pathlib import Path

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
                "cost_model": {"per_trade_bps": 0.0, "half_spread_bps": 0},
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


def test_multi_period_lookback_reaches_labels_selection_and_regime_report() -> None:
    """Two unclamped windows see different signals in the same real engine run."""
    frame = pd.read_csv(
        Path(__file__).parent / "fixtures" / "regime_lookback_sensitive.csv",
        parse_dates=["Date"],
    )
    configs = []
    for lookback in (2, 5):
        settings = _regime_settings(True, "SPX")
        settings["lookback"] = lookback
        config = _make_config(settings)
        config.multi_period.update(in_sample_len=6, end="2020-08")
        assert config.multi_period["in_sample_len"] > lookback
        configs.append(config)
    short_config, long_config = (config.model_dump() for config in configs)
    assert short_config["regime"].pop("lookback") == 2
    assert long_config["regime"].pop("lookback") == 5
    assert short_config == long_config

    short = run_multi_period(configs[0], df=frame.copy())
    long = run_multi_period(configs[1], df=frame.copy())
    assert len(short) == len(long) == 2
    # The six-month in-sample window ends in two negative proxy returns:
    # their compounded return is negative, while the five-month return is positive.
    in_end = pd.Timestamp("2020-06-30")
    assert short[0]["regime_labels"].loc[in_end] == "Risk-Off"
    assert long[0]["regime_labels"].loc[in_end] == "Risk-On"
    assert len(short[0]["selected_funds"]) == 2
    assert len(long[0]["selected_funds"]) == 3
    # The first actual out-of-sample report also receives the changed label.
    out_date = pd.Timestamp("2020-07-31")
    assert short[0]["regime_labels_out"].loc[out_date] == "Risk-Off"
    assert long[0]["regime_labels_out"].loc[out_date] == "Risk-On"
    for result, label in ((short[0], "Risk-Off"), (long[0], "Risk-On")):
        table = result["performance_by_regime"]
        other = "Risk-On" if label == "Risk-Off" else "Risk-Off"
        assert table.loc["Observations", ("User", label)] == 1
        assert table.loc["Observations", ("User", other)] == 0
