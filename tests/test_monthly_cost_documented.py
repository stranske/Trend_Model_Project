"""Regression coverage for the documented ``run.monthly_cost`` knob."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from trend_analysis import api
from trend_analysis.config import Config


def _make_df() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame(
        {
            "Date": dates,
            "RF": 0.0,
            "A": [0.010, 0.013, 0.012, 0.014, 0.011, 0.015],
            "B": [0.008, 0.010, 0.009, 0.012, 0.010, 0.013],
        }
    )


def _make_cfg(monthly_cost: float) -> Config:
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
        portfolio={},
        metrics={},
        export={},
        run={"monthly_cost": monthly_cost},
    )


def test_defaults_document_run_monthly_cost() -> None:
    defaults = yaml.safe_load(Path("config/defaults.yml").read_text())

    assert "monthly_cost" in defaults["run"]
    assert defaults["run"]["monthly_cost"] == pytest.approx(0.0)


def test_run_monthly_cost_lowers_net_periodic_returns() -> None:
    returns = _make_df()
    baseline = api.run_simulation(_make_cfg(0.0), returns.copy())
    charged = api.run_simulation(_make_cfg(0.0025), returns.copy())

    baseline_scaled = baseline.details["out_sample_scaled"][["A", "B"]]
    charged_scaled = charged.details["out_sample_scaled"][["A", "B"]]

    pd.testing.assert_frame_equal(charged_scaled, baseline_scaled - 0.0025)
