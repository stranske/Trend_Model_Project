"""Integration checks for robustness settings flowing through the pipeline."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import pytest

from trend_analysis import api


def _make_collinear_returns(periods: int = 24) -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=periods, freq="ME")
    rng = np.random.default_rng(17)
    base = rng.normal(0.01, 0.02, size=periods)
    return pd.DataFrame(
        {
            "Date": dates,
            "FundA": base + rng.normal(0, 0.0005, size=periods),
            "FundB": base * 1.5 + rng.normal(0, 0.0005, size=periods),
            "FundC": base * 0.5 + rng.normal(0, 0.0005, size=periods),
            "RF": np.full(periods, 0.001),
        }
    )


def _make_config(*, safe_mode: str) -> object:
    from types import SimpleNamespace

    return SimpleNamespace(
        sample_split={
            "in_start": "2020-01",
            "in_end": "2020-12",
            "out_start": "2021-01",
            "out_end": "2021-12",
        },
        data={
            "date_column": "Date",
            "frequency": "ME",
            "risk_free_column": "RF",
            "allow_risk_free_fallback": False,
        },
        portfolio={
            "selection_mode": "all",
            "weighting_scheme": "robust_mv",
            "constraints": {"long_only": True},
            "robustness": {
                "shrinkage": {"enabled": False},
                "condition_check": {
                    "enabled": True,
                    "threshold": 1.0,
                    "safe_mode": safe_mode,
                },
            },
        },
        vol_adjust={"enabled": False},
        metrics={},
        run={"monthly_cost": 0.0},
        benchmarks={},
    )


def _make_top_level_config(*, safe_mode: str) -> object:
    from types import SimpleNamespace

    return SimpleNamespace(
        sample_split={
            "in_start": "2020-01",
            "in_end": "2020-12",
            "out_start": "2021-01",
            "out_end": "2021-12",
        },
        data={
            "date_column": "Date",
            "frequency": "ME",
            "risk_free_column": "RF",
            "allow_risk_free_fallback": False,
        },
        portfolio={
            "selection_mode": "all",
            "weighting_scheme": "robust_mv",
            "constraints": {"long_only": True},
            "robustness": {
                "shrinkage": {"enabled": False},
                "condition_threshold": 1.0,
                "safe_mode": safe_mode,
                "diagonal_loading_factor": 1.0e-6,
            },
        },
        vol_adjust={"enabled": False},
        metrics={},
        run={"monthly_cost": 0.0},
        benchmarks={},
    )


def _extract_weights(result: api.RunResult) -> pd.Series:
    weights = result.details.get("fund_weights", {})
    return pd.Series(weights, dtype=float)


def test_condition_threshold_triggers_safe_mode() -> None:
    returns = _make_collinear_returns()
    cfg = _make_config(safe_mode="hrp")

    result = api.run_simulation(cfg, returns)

    fallback = result.fallback_info
    assert fallback is not None
    assert fallback["reason"] == "condition_threshold_exceeded"
    assert fallback["safe_mode"] == "hrp"
    assert fallback["condition_threshold"] == 1.0
    assert fallback["condition_source"] == "raw_cov"

    in_sample = (
        returns.set_index("Date")
        .loc["2020-01-31":"2020-12-31", ["FundA", "FundB", "FundC"]]
        .cov()
    )
    expected_condition = float(np.linalg.cond(in_sample.values))
    assert fallback["condition_number"] == pytest.approx(expected_condition)

    diagnostics = result.details.get("weight_engine_diagnostics", {})
    assert diagnostics.get("used_safe_mode") is True
    assert diagnostics.get("fallback_used") is True


def test_safe_mode_changes_fallback_weights() -> None:
    returns = _make_collinear_returns()

    hrp_result = api.run_simulation(_make_config(safe_mode="hrp"), returns)
    rp_result = api.run_simulation(_make_config(safe_mode="risk_parity"), returns)

    hrp_weights = _extract_weights(hrp_result)
    rp_weights = _extract_weights(rp_result)

    assert hrp_result.fallback_info is not None
    assert rp_result.fallback_info is not None
    assert not np.allclose(hrp_weights.values, rp_weights.values, rtol=1e-3, atol=1e-4)


def test_fallback_emits_warning(caplog: pytest.LogCaptureFixture) -> None:
    returns = _make_collinear_returns()
    cfg = _make_config(safe_mode="risk_parity")

    caplog.set_level(logging.WARNING)
    api.run_simulation(cfg, returns)

    assert "Ill-conditioned covariance matrix" in caplog.text


def test_condition_threshold_alias_triggers_safe_mode() -> None:
    returns = _make_collinear_returns()
    cfg = _make_config(safe_mode="risk_parity")
    cfg.portfolio["robustness"]["condition_check"] = {
        "enabled": True,
        "condition_threshold": 1.0,
        "safe_mode": "risk_parity",
    }

    result = api.run_simulation(cfg, returns)

    fallback = result.fallback_info
    assert fallback is not None
    assert fallback["reason"] == "condition_threshold_exceeded"
    assert fallback["safe_mode"] == "risk_parity"
    assert fallback["condition_threshold"] == 1.0
    diagnostics = result.details.get("weight_engine_diagnostics", {})
    assert diagnostics.get("used_safe_mode") is True


def test_top_level_robustness_keys_trigger_fallback() -> None:
    returns = _make_collinear_returns()
    cfg = _make_top_level_config(safe_mode="risk_parity")

    result = api.run_simulation(cfg, returns)

    fallback = result.fallback_info
    assert fallback is not None
    assert fallback["reason"] == "condition_threshold_exceeded"
    assert fallback["safe_mode"] == "risk_parity"
    assert fallback["condition_threshold"] == 1.0
    diagnostics = result.details.get("weight_engine_diagnostics", {})
    assert diagnostics.get("used_safe_mode") is True
