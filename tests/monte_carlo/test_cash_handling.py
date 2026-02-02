import numpy as np
import pandas as pd
import pytest

from trend_analysis.pipeline import run_analysis


def _make_frame() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame(
        {
            "Date": dates,
            "RF": [0.01] * len(dates),
            "FundA": [0.02, 0.01, 0.03, 0.04, 0.01, 0.0],
            "FundB": [0.0, 0.01, -0.01, 0.02, 0.01, 0.0],
        }
    )


def test_cash_weight_series_and_returns_include_risk_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _make_frame()
    monkeypatch.setattr(
        "trend_analysis.pipeline.compute_trend_signals",
        lambda signal_inputs, *_args, **_kwargs: pd.DataFrame(
            {"FundA": 0.0, "FundB": np.nan}, index=signal_inputs.index
        ),
    )

    result = run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        target_vol=0.1,
        monthly_cost=0.0,
        custom_weights={"FundA": 80.0, "FundB": 20.0},
        weight_policy={"mode": "cash"},
        risk_free_column="RF",
    )

    assert result is not None
    cash_weight = result.get("cash_weight")
    assert cash_weight == pytest.approx(0.2)

    cash_series = result.get("cash_weight_series")
    assert isinstance(cash_series, pd.Series)
    assert cash_series.index.equals(result["out_sample_scaled"].index)
    assert np.allclose(cash_series.values, cash_weight)

    out_scaled = result["out_sample_scaled"]
    rf_out = result["risk_free_out_sample"]
    fund_weights = result["fund_weights"]
    weights = np.array([fund_weights.get(c, 0.0) for c in out_scaled.columns])
    expected = out_scaled.mul(weights, axis=1).sum(axis=1) + rf_out * cash_weight

    portfolio_user = result.get("portfolio_user_weight")
    assert isinstance(portfolio_user, pd.Series)
    pd.testing.assert_series_equal(portfolio_user, expected)


def test_cash_returns_use_override_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _make_frame()
    monkeypatch.setattr(
        "trend_analysis.pipeline.compute_trend_signals",
        lambda signal_inputs, *_args, **_kwargs: pd.DataFrame(
            {"FundA": 0.0, "FundB": np.nan}, index=signal_inputs.index
        ),
    )

    override_rate = 0.02
    result = run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        target_vol=0.1,
        monthly_cost=0.0,
        custom_weights={"FundA": 80.0, "FundB": 20.0},
        weight_policy={"mode": "cash"},
        risk_free_column="RF",
        risk_free_override=override_rate,
    )

    assert result is not None
    cash_weight = result.get("cash_weight")
    assert cash_weight == pytest.approx(0.2)

    rf_out = result["risk_free_out_sample"]
    assert np.allclose(rf_out.values, override_rate)

    out_scaled = result["out_sample_scaled"]
    fund_weights = result["fund_weights"]
    weights = np.array([fund_weights.get(c, 0.0) for c in out_scaled.columns])
    expected = out_scaled.mul(weights, axis=1).sum(axis=1) + override_rate * cash_weight

    portfolio_user = result.get("portfolio_user_weight")
    assert isinstance(portfolio_user, pd.Series)
    pd.testing.assert_series_equal(portfolio_user, expected)


def test_underinvestment_preserves_cash_without_cash_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _make_frame()
    monkeypatch.setattr(
        "trend_analysis.pipeline.compute_trend_signals",
        lambda signal_inputs, *_args, **_kwargs: pd.DataFrame(
            {"FundA": 0.0, "FundB": 0.0}, index=signal_inputs.index
        ),
    )

    result = run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        target_vol=0.1,
        monthly_cost=0.0,
        custom_weights={"FundA": 80.0, "FundB": 10.0},
        risk_free_column="RF",
    )

    assert result is not None
    cash_weight = result.get("cash_weight")
    assert cash_weight == pytest.approx(0.1)

    fund_weights = result.get("fund_weights")
    assert isinstance(fund_weights, dict)
    assert sum(fund_weights.values()) == pytest.approx(0.9)

    out_scaled = result["out_sample_scaled"]
    rf_out = result["risk_free_out_sample"]
    weights = np.array([fund_weights.get(c, 0.0) for c in out_scaled.columns])
    expected = out_scaled.mul(weights, axis=1).sum(axis=1) + rf_out * cash_weight

    portfolio_user = result.get("portfolio_user_weight")
    assert isinstance(portfolio_user, pd.Series)
    pd.testing.assert_series_equal(portfolio_user, expected)
