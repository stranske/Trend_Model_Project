import numpy as np
import pandas as pd

from trend_analysis.pipeline import _run_analysis
from trend_analysis.risk import RiskWindow, compute_constrained_weights, realised_volatility


def _heterogeneous_returns(periods: int = 24) -> pd.DataFrame:
    index = pd.date_range("2022-01-31", periods=periods, freq="ME")
    steps = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "Steady": 0.006 + 0.003 * np.sin(steps / 2.0),
            "Volatile": 0.008 + 0.050 * np.sin(steps / 1.7),
        },
        index=index,
    )


def _analysis_with_heterogeneous_returns(
    *,
    target_vol: float = 0.12,
    monthly_cost: float = 0.0,
    risk_free_rate: float = 0.0,
) -> tuple[dict[object, object], pd.DataFrame]:
    returns = _heterogeneous_returns().assign(RF=risk_free_rate).reset_index(names="Date")
    result = _run_analysis(
        returns,
        in_start="2022-01",
        in_end="2022-12",
        out_start="2023-01",
        out_end="2023-12",
        target_vol=target_vol,
        monthly_cost=monthly_cost,
        selection_mode="all",
        risk_free_column="RF",
        allow_risk_free_fallback=False,
    )
    assert result is not None
    return result, returns


def test_scale_factor_has_one_effective_owner() -> None:
    returns = _heterogeneous_returns()
    weights, diagnostics = compute_constrained_weights(
        {"Steady": 0.5, "Volatile": 0.5},
        returns,
        window=RiskWindow(length=6),
        target_vol=0.12,
        periods_per_year=12,
        floor_vol=None,
        long_only=True,
        max_weight=None,
    )

    expected = diagnostics.scale_factors / diagnostics.scale_factors.sum()
    pd.testing.assert_series_equal(weights, expected, check_names=False)
    assert not np.isclose(
        diagnostics.scale_factors["Steady"], diagnostics.scale_factors["Volatile"]
    )

    raw_portfolio = returns.mul(weights, axis=1).sum(axis=1)
    expected_volatility = realised_volatility(
        raw_portfolio.to_frame("portfolio"), RiskWindow(length=6), periods_per_year=12
    )["portfolio"]
    pd.testing.assert_series_equal(diagnostics.portfolio_volatility, expected_volatility)

    result, pipeline_returns = _analysis_with_heterogeneous_returns()
    expected_user = (
        pipeline_returns.set_index("Date")
        .loc["2023-01-01":, ["Steady", "Volatile"]]
        .mul(pd.Series(result["fund_weights"]), axis=1)
        .sum(axis=1)
    )
    pd.testing.assert_series_equal(
        result["portfolio_user_weight"], expected_user, check_names=False, check_freq=False
    )


def test_portfolio_volatility_diagnostic_matches_user_returns() -> None:
    result, returns = _analysis_with_heterogeneous_returns()
    raw_out_returns = returns.set_index("Date").loc["2023-01-01":, ["Steady", "Volatile"]]
    expected_user = raw_out_returns.mul(pd.Series(result["fund_weights"]), axis=1).sum(axis=1)
    pd.testing.assert_series_equal(
        result["portfolio_user_weight"], expected_user, check_names=False, check_freq=False
    )
    reported = result["risk_diagnostics"]["portfolio_volatility"]
    expected = realised_volatility(
        expected_user.to_frame("portfolio"), RiskWindow(length=12), periods_per_year=12
    )["portfolio"]
    expected.index.name = None
    pd.testing.assert_series_equal(reported, expected, check_freq=False)


def test_target_volatility_retains_its_exposure_magnitude() -> None:
    returns = _heterogeneous_returns()
    low_target_weights, low_target_diagnostics = compute_constrained_weights(
        {"Steady": 0.5, "Volatile": 0.5},
        returns,
        window=RiskWindow(length=6),
        target_vol=0.001,
        periods_per_year=12,
        floor_vol=None,
        long_only=True,
        max_weight=None,
    )
    high_target_weights, high_target_diagnostics = compute_constrained_weights(
        {"Steady": 0.5, "Volatile": 0.5},
        returns,
        window=RiskWindow(length=6),
        target_vol=0.50,
        periods_per_year=12,
        floor_vol=None,
        long_only=True,
        max_weight=None,
    )

    assert low_target_weights.sum() < high_target_weights.sum()
    assert low_target_weights.sum() < 1.0
    assert high_target_weights.sum() == 1.0
    assert (
        low_target_diagnostics.portfolio_volatility.dropna().iloc[-1]
        < high_target_diagnostics.portfolio_volatility.dropna().iloc[-1]
    )


def test_equal_weight_comparison_uses_documented_scaling_contract() -> None:
    result, returns = _analysis_with_heterogeneous_returns()
    out_returns = returns.set_index("Date").loc["2023-01-01":, ["Steady", "Volatile"]]
    expected_equal_weight = out_returns.mean(axis=1)
    pd.testing.assert_series_equal(
        result["portfolio_equal_weight"], expected_equal_weight, check_names=False, check_freq=False
    )


def test_partial_exposure_and_cash_returns_match_reported_volatility() -> None:
    result, _returns = _analysis_with_heterogeneous_returns(
        target_vol=0.001,
        monthly_cost=0.0005,
        risk_free_rate=0.01,
    )
    reported = result["risk_diagnostics"]["portfolio_volatility"]
    expected = realised_volatility(
        result["portfolio_user_weight"].to_frame("portfolio"),
        RiskWindow(length=12),
        periods_per_year=12,
    )["portfolio"]
    expected.index.name = None
    pd.testing.assert_series_equal(reported, expected, check_freq=False)


def test_turnover_respects_exposure_scaled_final_weights() -> None:
    returns = _heterogeneous_returns()
    prev = pd.Series({"Steady": 0.5, "Volatile": 0.5})
    _, diagnostics = compute_constrained_weights(
        prev,
        returns,
        window=RiskWindow(length=6),
        target_vol=0.001,
        periods_per_year=12,
        floor_vol=None,
        long_only=True,
        max_weight=None,
        previous_weights=prev,
        max_turnover=0.01,
    )
    assert diagnostics.turnover_value <= 0.01 + 1e-9
