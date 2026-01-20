import pandas as pd

from trend_analysis.reporting.portfolio_series import select_primary_portfolio_series


def test_select_primary_portfolio_series_prefers_user_weight_combined() -> None:
    combined = pd.Series([0.05, 0.02], index=[0, 1])
    user = pd.Series([0.1, 0.03], index=[0, 1])
    equal_combined = pd.Series([0.2, 0.04], index=[0, 1])
    equal = pd.Series([0.3, 0.05], index=[0, 1])
    res = {
        "portfolio_user_weight_combined": combined,
        "portfolio_user_weight": user,
        "portfolio_equal_weight_combined": equal_combined,
        "portfolio_equal_weight": equal,
    }

    selected = select_primary_portfolio_series(res)

    pd.testing.assert_series_equal(selected, combined)


def test_select_primary_portfolio_series_falls_back_to_fund_weights() -> None:
    out_sample_scaled = pd.DataFrame(
        {"A": [0.1, 0.0], "B": [0.05, 0.02]},
        index=pd.date_range("2024-01-31", periods=2, freq="M"),
    )
    res = {
        "out_sample_scaled": out_sample_scaled,
        "fund_weights": {"A": 2.0, "B": 1.0},
    }

    selected = select_primary_portfolio_series(res)

    expected_weights = pd.Series({"A": 2.0 / 3.0, "B": 1.0 / 3.0})
    expected = out_sample_scaled.mul(expected_weights, axis=1).sum(axis=1)
    pd.testing.assert_series_equal(selected, expected)


def test_select_primary_portfolio_series_uses_ew_weights_when_fund_missing() -> None:
    out_sample_scaled = pd.DataFrame(
        {"A": [0.1, 0.0], "B": [0.05, 0.02]},
        index=pd.date_range("2024-01-31", periods=2, freq="M"),
    )
    res = {
        "out_sample_scaled": out_sample_scaled,
        "ew_weights": {"A": 1.0, "B": 3.0},
    }

    selected = select_primary_portfolio_series(res)

    expected_weights = pd.Series({"A": 0.25, "B": 0.75})
    expected = out_sample_scaled.mul(expected_weights, axis=1).sum(axis=1)
    pd.testing.assert_series_equal(selected, expected)


def test_select_primary_portfolio_series_defaults_to_equal_weights() -> None:
    out_sample_scaled = pd.DataFrame(
        {"A": [0.1, 0.0], "B": [0.05, 0.02]},
        index=pd.date_range("2024-01-31", periods=2, freq="M"),
    )
    res = {"out_sample_scaled": out_sample_scaled}

    selected = select_primary_portfolio_series(res)

    expected_weights = pd.Series({"A": 0.5, "B": 0.5})
    expected = out_sample_scaled.mul(expected_weights, axis=1).sum(axis=1)
    pd.testing.assert_series_equal(selected, expected)
