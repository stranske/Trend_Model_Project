import numpy as np
import pandas as pd

from trend_analysis.core.rank_selection import RiskStatsConfig
from trend_analysis.pipeline import _run_analysis
from trend_analysis.plugins import WeightEngine, weight_engine_registry

RUN_KWARGS = {"risk_free_column": "RF", "allow_risk_free_fallback": False}


@weight_engine_registry.register("negative_test_engine")
class NegativeTestEngine(WeightEngine):
    """Test-only engine that emits a deliberate short weight."""

    def weight(self, cov: pd.DataFrame) -> pd.Series:
        if cov.empty:
            return pd.Series(dtype=float)
        weights = pd.Series(0.0, index=cov.index, dtype=float)
        if len(weights) == 1:
            weights.iloc[0] = 1.0
        else:
            weights.iloc[0] = 0.6
            weights.iloc[1] = -0.2
            if len(weights) > 2:
                weights.iloc[2] = 0.6
        return weights


def make_dummy_returns(n_months: int = 24) -> pd.DataFrame:
    dates = pd.date_range("2022-01-31", periods=n_months, freq="ME")
    rng = np.random.default_rng(42)
    data = {
        "Date": dates,
        "FundA": rng.normal(0.01, 0.02, size=n_months),
        "FundB": rng.normal(0.008, 0.02, size=n_months),
        "FundC": rng.normal(0.011, 0.02, size=n_months),
        "FundD": rng.normal(0.009, 0.02, size=n_months),
        "RF": np.zeros(n_months),
    }
    return pd.DataFrame(data)


def test_pipeline_applies_cash_and_max_weight_constraints():
    df = make_dummy_returns()
    res = _run_analysis(
        df,
        in_start="2022-01",
        in_end="2022-12",
        out_start="2023-01",
        out_end="2023-12",
        target_vol=0.10,
        monthly_cost=0.0,
        selection_mode="all",
        stats_cfg=RiskStatsConfig(),
        constraints={
            "long_only": True,
            "max_weight": 0.4,
            "cash_weight": 0.1,
        },
        **RUN_KWARGS,
    )
    assert res is not None
    weights = res["fund_weights"]  # mapping fund->weight (excludes CASH synthetic line)
    assert isinstance(weights, dict)
    total = sum(weights.values())
    assert 0.99 <= total <= 1.01
    assert all(w <= 0.4 + 1e-9 for w in weights.values())
    # Ensure no CASH synthetic line leaked into fund weights
    assert "CASH" not in weights


def test_pipeline_max_weight_with_vol_adjust_enabled():
    df = make_dummy_returns()
    res = _run_analysis(
        df,
        in_start="2022-01",
        in_end="2022-12",
        out_start="2023-01",
        out_end="2023-12",
        target_vol=0.10,
        monthly_cost=0.0,
        selection_mode="all",
        stats_cfg=RiskStatsConfig(),
        constraints={
            "long_only": True,
            "max_weight": 0.35,
        },
        **RUN_KWARGS,
    )
    assert res is not None
    weights = res["fund_weights"]
    assert isinstance(weights, dict)
    assert all(weight <= 0.35 + 1e-9 for weight in weights.values())


def test_pipeline_long_only_blocks_negative_custom_weights():
    df = make_dummy_returns()
    custom_weights = {"FundA": 70.0, "FundB": -20.0, "FundC": 50.0, "FundD": 0.0}

    res_long_only = _run_analysis(
        df,
        in_start="2022-01",
        in_end="2022-12",
        out_start="2023-01",
        out_end="2023-12",
        target_vol=None,
        monthly_cost=0.0,
        selection_mode="all",
        custom_weights=custom_weights,
        stats_cfg=RiskStatsConfig(),
        constraints={"long_only": True},
        **RUN_KWARGS,
    )
    assert res_long_only is not None
    weights_long = res_long_only["fund_weights"]
    assert all(weight >= 0 for weight in weights_long.values())

    res_short_ok = _run_analysis(
        df,
        in_start="2022-01",
        in_end="2022-12",
        out_start="2023-01",
        out_end="2023-12",
        target_vol=None,
        monthly_cost=0.0,
        selection_mode="all",
        custom_weights=custom_weights,
        stats_cfg=RiskStatsConfig(),
        constraints={"long_only": False},
        **RUN_KWARGS,
    )
    assert res_short_ok is not None
    weights_short = res_short_ok["fund_weights"]
    assert any(weight < 0 for weight in weights_short.values())


def test_pipeline_long_only_clips_negative_weight_engine_weights():
    df = make_dummy_returns()
    res_long_only = _run_analysis(
        df,
        in_start="2022-01",
        in_end="2022-12",
        out_start="2023-01",
        out_end="2023-12",
        target_vol=None,
        monthly_cost=0.0,
        selection_mode="all",
        weighting_scheme="negative_test_engine",
        stats_cfg=RiskStatsConfig(),
        constraints={"long_only": True},
        **RUN_KWARGS,
    )
    assert res_long_only is not None
    weights_long = res_long_only["fund_weights"]
    assert all(weight >= 0 for weight in weights_long.values())

    res_short_ok = _run_analysis(
        df,
        in_start="2022-01",
        in_end="2022-12",
        out_start="2023-01",
        out_end="2023-12",
        target_vol=None,
        monthly_cost=0.0,
        selection_mode="all",
        weighting_scheme="negative_test_engine",
        stats_cfg=RiskStatsConfig(),
        constraints={"long_only": False},
        **RUN_KWARGS,
    )
    assert res_short_ok is not None
    weights_short = res_short_ok["fund_weights"]
    assert any(weight < 0 for weight in weights_short.values())
