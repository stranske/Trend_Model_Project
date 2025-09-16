import numpy as np
import pandas as pd
import pytest

from trend_analysis.core import rank_selection as rs


def make_simple_returns() -> pd.DataFrame:
    dates = pd.date_range("2021-01-31", periods=6, freq="ME")
    data = {
        "Alpha One": [0.02, 0.03, 0.01, 0.04, 0.00, 0.02],
        "Alpha Two": [0.03, 0.02, 0.01, 0.03, 0.02, 0.01],
        "Beta Core": [0.01, 0.02, 0.02, 0.01, 0.00, 0.01],
        "  gamma  ": [0.05, 0.01, 0.02, 0.03, 0.02, 0.02],
        "": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    }
    return pd.DataFrame(data, index=dates)


def test_rank_select_funds_normalises_blank_and_duplicate_columns():
    df = make_simple_returns().rename(columns={"Alpha Two": "  alpha one "})

    cfg = rs.RiskStatsConfig(risk_free=0.0)
    result = rs.rank_select_funds(
        df,
        cfg,
        inclusion_approach="top_n",
        n=df.shape[1],
    )

    names = set(result)
    assert "Alpha One" in names
    assert any(name.startswith("alpha one") for name in names)
    assert "Unnamed_5" in names  # empty header receives deterministic name


def test_rank_select_funds_limit_one_per_firm_backfills_duplicates():
    df = make_simple_returns()[["Alpha One", "Alpha Two", "Beta Core"]]
    cfg = rs.RiskStatsConfig(risk_free=0.0)

    selected = rs.rank_select_funds(
        df,
        cfg,
        inclusion_approach="top_n",
        n=3,
        limit_one_per_firm=True,
    )
    assert len(selected) == 3
    assert "Alpha Two" in selected  # backfilled after unique firms exhausted


def test_rank_select_funds_without_limit_one_per_firm_keeps_all():
    df = make_simple_returns()[["Alpha One", "Alpha Two", "Beta Core"]]
    cfg = rs.RiskStatsConfig(risk_free=0.0)

    selected = rs.rank_select_funds(
        df,
        cfg,
        inclusion_approach="top_n",
        n=2,
        limit_one_per_firm=False,
    )
    assert {"Alpha One", "Alpha Two"}.issubset(selected)


def test_rank_select_funds_threshold_branch_and_transform_alias():
    df = make_simple_returns()[["Alpha One", "Alpha Two", "Beta Core"]]
    cfg = rs.RiskStatsConfig(risk_free=0.0)

    selected = rs.rank_select_funds(
        df,
        cfg,
        inclusion_approach="threshold",
        threshold=0.5,
        transform_mode="rank",
    )
    assert isinstance(selected, list)


def test_rank_select_funds_blended_requires_weights():
    df = make_simple_returns()[["Alpha One", "Alpha Two"]]
    cfg = rs.RiskStatsConfig(risk_free=0.0)

    with pytest.raises(ValueError, match="blended score requires blended_weights"):
        rs.rank_select_funds(df, cfg, inclusion_approach="top_n", n=1, score_by="blended")


@pytest.mark.parametrize(
    "approach,kwargs,expected",
    [
        ("top_n", {"n": 2}, ["Alpha One", "Alpha Two"]),
        ("top_pct", {"pct": 0.5}, ["Alpha One", "Alpha Two"]),
        ("threshold", {"threshold": 0.1}, ["Alpha One", "Alpha Two", "Beta Core"]),
    ],
)
def test_some_function_missing_annotation_branches(approach, kwargs, expected):
    series = pd.Series([0.6, 0.5, 0.4], index=["Alpha One", "Alpha Two", "Beta Core"])
    result = rs.some_function_missing_annotation(series, approach, ascending=False, **kwargs)
    assert list(result) == expected[: len(result)]


def test_canonical_metric_list_alias_and_default():
    names = rs.canonical_metric_list(["annual_return", "Sharpe", "Custom"])
    assert names[:2] == ["AnnualReturn", "Sharpe"]

    all_metrics = rs.canonical_metric_list()
    assert "AnnualReturn" in all_metrics


def test_register_metric_decorator_registers_callable():
    @rs.register_metric("CoverageTest")
    def _dummy(series, **kwargs):
        return series.mean()

    df = make_simple_returns()[["Alpha One", "Alpha Two"]]
    cfg = rs.RiskStatsConfig(risk_free=0.0)
    result = rs._compute_metric_series(df, "CoverageTest", cfg)
    assert isinstance(result, pd.Series)


def test_quality_filter_and_private_variant_share_logic():
    df = make_simple_returns().reset_index().rename(columns={"index": "Date"})
    cfg = rs.FundSelectionConfig(max_missing_months=1, max_missing_ratio=0.5)
    eligible_public = rs.quality_filter(df, cfg)
    eligible_private = rs._quality_filter(df, eligible_public, "2021-01", "2021-12", cfg)
    assert eligible_public == eligible_private


def test_select_funds_random_mode_uses_numpy_choice(monkeypatch):
    df = make_simple_returns().reset_index().rename(columns={"index": "Date"})
    cfg = rs.FundSelectionConfig()

    called = {}

    def fake_choice(arr, size, replace):
        called["args"] = (tuple(arr), size, replace)
        return np.array(arr[:size])

    monkeypatch.setattr(np.random, "choice", fake_choice)

    result = rs.select_funds(
        df,
        "Date",
        mode="random",
        n=2,
        quality_cfg=cfg,
    )
    assert len(result) == 2
    assert called["args"][-1] is False
