from types import SimpleNamespace

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
    unnamed_idx = list(df.columns).index("")
    # Column indices are 0-based but the sanitiser uses 1-based suffixes.
    expected_unnamed = f"Unnamed_{unnamed_idx + 1}"
    assert expected_unnamed in names  # empty header receives deterministic name


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


def test_rank_select_funds_excludes_bottom_k():
    dates = pd.date_range("2022-01-31", periods=6, freq="ME")
    df = pd.DataFrame(
        {
            "FundA": [0.04, 0.03, 0.05, 0.04, 0.03, 0.04],
            "FundB": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            "FundC": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            "FundD": [-0.01, -0.01, -0.01, -0.01, -0.01, -0.01],
        },
        index=dates,
    )
    cfg = rs.RiskStatsConfig(risk_free=0.0)

    selected = rs.rank_select_funds(
        df,
        cfg,
        inclusion_approach="top_n",
        n=4,
        score_by="annual_return",
        bottom_k=1,
    )

    assert "FundD" not in selected
    assert len(selected) == 3


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


def test_json_default_serialises_supported_types():
    sequence = rs._json_default(("a", "b"))
    assert sequence == ["a", "b"]

    unordered = rs._json_default({1, 2})
    assert sorted(unordered) == [1, 2]

    array = rs._json_default(np.array([3, 4]))
    assert array == [3, 4]

    scalar = rs._json_default(np.float32(5.5))
    assert scalar == pytest.approx(5.5)

    with pytest.raises(TypeError):
        rs._json_default(object())


def test_canonicalise_and_canonical_columns_behaviour():
    labels = ["Alpha", "Alpha", "", "Beta"]
    canonical = rs._canonicalise_labels(labels)
    assert canonical == ["Alpha", "Alpha_2", "Unnamed_3", "Beta"]

    empty = pd.DataFrame()
    assert rs._ensure_canonical_columns(empty) is empty

    df = pd.DataFrame([[1, 2]], columns=["X", "X"])
    canon_df = rs._ensure_canonical_columns(df)
    assert list(canon_df.columns) == ["X", "X_2"]
    assert list(df.columns) == ["X", "X"]


def test_stats_cfg_hash_accounts_for_dynamic_attributes():
    cfg = rs.RiskStatsConfig()
    base = rs._stats_cfg_hash(cfg)

    cfg.debug_flag = 1
    first = rs._stats_cfg_hash(cfg)
    cfg.debug_flag = 2
    second = rs._stats_cfg_hash(cfg)

    other = rs.RiskStatsConfig()
    other.debug_flag = 1

    assert base != first != second
    assert first == rs._stats_cfg_hash(other)


def test_make_window_key_and_cache_helpers_round_trip():
    cfg = rs.RiskStatsConfig()
    universe = ["B", "A"]
    key1 = rs.make_window_key("2021-01", "2021-06", universe, cfg)
    key2 = rs.make_window_key("2021-01", "2021-06", reversed(universe), cfg)
    assert key1 == key2

    rs.clear_window_metric_cache()
    df = make_simple_returns()[["Alpha One", "Alpha Two"]]
    bundle = rs.WindowMetricBundle(
        key=key1,
        start="2021-01",
        end="2021-06",
        freq="ME",
        stats_cfg_hash=rs._stats_cfg_hash(cfg),
        universe=tuple(df.columns),
        in_sample_df=df,
        _metrics=pd.DataFrame(index=df.columns, dtype=float),
    )
    rs.store_window_metric_bundle(key1, bundle)
    cached = rs.get_window_metric_bundle(key1)
    assert cached is bundle


def test_rank_select_funds_replaces_mismatched_cached_bundle():
    """A cached bundle with the wrong universe should be ignored."""

    df = make_simple_returns()[["Alpha One", "Beta Core"]]
    cfg = rs.RiskStatsConfig(risk_free=0.0)
    window_key = rs.make_window_key("2021-01", "2021-06", df.columns, cfg)

    rs.clear_window_metric_cache()

    wrong_bundle = rs.WindowMetricBundle(
        key=window_key,
        start="2021-01",
        end="2021-06",
        freq="ME",
        stats_cfg_hash=rs._stats_cfg_hash(cfg),
        universe=("Other",),
        in_sample_df=pd.DataFrame({"Other": df.iloc[:, 0]}),
        _metrics=pd.DataFrame(index=["Other"], dtype=float),
    )
    rs.store_window_metric_bundle(window_key, wrong_bundle)

    selected = rs.rank_select_funds(
        df,
        cfg,
        window_key=window_key,
        inclusion_approach="top_n",
        n=1,
    )

    assert selected and selected[0] in df.columns
    cached = rs.get_window_metric_bundle(window_key)
    assert cached is not wrong_bundle
    assert cached is not None
    assert cached.universe == tuple(df.columns)


def test_window_metric_bundle_covariance_branch(monkeypatch):
    df = pd.DataFrame(
        {
            "Alpha": [0.01, 0.02, 0.03],
            "Beta": [0.02, 0.01, 0.04],
        }
    )
    cfg = rs.RiskStatsConfig()
    stats_hash = rs._stats_cfg_hash(cfg)
    bundle = rs.WindowMetricBundle(
        key=None,
        start="2021-01",
        end="2021-03",
        freq="ME",
        stats_cfg_hash=stats_hash,
        universe=tuple(df.columns),
        in_sample_df=df,
        _metrics=pd.DataFrame(index=df.columns, dtype=float),
    )

    payload = SimpleNamespace(cov=np.array([[1.0, 0.5], [0.5, 1.0]]))
    calls = {"count": 0}

    def fake_cov(bundle_arg, cov_cache, *, enable_cov_cache, incremental_cov):
        calls["count"] += 1
        assert bundle_arg is bundle
        assert cov_cache is None
        assert enable_cov_cache is False
        assert incremental_cov is False
        return payload

    monkeypatch.setattr(rs, "_compute_covariance_payload", fake_cov)

    series = bundle.ensure_metric(
        "AvgCorr",
        cfg,
        cov_cache=None,
        enable_cov_cache=False,
        incremental_cov=False,
    )
    assert calls["count"] == 1
    assert bundle.cov_payload is payload
    assert series.name == "AvgCorr"
    assert pytest.approx(series.loc["Alpha"]) == 0.5
    assert pytest.approx(series.loc["Beta"]) == 0.5

    cached = bundle.ensure_metric("AvgCorr", cfg)
    assert calls["count"] == 1  # second call hits cache
    assert cached is series


def test_apply_transform_variants_and_errors():
    series = pd.Series([3.0, 2.0, 1.0], index=["A", "B", "C"])

    raw = rs._apply_transform(series, mode="raw")
    assert raw is series

    rank = rs._apply_transform(series, mode="rank")
    assert list(rank) == [1.0, 2.0, 3.0]

    percentile = rs._apply_transform(series, mode="percentile", rank_pct=1 / 3)
    assert percentile.notna().sum() == 1

    constant = pd.Series([1.0, 1.0, 1.0], index=series.index)
    zscored = rs._apply_transform(constant, mode="zscore", window=10)
    assert all(value == 0.0 for value in zscored)

    with pytest.raises(ValueError):
        rs._apply_transform(series, mode="percentile")

    with pytest.raises(ValueError):
        rs._apply_transform(series, mode="unknown")
