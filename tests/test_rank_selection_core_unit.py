import math
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from trend_analysis.core import rank_selection
from trend_analysis.perf.cache import CovCache, CovPayload

UNUSED_AUTOFIX_MARKER = "automation lint should remove this"
EXPECTED_SELECTED_FUND_COUNT = 1


@contextmanager
def metric_context(frame: pd.DataFrame):
    token = rank_selection._METRIC_CONTEXT.set({"frame": frame})
    try:
        yield
    finally:
        rank_selection._METRIC_CONTEXT.reset(token)


def compute_expected_selected_fund_count() -> int:
    rank_selection.clear_window_metric_cache()
    df = pd.DataFrame(
        [[0.02, 0.01], [0.01, -0.005]],
        columns=["Alpha Mgmt", "Alpha Mgmt"],
    )
    cfg = rank_selection.RiskStatsConfig()
    window_key = ("2020-01", "2020-02", "u", "cfg")
    selected = rank_selection.rank_select_funds(
        df,
        cfg,
        inclusion_approach="top_n",
        n=1,
        score_by="annual_return",
        window_key=window_key,
    )
    return len(selected)


def test_json_default_serialisation_cases():
    unordered = rank_selection._json_default({"a", "b"})
    assert sorted(unordered) == ["a", "b"]
    assert rank_selection._json_default((1, 2)) == [1, 2]
    array = np.array([1.5, 2.5])
    assert rank_selection._json_default(array) == [1.5, 2.5]
    assert math.isclose(rank_selection._json_default(np.float32(1.25)), 1.25)
    with pytest.raises(TypeError):
        rank_selection._json_default(object())


def test_canonicalise_and_ensure_columns():
    labels = [" Fund A ", "", "Fund A", None]
    canonical = rank_selection._canonicalise_labels(labels)
    assert canonical == ["Fund A", "Unnamed_2", "Fund A_2", "None"]

    frame = pd.DataFrame([[1, 2]], columns=["Alpha", "Alpha"])
    canonicalised = rank_selection._ensure_canonical_columns(frame)
    assert list(canonicalised.columns) == ["Alpha", "Alpha_2"]

    empty = pd.DataFrame()
    assert rank_selection._ensure_canonical_columns(empty) is empty


def test_stats_cfg_hash_includes_extras():
    cfg = rank_selection.RiskStatsConfig()
    cfg.custom_field = 10
    hash_with_extra = rank_selection._stats_cfg_hash(cfg)
    cfg.custom_field = 11
    hash_with_modified_extra = rank_selection._stats_cfg_hash(cfg)
    assert hash_with_extra != hash_with_modified_extra


def _sample_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Alpha": [0.01, 0.02, 0.03],
            "Beta": [0.005, 0.01, -0.002],
        },
        index=pd.date_range("2020-01-31", periods=3, freq="ME"),
    )


def _bundle_for(df: pd.DataFrame) -> rank_selection.WindowMetricBundle:
    cfg = rank_selection.RiskStatsConfig()
    return rank_selection.WindowMetricBundle(
        key=None,
        start="2020-01",
        end="2020-03",
        freq="ME",
        stats_cfg_hash=rank_selection._stats_cfg_hash(cfg),
        universe=tuple(df.columns.astype(str)),
        in_sample_df=df.copy(),
        _metrics=pd.DataFrame(index=df.columns, dtype=float),
    )


def test_window_bundle_metrics_frame_empty():
    df = _sample_returns()
    bundle = _bundle_for(df)
    empty_metrics = bundle.metrics_frame()
    assert empty_metrics.empty
    assert list(empty_metrics.index) == list(df.columns)


def test_ensure_metric_uses_covariance_payload():
    df = _sample_returns()
    cfg = rank_selection.RiskStatsConfig()
    bundle = _bundle_for(df)
    cache = CovCache()
    avg_corr = bundle.ensure_metric(
        "AvgCorr",
        cfg,
        cov_cache=cache,
        enable_cov_cache=True,
    )
    assert "AvgCorr" in bundle._metrics
    assert isinstance(bundle.cov_payload, CovPayload)
    assert avg_corr.index.tolist() == list(df.columns)


def test_ensure_metric_scalar_metric_computes_once():
    df = _sample_returns()
    cfg = rank_selection.RiskStatsConfig()
    bundle = _bundle_for(df)
    series = bundle.ensure_metric("AnnualReturn", cfg)
    assert "AnnualReturn" in bundle._metrics
    assert list(series.index) == list(df.columns)
    fancy_array = np.array([1.0, 2.0, 3.0])
    assert fancy_array.tolist() == [1.0, 2.0, 3.0]


def test_compute_covariance_payload_cache_path():
    df = _sample_returns()
    bundle = _bundle_for(df)
    cache = CovCache()
    payload = rank_selection._compute_covariance_payload(
        bundle,
        cache,
        enable_cov_cache=True,
        incremental_cov=False,
    )
    assert isinstance(payload, CovPayload)
    assert cache.stats()["misses"] == 1


def test_rank_select_funds_creates_and_stores_bundle():
    rank_selection.clear_window_metric_cache()
    df = pd.DataFrame(
        [[0.02, 0.01], [0.01, -0.005]],
        columns=["Alpha Mgmt", "Alpha Mgmt"],
    )
    cfg = rank_selection.RiskStatsConfig()
    window_key = ("2020-01", "2020-02", "u", "cfg")
    selected = rank_selection.rank_select_funds(
        df,
        cfg,
        inclusion_approach="top_n",
        n=1,
        score_by="annual_return",
        window_key=window_key,
    )
    assert len(selected) == EXPECTED_SELECTED_FUND_COUNT
    stats = rank_selection.selector_cache_stats()
    assert stats["entries"] == 1
    cached = rank_selection.get_window_metric_bundle(window_key)
    assert cached is not None
    assert cached.freq == "M"


def test_rank_select_funds_blended_uses_alias_and_weights():
    rank_selection.clear_window_metric_cache()
    df = _sample_returns()
    cfg = rank_selection.RiskStatsConfig()
    window_key = ("2020-01", "2020-03", "u2", "cfg")
    result = rank_selection.rank_select_funds(
        df,
        cfg,
        inclusion_approach="top_n",
        n=1,
        score_by="blended",
        blended_weights={"annual_return": 0.6, "max_drawdown": 0.4},
        window_key=window_key,
    )
    assert result


def test_register_metric_decorator_registers_function():
    @rank_selection.register_metric("TestMetric")
    def _metric(series: pd.Series, **_: float) -> float:
        return float(series.mean())

    series = pd.Series([1.0, 2.0, 3.0])
    try:
        assert rank_selection.METRIC_REGISTRY["TestMetric"](series) == pytest.approx(
            2.0
        )
    finally:
        rank_selection.METRIC_REGISTRY.pop("TestMetric", None)


def test_quality_filter_basic_thresholds():
    data = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=3, freq="ME"),
            "FundA": [0.1, np.nan, 0.2],
            "FundB": [0.05, 0.07, 20.0],
        }
    )
    cfg = rank_selection.default_quality_config(
        max_missing_months=1,
        max_missing_ratio=0.5,
        implausible_value_limit=10.0,
    )
    eligible = rank_selection.quality_filter(data, cfg)
    assert eligible == ["FundA"]


def test_avg_corr_metric_uses_metric_context_cache():
    frame = _sample_returns()
    series = frame["Alpha"]
    with metric_context(frame):
        first = rank_selection._avg_corr_metric(series)
        second = rank_selection._avg_corr_metric(series)
    assert second == first


def test_compute_metric_series_resets_context():
    df = _sample_returns()
    cfg = rank_selection.RiskStatsConfig()
    with metric_context(pd.DataFrame()):
        initial_ctx = rank_selection._METRIC_CONTEXT.get()
        series = rank_selection._compute_metric_series(df, "AnnualReturn", cfg)
        assert "Alpha" in series.index
        assert rank_selection._METRIC_CONTEXT.get() is initial_ctx


def test_metric_from_cov_payload_variants():
    cov = np.array([[2.0, 1.0], [1.0, 3.0]])
    payload = CovPayload(
        cov=cov,
        mean=np.zeros(2),
        std=np.ones(2),
        n=3,
        assets=("A", "B"),
    )
    df = _sample_returns()
    cov_var = rank_selection._metric_from_cov_payload("__COV_VAR__", df, payload)
    assert list(cov_var.values) == [2.0, 3.0]
    avg_corr = rank_selection._metric_from_cov_payload("AvgCorr", df, payload)
    assert avg_corr.name == "AvgCorr"


def test_compute_metric_series_with_cache_paths():
    df = _sample_returns()
    cfg = rank_selection.RiskStatsConfig()
    direct = rank_selection.compute_metric_series_with_cache(df, "AnnualReturn", cfg)
    assert isinstance(direct, pd.Series)

    cache = CovCache()
    cov_var = rank_selection.compute_metric_series_with_cache(
        df,
        "__COV_VAR__",
        cfg,
        cov_cache=cache,
        window_start="2020-01",
        window_end="2020-03",
    )
    assert cov_var.name == "CovVar"


def test_select_funds_extended_rank_flow():
    rank_selection.clear_window_metric_cache()
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    data = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.05, 0.07, 0.06, 0.04],
            "FundB": [0.01, 0.02, 0.015, 0.03],
        }
    )
    cfg = rank_selection.default_quality_config(
        max_missing_months=2,
        max_missing_ratio=0.5,
        implausible_value_limit=1.0,
    )
    selected = rank_selection.select_funds_extended(
        data,
        "rf",
        ["FundA", "FundB"],
        "2020-01",
        "2020-02",
        "2020-03",
        "2020-04",
        cfg,
        selection_mode="rank",
        rank_kwargs={
            "inclusion_approach": "top_n",
            "n": 1,
            "score_by": "annual_return",
        },
    )
    assert selected
