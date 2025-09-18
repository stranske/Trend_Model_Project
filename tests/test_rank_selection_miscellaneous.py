import numpy as np
import pandas as pd
import pytest

from trend_analysis.core import rank_selection
from trend_analysis.perf.cache import CovCache, CovPayload


@pytest.fixture(autouse=True)
def reset_cache_state():
    """Ensure selector cache counters do not leak between tests."""
    rank_selection.clear_window_metric_cache()
    yield
    rank_selection.clear_window_metric_cache()


def test_canonicalise_labels_handles_duplicates_and_blanks():
    labels = [" Fund A ", "", "Fund A", "Fund B", "Fund B"]
    canonical = rank_selection._canonicalise_labels(labels)
    assert canonical == [
        "Fund A",
        "Unnamed_2",
        "Fund A_2",
        "Fund B",
        "Fund B_2",
    ]


def test_ensure_canonical_columns_idempotent_and_mutating_copy():
    frame = pd.DataFrame({"Col A": [1, 2], "Col B": [3, 4]})
    same = rank_selection._ensure_canonical_columns(frame)
    assert same is frame

    messy = pd.DataFrame({" Col A ": [1], "": [2]})
    cleaned = rank_selection._ensure_canonical_columns(messy)
    assert list(cleaned.columns) == ["Col A", "Unnamed_2"]
    assert cleaned is not messy


def test_stats_cfg_hash_includes_dynamic_attributes():
    cfg = rank_selection.RiskStatsConfig()
    baseline = rank_selection._stats_cfg_hash(cfg)

    cfg.custom_setting = {"alpha": 0.5}
    with_extras = rank_selection._stats_cfg_hash(cfg)
    assert with_extras != baseline

    cfg_clone = rank_selection.RiskStatsConfig()
    cfg_clone.custom_setting = {"alpha": 0.5}
    assert rank_selection._stats_cfg_hash(cfg_clone) == with_extras


@pytest.fixture
def sample_bundle():
    df = pd.DataFrame(
        {
            "Alpha Capital A": [0.01, 0.02, 0.015, 0.018],
            "Alpha Capital B": [0.012, 0.022, 0.016, 0.02],
        }
    )
    cfg_hash = rank_selection._stats_cfg_hash(rank_selection.RiskStatsConfig())
    bundle = rank_selection.WindowMetricBundle(
        key=None,
        start="2020-01",
        end="2020-04",
        freq="M",
        stats_cfg_hash=cfg_hash,
        universe=tuple(df.columns),
        in_sample_df=df,
        _metrics=pd.DataFrame(index=df.columns, dtype=float),
    )
    return bundle


def test_window_metric_bundle_metrics_frame_and_cache(sample_bundle):
    bundle = sample_bundle
    empty_metrics = bundle.metrics_frame()
    assert empty_metrics.index.tolist() == list(bundle.in_sample_df.columns)
    assert empty_metrics.empty

    cfg = rank_selection.RiskStatsConfig()
    cache = CovCache()
    avg_corr = bundle.ensure_metric("AvgCorr", cfg, cov_cache=cache)
    assert avg_corr.between(-1, 1).all()
    assert bundle.cov_payload is not None

    cached_again = bundle.ensure_metric("AvgCorr", cfg)
    assert cached_again.equals(avg_corr)

    variances = bundle.ensure_metric("__COV_VAR__", cfg)
    assert list(variances.index) == list(bundle.in_sample_df.columns)
    assert (variances >= 0).all()


def test_cov_metric_from_payload_handles_single_asset():
    payload = CovPayload(
        cov=np.array([[0.04]]),
        mean=np.array([0.01]),
        std=np.array([0.2]),
        n=5,
        assets=("Solo",),
    )
    series = rank_selection._cov_metric_from_payload("AvgCorr", payload, ["Solo"])
    assert series.name == "AvgCorr"
    assert series.iloc[0] == 0.0

    variances = rank_selection._cov_metric_from_payload("__COV_VAR__", payload, ["Solo"])
    assert variances.iloc[0] == pytest.approx(0.04)


def test_get_metric_context_requires_context_token():
    with pytest.raises(RuntimeError):
        rank_selection._get_metric_context()

    token = rank_selection._METRIC_CONTEXT.set({"frame": "dummy"})
    try:
        context = rank_selection._get_metric_context()
        assert context == {"frame": "dummy"}
    finally:
        rank_selection._METRIC_CONTEXT.reset(token)


def test_quality_filters_remove_missing_and_implausible():
    dates = pd.period_range("2020-01", periods=4, freq="M").to_timestamp("M")
    df = pd.DataFrame(
        {
            "Date": dates,
            "Fund A": [0.01, 0.02, np.nan, np.nan],
            "Fund B": [0.5, 0.6, 0.7, 0.8],
            "Fund C": [0.01, 0.01, 0.01, 0.01],
        }
    )
    cfg = rank_selection.FundSelectionConfig(
        max_missing_months=1,
        max_missing_ratio=0.3,
        implausible_value_limit=0.3,
    )
    eligible = rank_selection.quality_filter(df, cfg)
    assert eligible == ["Fund C"]

    window = rank_selection._quality_filter(
        df,
        ["Fund A", "Fund C"],
        in_sdate="2020-01",
        out_edate="2020-04",
        cfg=cfg,
    )
    assert window == ["Fund C"]


def test_some_function_missing_annotation_branches():
    scores = pd.Series({"Fund A": 2.0, "Fund B": 1.5, "Fund C": 1.0})
    assert rank_selection.some_function_missing_annotation(scores, "top_n", n=2) == [
        "Fund C",
        "Fund B",
    ]

    assert rank_selection.some_function_missing_annotation(scores, "top_pct", pct=0.5) == [
        "Fund C",
        "Fund B",
    ]

    assert rank_selection.some_function_missing_annotation(
        scores,
        "threshold",
        threshold=1.5,
        ascending=False,
    ) == ["Fund A", "Fund B"]

    assert (
        rank_selection.some_function_missing_annotation(scores, "unsupported")
        == []
    )


def test_rank_select_funds_validates_bundle_alignment(sample_bundle):
    bundle = sample_bundle
    cfg = rank_selection.RiskStatsConfig()
    df = bundle.in_sample_df.rename(columns={c: f"X_{c}" for c in bundle.universe})

    with pytest.raises(ValueError, match="columns"):
        rank_selection.rank_select_funds(df, cfg, bundle=bundle)

    mismatched_bundle = rank_selection.WindowMetricBundle(
        key=None,
        start=bundle.start,
        end=bundle.end,
        freq=bundle.freq,
        stats_cfg_hash="different",
        universe=tuple(df.columns),
        in_sample_df=df,
        _metrics=pd.DataFrame(index=df.columns, dtype=float),
    )

    with pytest.raises(ValueError, match="stats configuration"):
        rank_selection.rank_select_funds(df, cfg, bundle=mismatched_bundle)


def test_rank_select_funds_uses_cached_bundle():
    df = pd.DataFrame(
        {
            "Gamma Partners": [0.03, 0.02, 0.01],
            "Omega Partners": [0.01, 0.015, 0.02],
        }
    )
    cfg = rank_selection.RiskStatsConfig()
    window_key = ("2020-01", "2020-03", "hash", "M")
    bundle = rank_selection.WindowMetricBundle(
        key=window_key,
        start="2020-01",
        end="2020-03",
        freq="M",
        stats_cfg_hash=rank_selection._stats_cfg_hash(cfg),
        universe=tuple(df.columns),
        in_sample_df=df,
        _metrics=pd.DataFrame(index=df.columns, dtype=float),
    )
    rank_selection.store_window_metric_bundle(window_key, bundle)

    selected = rank_selection.rank_select_funds(
        df,
        cfg,
        window_key=window_key,
        inclusion_approach="top_n",
        n=1,
    )
    assert selected in ([bundle.universe[0]], [bundle.universe[1]])

    stats = rank_selection.selector_cache_stats()
    assert stats["selector_cache_hits"] >= 1

    new_key = ("2020-01", "2020-03", "other", "M")
    fresh_selection = rank_selection.rank_select_funds(
        df,
        cfg,
        window_key=new_key,
        inclusion_approach="top_n",
        n=1,
    )
    assert len(fresh_selection) == 1
    assert rank_selection.get_window_metric_bundle(new_key) is not None
