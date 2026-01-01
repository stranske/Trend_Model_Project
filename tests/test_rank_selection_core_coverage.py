from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from numpy._core import _methods

import trend_analysis.core.rank_selection as rs


@pytest.fixture(autouse=True)
def _safe_numpy_sum(monkeypatch: pytest.MonkeyPatch) -> None:
    def safe_umr_sum(
        a,
        axis=None,
        dtype=None,
        out=None,
        keepdims=False,
        initial=_methods._NoValue,
        where=True,
    ):
        if initial is _methods._NoValue:
            initial_value = 0
        else:
            initial_value = initial
        return np.add.reduce(
            a,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            where=where,
            initial=initial_value,
        )

    def safe_umr_prod(
        a,
        axis=None,
        dtype=None,
        out=None,
        keepdims=False,
        initial=_methods._NoValue,
        where=True,
    ):
        if initial is _methods._NoValue:
            initial_value = 1
        else:
            initial_value = initial
        return np.multiply.reduce(
            a,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            where=where,
            initial=initial_value,
        )

    monkeypatch.setattr(_methods, "umr_sum", safe_umr_sum)
    monkeypatch.setattr(_methods, "umr_prod", safe_umr_prod)


def _simple_frame(columns: list[str] | None = None) -> pd.DataFrame:
    data = {
        "Alpha Fund": [0.01, 0.02, 0.03],
        "BETA Growth": [0.02, 0.01, 0.0],
        "ABC Value": [0.0, 0.005, 0.01],
    }
    if columns is None:
        return pd.DataFrame(data)
    return pd.DataFrame({col: data[col] for col in columns})


def test_rank_selection_diagnostics_message_includes_threshold():
    diag = rs.RankSelectionDiagnostics(
        reason="Filtered",
        metric="AnnualReturn",
        transform="raw",
        inclusion_approach="threshold",
        total_candidates=3,
        non_null_scores=2,
        threshold=0.5,
    )

    message = diag.message()
    assert "threshold=0.5" in message
    assert "metric=AnnualReturn" in message


def test_window_metric_bundle_reports_metrics_and_hits_cache():
    rs.clear_window_metric_cache()
    frame = _simple_frame(["Alpha Fund", "BETA Growth"])
    cfg = rs.RiskStatsConfig()
    bundle = rs.WindowMetricBundle(
        key=None,
        start="2020-01",
        end="2020-02",
        freq="M",
        stats_cfg_hash=rs._stats_cfg_hash(cfg),
        universe=tuple(frame.columns),
        in_sample_df=frame,
        _metrics=pd.DataFrame(
            {"AnnualReturn": pd.Series([1.0, 2.0], index=frame.columns)}
        ),
    )

    assert bundle.available_metrics() == ["AnnualReturn"]
    cached = bundle.ensure_metric("AnnualReturn", cfg)
    assert cached.to_dict() == {"Alpha Fund": 1.0, "BETA Growth": 2.0}
    assert rs.selector_cache_hits >= 1


def test_window_metric_bundle_cov_metric_uses_payload():
    frame = _simple_frame(["Alpha Fund"])
    cfg = rs.RiskStatsConfig()
    bundle = rs.WindowMetricBundle(
        key=None,
        start="2020-01",
        end="2020-02",
        freq="M",
        stats_cfg_hash=rs._stats_cfg_hash(cfg),
        universe=tuple(frame.columns),
        in_sample_df=frame,
        _metrics=pd.DataFrame(index=frame.columns, dtype=float),
    )
    bundle.cov_payload = SimpleNamespace(cov=np.array([[1.0]]))

    avg_corr = bundle.ensure_metric("AvgCorr", cfg, enable_cov_cache=False)
    assert avg_corr.to_dict() == {"Alpha Fund": 0.0}
    assert "AvgCorr" in bundle._metrics.columns


def test_compute_covariance_payload_cache_and_no_cache(monkeypatch):
    frame = _simple_frame(["Alpha Fund", "BETA Growth"])
    cfg = rs.RiskStatsConfig()
    bundle = rs.WindowMetricBundle(
        key=None,
        start="2020-01",
        end="2020-02",
        freq="M",
        stats_cfg_hash=rs._stats_cfg_hash(cfg),
        universe=tuple(frame.columns),
        in_sample_df=frame,
        _metrics=pd.DataFrame(index=frame.columns, dtype=float),
    )
    payload = SimpleNamespace(cov=np.eye(2))
    calls: dict[str, object] = {}

    def fake_compute(_frame, materialise_aggregates=False):
        calls["materialise"] = materialise_aggregates
        return payload

    class DummyCache:
        def make_key(self, start, end, columns, freq):
            calls["key"] = (start, end, tuple(columns), freq)
            return "cache-key"

        def get_or_compute(self, key, fn):
            calls["cache_key"] = key
            return fn()

    monkeypatch.setattr(
        rs.importlib.import_module("trend_analysis.perf.cache"),
        "compute_cov_payload",
        fake_compute,
    )

    cache = DummyCache()
    cached = rs._compute_covariance_payload(
        bundle, cache, enable_cov_cache=True, incremental_cov=True
    )
    assert cached is payload
    assert calls["cache_key"] == "cache-key"
    assert calls["materialise"] is True

    calls.clear()
    uncached = rs._compute_covariance_payload(
        bundle, cache, enable_cov_cache=False, incremental_cov=False
    )
    assert uncached is payload
    assert "cache_key" not in calls


def test_compute_metric_series_with_cache_handles_cov_metrics(monkeypatch):
    frame = _simple_frame(["Alpha Fund"])
    cfg = rs.RiskStatsConfig()
    payload = SimpleNamespace(cov=np.array([[1.0]]))

    monkeypatch.setattr(
        rs.importlib.import_module("trend_analysis.perf.cache"),
        "compute_cov_payload",
        lambda _frame, materialise_aggregates=False: payload,
    )

    cov_var = rs.compute_metric_series_with_cache(
        frame,
        "__COV_VAR__",
        cfg,
        enable_cache=False,
    )
    assert cov_var.to_dict() == {"Alpha Fund": 1.0}

    avg_corr = rs.compute_metric_series_with_cache(
        frame,
        "AvgCorr",
        cfg,
        enable_cache=False,
    )
    assert avg_corr.to_dict() == {"Alpha Fund": 0.0}


def test_blended_score_normalizes_weights_and_inverts_ascending(monkeypatch):
    frame = _simple_frame(["Alpha Fund", "BETA Growth"])
    cfg = rs.RiskStatsConfig()

    def fake_metric(_frame, metric_name, _cfg, *, risk_free_override=None):
        if metric_name == "AnnualReturn":
            return pd.Series([0.1, 0.2], index=_frame.columns)
        if metric_name == "MaxDrawdown":
            return pd.Series([0.3, 0.1], index=_frame.columns)
        raise AssertionError("Unexpected metric")

    monkeypatch.setattr(rs, "_call_metric_series", fake_metric)
    monkeypatch.setattr(rs, "_zscore", lambda series: series.astype(float))

    combo = rs.blended_score(
        frame,
        {"annual_return": 2.0, "MaxDrawdown": 1.0},
        cfg,
    )
    assert list(combo.index) == ["Alpha Fund", "BETA Growth"]
    assert combo["BETA Growth"] > combo["Alpha Fund"]

    with pytest.raises(ValueError, match="non-empty weights"):
        rs.blended_score(frame, {}, cfg)

    with pytest.raises(ValueError, match="Sum of weights must not be zero"):
        rs.blended_score(frame, {"AnnualReturn": 0.0}, cfg)


def test_rank_select_funds_branches_and_cache_side_effects(monkeypatch):
    frame = _simple_frame()
    cfg = rs.RiskStatsConfig()
    scores = pd.Series([0.2, 0.1, -0.1], index=frame.columns)

    monkeypatch.setattr(
        rs.WindowMetricBundle,
        "ensure_metric",
        lambda _self, *_args, **_kwargs: scores,
    )

    rs.clear_window_metric_cache()
    selection = rs.rank_select_funds(
        frame,
        cfg,
        inclusion_approach="top_pct",
        pct=0.5,
        score_by="annual_return",
        transform_mode="rank",
    )
    assert selection[0] == "Alpha Fund"

    selection = rs.rank_select_funds(
        frame,
        cfg,
        inclusion_approach="threshold",
        threshold=0.15,
        score_by="MaxDrawdown",
    )
    assert selection == ["ABC Value", "BETA Growth"]

    window_key = rs.make_window_key("2020-01", "2020-02", frame.columns, cfg)
    rs.rank_select_funds(
        frame,
        cfg,
        inclusion_approach="top_n",
        n=1,
        score_by="annual_return",
        window_key=window_key,
        risk_free=0.01,
    )
    assert rs.get_window_metric_bundle(window_key) is None


def test_rank_select_funds_reports_empty_scores(monkeypatch):
    frame = _simple_frame()
    cfg = rs.RiskStatsConfig()
    empty_scores = pd.Series([np.nan, np.nan, np.nan], index=frame.columns)
    monkeypatch.setattr(
        rs.WindowMetricBundle,
        "ensure_metric",
        lambda _self, *_args, **_kwargs: empty_scores,
    )

    with pytest.warns(RuntimeWarning):
        selected, diagnostics = rs.rank_select_funds(
            frame,
            cfg,
            inclusion_approach="top_n",
            n=1,
            return_diagnostics=True,
        )

    assert selected == []
    assert diagnostics is not None
    assert (
        diagnostics.reason
        == "No candidate scores available after filtering and transform"
    )


def test_rank_select_funds_rejects_bad_inputs():
    frame = _simple_frame()
    cfg = rs.RiskStatsConfig()
    bundle = rs.WindowMetricBundle(
        key=None,
        start="2020-01",
        end="2020-02",
        freq="M",
        stats_cfg_hash="bad-hash",
        universe=("Other",),
        in_sample_df=frame,
        _metrics=pd.DataFrame(index=frame.columns, dtype=float),
    )

    with pytest.raises(ValueError, match="bundle does not match DataFrame columns"):
        rs.rank_select_funds(
            frame,
            cfg,
            inclusion_approach="top_n",
            n=1,
            bundle=bundle,
        )

    with pytest.raises(ValueError, match="stats configuration"):
        bundle.universe = tuple(frame.columns)
        rs.rank_select_funds(
            frame,
            cfg,
            inclusion_approach="top_n",
            n=1,
            bundle=bundle,
        )

    with pytest.raises(ValueError, match="Unknown inclusion_approach"):
        rs.rank_select_funds(
            frame,
            cfg,
            inclusion_approach="bogus",
            n=1,
            bundle=rs.WindowMetricBundle(
                key=None,
                start="2020-01",
                end="2020-02",
                freq="M",
                stats_cfg_hash=rs._stats_cfg_hash(cfg),
                universe=tuple(frame.columns),
                in_sample_df=frame,
                _metrics=pd.DataFrame(
                    {"AnnualReturn": pd.Series([0.1, 0.2, 0.3], index=frame.columns)}
                ),
            ),
        )

    with pytest.raises(ValueError, match="blended score requires blended_weights"):
        rs.rank_select_funds(
            frame,
            cfg,
            inclusion_approach="top_n",
            n=1,
            score_by="blended",
        )


def test_build_ui_loads_helpers(monkeypatch):
    class DummyModule:
        @staticmethod
        def build_ui():
            return "ui-ready"

    monkeypatch.setattr(rs.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(rs.importlib, "import_module", lambda _name: DummyModule())

    assert rs.build_ui() == "ui-ready"


def test_metrics_frame_empty_and_as_frame():
    frame = _simple_frame(["Alpha Fund"])
    cfg = rs.RiskStatsConfig()
    bundle = rs.WindowMetricBundle(
        key=None,
        start="2020-01",
        end="2020-02",
        freq="M",
        stats_cfg_hash=rs._stats_cfg_hash(cfg),
        universe=tuple(frame.columns),
        in_sample_df=frame,
        _metrics=pd.DataFrame(),
    )

    metrics = bundle.metrics_frame()
    assert metrics.empty
    assert list(metrics.index) == ["Alpha Fund"]
    assert bundle.as_frame().equals(metrics)


def test_window_metric_cache_scoped_clear():
    rs.clear_window_metric_cache()
    frame = _simple_frame(["Alpha Fund"])
    cfg = rs.RiskStatsConfig()
    bundle = rs.WindowMetricBundle(
        key=None,
        start="2020-01",
        end="2020-02",
        freq="M",
        stats_cfg_hash=rs._stats_cfg_hash(cfg),
        universe=tuple(frame.columns),
        in_sample_df=frame,
        _metrics=pd.DataFrame(index=frame.columns, dtype=float),
    )
    key = ("2020-01", "2020-02", "u", "h")

    with rs.selector_cache_scope("scoped"):
        rs.store_window_metric_bundle(key, bundle)
        assert rs.get_window_metric_bundle(key) is bundle

    rs.clear_window_metric_cache("scoped")
    with rs.selector_cache_scope("scoped"):
        assert rs.get_window_metric_bundle(key) is None


def test_cov_metric_from_payload_multiple_columns():
    payload = SimpleNamespace(cov=np.array([[1.0, 0.5], [0.5, 1.0]]))
    avg_corr = rs._cov_metric_from_payload("AvgCorr", payload, ["A", "B"])
    assert avg_corr.to_dict() == {"A": 0.5, "B": 0.5}


def test_apply_transform_variants_and_errors():
    series = pd.Series([1.0, 2.0, 3.0], index=["A", "B", "C"])
    ranked = rs._apply_transform(series, mode="rank")
    assert ranked["C"] == 1

    percentile = rs._apply_transform(series, mode="percentile", rank_pct=0.5)
    assert percentile.notna().sum() >= 1

    zscored = rs._apply_transform(series, mode="zscore", window=2)
    assert zscored.index.tolist() == ["A", "B", "C"]

    with pytest.raises(ValueError, match="rank_pct must be set"):
        rs._apply_transform(series, mode="percentile")

    with pytest.raises(ValueError, match="unknown transform mode"):
        rs._apply_transform(series, mode="unknown")


def test_rank_select_empty_universe_no_diagnostics():
    empty = pd.DataFrame()
    cfg = rs.RiskStatsConfig()
    assert rs.rank_select_funds(empty, cfg, inclusion_approach="top_n", n=1) == []


def test_rank_select_cached_bundle_mismatch(monkeypatch):
    frame = _simple_frame(["Alpha Fund", "BETA Growth"])
    cfg = rs.RiskStatsConfig()
    window_key = rs.make_window_key("2020-01", "2020-02", frame.columns, cfg)
    cached = rs.WindowMetricBundle(
        key=window_key,
        start="2020-01",
        end="2020-02",
        freq="M",
        stats_cfg_hash=rs._stats_cfg_hash(cfg),
        universe=("Other",),
        in_sample_df=frame,
        _metrics=pd.DataFrame(index=frame.columns, dtype=float),
    )
    rs.store_window_metric_bundle(window_key, cached)

    scores = pd.Series([0.2, 0.1], index=frame.columns)
    monkeypatch.setattr(
        rs.WindowMetricBundle,
        "ensure_metric",
        lambda _self, *_args, **_kwargs: scores,
    )

    selection = rs.rank_select_funds(
        frame,
        cfg,
        inclusion_approach="top_n",
        n=1,
        window_key=window_key,
    )
    assert selection == ["Alpha Fund"]


def test_rank_select_cache_hit_updates_freq():
    frame = _simple_frame(["Alpha Fund", "BETA Growth"])
    cfg = rs.RiskStatsConfig()
    window_key = rs.make_window_key("2020-01", "2020-02", frame.columns, cfg)
    cached = rs.WindowMetricBundle(
        key=window_key,
        start="2020-01",
        end="2020-02",
        freq="M",
        stats_cfg_hash=rs._stats_cfg_hash(cfg),
        universe=tuple(frame.columns),
        in_sample_df=frame,
        _metrics=pd.DataFrame(
            {"AnnualReturn": pd.Series([0.2, 0.1], index=frame.columns)}
        ),
    )
    rs.store_window_metric_bundle(window_key, cached)

    rs.rank_select_funds(
        frame,
        cfg,
        inclusion_approach="top_n",
        n=1,
        window_key=window_key,
        freq="Q",
    )
    assert cached.freq == "Q"


def test_rank_select_inclusion_errors():
    frame = _simple_frame(["Alpha Fund", "BETA Growth"])
    cfg = rs.RiskStatsConfig()
    bundle = rs.WindowMetricBundle(
        key=None,
        start="2020-01",
        end="2020-02",
        freq="M",
        stats_cfg_hash=rs._stats_cfg_hash(cfg),
        universe=tuple(frame.columns),
        in_sample_df=frame,
        _metrics=pd.DataFrame(
            {"AnnualReturn": pd.Series([0.2, 0.1], index=frame.columns)}
        ),
    )

    with pytest.raises(ValueError, match="top_n requires parameter n"):
        rs.rank_select_funds(frame, cfg, inclusion_approach="top_n", bundle=bundle)

    with pytest.raises(ValueError, match="top_pct requires 0 < pct <= 1"):
        rs.rank_select_funds(
            frame,
            cfg,
            inclusion_approach="top_pct",
            pct=1.5,
            bundle=bundle,
        )

    with pytest.raises(
        ValueError, match="threshold approach requires parameter threshold"
    ):
        rs.rank_select_funds(
            frame,
            cfg,
            inclusion_approach="threshold",
            bundle=bundle,
        )


def test_rank_select_threshold_filtered_out_no_diagnostics(monkeypatch):
    frame = _simple_frame(["Alpha Fund", "BETA Growth"])
    cfg = rs.RiskStatsConfig()
    scores = pd.Series([0.2, 0.1], index=frame.columns)
    monkeypatch.setattr(
        rs.WindowMetricBundle,
        "ensure_metric",
        lambda _self, *_args, **_kwargs: scores,
    )

    selection = rs.rank_select_funds(
        frame,
        cfg,
        inclusion_approach="threshold",
        threshold=0.05,
        score_by="MaxDrawdown",
    )
    assert selection == []


def test_get_metric_context_requires_context():
    with pytest.raises(RuntimeError, match="Metric context is unavailable"):
        rs._get_metric_context()


def test_avg_corr_metric_uses_context_and_cache():
    frame = pd.DataFrame({"A": [0.01, 0.02], "B": [0.02, 0.03]})
    token = rs._METRIC_CONTEXT.set({"frame": frame})
    try:
        result = rs._avg_corr_metric(pd.Series([0.01, 0.02], name="A"))
        assert result == pytest.approx(1.0)
        missing = rs._avg_corr_metric(pd.Series([0.0, 0.0], name="C"))
        assert missing == 0.0
    finally:
        rs._METRIC_CONTEXT.reset(token)


def test_compute_metric_series_requires_registered_metric():
    frame = _simple_frame(["Alpha Fund", "BETA Growth"])
    cfg = rs.RiskStatsConfig()
    with pytest.raises(ValueError, match="not registered"):
        rs._compute_metric_series(frame, "NotRegistered", cfg)


def test_ensure_cov_payload_sets_bundle(monkeypatch):
    frame = _simple_frame(["Alpha Fund"])
    cfg = rs.RiskStatsConfig()
    bundle = rs.WindowMetricBundle(
        key=None,
        start="2020-01",
        end="2020-02",
        freq="M",
        stats_cfg_hash=rs._stats_cfg_hash(cfg),
        universe=tuple(frame.columns),
        in_sample_df=frame,
        _metrics=pd.DataFrame(index=frame.columns, dtype=float),
    )
    payload = SimpleNamespace(cov=np.array([[1.0]]))
    monkeypatch.setattr(
        rs.importlib.import_module("trend_analysis.perf.cache"),
        "compute_cov_payload",
        lambda _frame: payload,
    )

    result = rs._ensure_cov_payload(frame, bundle)
    assert result is payload
    assert bundle.cov_payload is payload


def test_metric_from_cov_payload_multiple_columns():
    frame = _simple_frame(["Alpha Fund", "BETA Growth"])
    payload = SimpleNamespace(cov=np.array([[1.0, 0.5], [0.5, 1.0]]))
    avg_corr = rs._metric_from_cov_payload("AvgCorr", frame, payload)
    assert avg_corr.to_dict() == {"Alpha Fund": 0.5, "BETA Growth": 0.5}


def test_compute_metric_series_with_cache_non_cov_metric(monkeypatch):
    frame = _simple_frame(["Alpha Fund", "BETA Growth"])
    cfg = rs.RiskStatsConfig()
    expected = pd.Series([0.1, 0.2], index=frame.columns)
    monkeypatch.setattr(rs, "_call_metric_series", lambda *_args, **_kwargs: expected)

    result = rs.compute_metric_series_with_cache(frame, "AnnualReturn", cfg)
    assert result.equals(expected)


def test_zscore_handles_zero_and_nonzero_sigma():
    constant = pd.Series([1.0, 1.0], index=["A", "B"])
    assert rs._zscore(constant).to_list() == [0.0, 0.0]

    varied = pd.Series([1.0, 2.0, 3.0], index=["A", "B", "C"])
    zscores = rs._zscore(varied)
    assert zscores.isna().sum() == 0


def test_blended_score_uses_bundle_and_merges_weights(monkeypatch):
    frame = _simple_frame(["Alpha Fund", "BETA Growth"])
    cfg = rs.RiskStatsConfig()
    bundle = rs.WindowMetricBundle(
        key=None,
        start="2020-01",
        end="2020-02",
        freq="M",
        stats_cfg_hash=rs._stats_cfg_hash(cfg),
        universe=tuple(frame.columns),
        in_sample_df=frame,
        _metrics=pd.DataFrame(
            {
                "AnnualReturn": pd.Series([0.1, 0.2], index=frame.columns),
                "MaxDrawdown": pd.Series([0.3, 0.1], index=frame.columns),
            }
        ),
    )
    monkeypatch.setattr(rs, "_zscore", lambda series: series.astype(float))

    combo = rs.blended_score(
        frame,
        {"annual_return": 1.0, "AnnualReturn": 2.0, "MaxDrawdown": 1.0},
        cfg,
        bundle=bundle,
    )
    assert combo.index.tolist() == ["Alpha Fund", "BETA Growth"]


def test_metrics_frame_returns_copy_when_populated():
    frame = _simple_frame(["Alpha Fund"])
    cfg = rs.RiskStatsConfig()
    bundle = rs.WindowMetricBundle(
        key=None,
        start="2020-01",
        end="2020-02",
        freq="M",
        stats_cfg_hash=rs._stats_cfg_hash(cfg),
        universe=tuple(frame.columns),
        in_sample_df=frame,
        _metrics=pd.DataFrame({"AnnualReturn": pd.Series([0.1], index=frame.columns)}),
    )
    metrics = bundle.metrics_frame()
    metrics.iloc[0, 0] = 9.9
    assert bundle._metrics.iloc[0, 0] == 0.1


def test_window_metric_bundle_populates_cov_payload(monkeypatch):
    frame = _simple_frame(["Alpha Fund", "BETA Growth"])
    cfg = rs.RiskStatsConfig()
    bundle = rs.WindowMetricBundle(
        key=None,
        start="2020-01",
        end="2020-02",
        freq="M",
        stats_cfg_hash=rs._stats_cfg_hash(cfg),
        universe=tuple(frame.columns),
        in_sample_df=frame,
        _metrics=pd.DataFrame(index=frame.columns, dtype=float),
    )
    payload = SimpleNamespace(cov=np.array([[1.0, 0.0], [0.0, 1.0]]))

    monkeypatch.setattr(
        rs, "_compute_covariance_payload", lambda *_args, **_kwargs: payload
    )
    avg_corr = bundle.ensure_metric("AvgCorr", cfg)
    assert avg_corr.to_dict() == {"Alpha Fund": 0.0, "BETA Growth": 0.0}
    assert bundle.cov_payload is payload


def test_window_metric_bundle_uses_metric_cache(monkeypatch):
    frame = _simple_frame(["Alpha Fund"])
    cfg = rs.RiskStatsConfig()
    cfg.enable_metric_cache = True
    bundle = rs.WindowMetricBundle(
        key=None,
        start="2020-01",
        end="2020-02",
        freq="M",
        stats_cfg_hash=rs._stats_cfg_hash(cfg),
        universe=tuple(frame.columns),
        in_sample_df=frame,
        _metrics=pd.DataFrame(index=frame.columns, dtype=float),
    )
    called: dict[str, bool] = {"hit": False}

    def fake_cached(**kwargs):
        called["hit"] = True
        return pd.Series([0.2], index=frame.columns, name="AnnualReturn")

    monkeypatch.setattr(
        rs.importlib.import_module("trend_analysis.core.metric_cache"),
        "get_or_compute_metric_series",
        lambda **kwargs: fake_cached(**kwargs),
    )

    series = bundle.ensure_metric("AnnualReturn", cfg)
    assert called["hit"] is True
    assert series.to_dict() == {"Alpha Fund": 0.2}


def test_cov_metric_from_payload_supports_cov_var():
    payload = SimpleNamespace(cov=np.array([[2.0, 0.0], [0.0, 3.0]]))
    cov_var = rs._cov_metric_from_payload("__COV_VAR__", payload, ["A", "B"])
    assert cov_var.to_dict() == {"A": 2.0, "B": 3.0}


def test_reset_selector_cache_alias():
    rs.reset_selector_cache()
    stats = rs.selector_cache_stats()
    assert "entries" in stats


def test_apply_transform_zscore_window_and_zero_sigma():
    series = pd.Series([1.0, 1.0, 1.0], index=["A", "B", "C"])
    zscores = rs._apply_transform(series, mode="zscore", window=10)
    assert zscores.to_list() == [0.0, 0.0, 0.0]


def test_rank_select_empty_universe_with_diagnostics():
    empty = pd.DataFrame()
    cfg = rs.RiskStatsConfig()
    selection, diagnostics = rs.rank_select_funds(
        empty,
        cfg,
        inclusion_approach="top_n",
        n=1,
        return_diagnostics=True,
    )
    assert selection == []
    assert diagnostics is not None


def test_rank_select_blended_path_with_bundle(monkeypatch):
    frame = _simple_frame(["Alpha Fund", "BETA Growth"])
    cfg = rs.RiskStatsConfig()
    bundle = rs.WindowMetricBundle(
        key=None,
        start="2020-01",
        end="2020-02",
        freq="M",
        stats_cfg_hash=rs._stats_cfg_hash(cfg),
        universe=tuple(frame.columns),
        in_sample_df=frame,
        _metrics=pd.DataFrame(index=frame.columns, dtype=float),
    )
    monkeypatch.setattr(
        rs,
        "blended_score",
        lambda *_args, **_kwargs: pd.Series([1.0, 0.5], index=frame.columns),
    )

    selection = rs.rank_select_funds(
        frame,
        cfg,
        inclusion_approach="top_n",
        n=1,
        score_by="blended",
        blended_weights={"AnnualReturn": 1.0},
        bundle=bundle,
    )
    assert selection == ["Alpha Fund"]


def test_rank_select_returns_diagnostics_on_threshold(monkeypatch):
    frame = _simple_frame(["Alpha Fund", "BETA Growth"])
    cfg = rs.RiskStatsConfig()
    scores = pd.Series([0.2, 0.1], index=frame.columns)
    monkeypatch.setattr(
        rs.WindowMetricBundle,
        "ensure_metric",
        lambda _self, *_args, **_kwargs: scores,
    )

    selection, diagnostics = rs.rank_select_funds(
        frame,
        cfg,
        inclusion_approach="threshold",
        threshold=0.05,
        score_by="MaxDrawdown",
        return_diagnostics=True,
    )
    assert selection == []
    assert diagnostics is not None


def test_rank_select_dedupe_handles_edge_names(monkeypatch):
    frame = pd.DataFrame(
        {
            "!!!": [0.1, 0.2],
            "ABC Growth": [0.2, 0.1],
            "Alpha Beta": [0.05, 0.04],
        }
    )
    cfg = rs.RiskStatsConfig()
    scores = pd.Series([0.3, 0.2, 0.1], index=frame.columns)
    monkeypatch.setattr(
        rs.WindowMetricBundle,
        "ensure_metric",
        lambda _self, *_args, **_kwargs: scores,
    )

    selection = rs.rank_select_funds(
        frame,
        cfg,
        inclusion_approach="top_n",
        n=2,
    )
    assert len(selection) == 2


def test_canonical_metric_list_maps_aliases():
    metrics = rs.canonical_metric_list(["annual_return", "MaxDrawdown"])
    assert metrics == ["AnnualReturn", "MaxDrawdown"]


def test_avg_corr_metric_shortcuts_for_empty_context():
    token = rs._METRIC_CONTEXT.set({"frame": pd.DataFrame()})
    try:
        result = rs._avg_corr_metric(pd.Series([0.0, 0.0], name="A"))
        assert result == 0.0
    finally:
        rs._METRIC_CONTEXT.reset(token)


def test_avg_corr_metric_single_column_shortcut():
    frame = pd.DataFrame({"A": [0.01, 0.02]})
    token = rs._METRIC_CONTEXT.set({"frame": frame})
    try:
        result = rs._avg_corr_metric(pd.Series([0.01, 0.02], name="A"))
        assert result == 0.0
    finally:
        rs._METRIC_CONTEXT.reset(token)


def test_compute_metric_series_runs_registered_metric():
    frame = _simple_frame(["Alpha Fund", "BETA Growth"])
    cfg = rs.RiskStatsConfig(risk_free=0.0)
    series = rs._compute_metric_series(frame, "AnnualReturn", cfg)
    assert series.index.tolist() == ["Alpha Fund", "BETA Growth"]


def test_metric_fn_accepts_risk_free_override_flag():
    def with_override(*_args, risk_free_override=None, **_kwargs):
        return risk_free_override is not None

    def no_override(*_args, **_kwargs):
        return False

    assert rs._metric_fn_accepts_risk_free_override(with_override) is True
    assert rs._metric_fn_accepts_risk_free_override(no_override) is False


def test_call_metric_series_uses_override_when_supported(monkeypatch):
    frame = _simple_frame(["Alpha Fund"])
    cfg = rs.RiskStatsConfig()

    def with_override(in_sample_df, metric_name, stats_cfg, *, risk_free_override=None):
        return pd.Series([risk_free_override], index=in_sample_df.columns)

    monkeypatch.setattr(rs, "_compute_metric_series", with_override)
    result = rs._call_metric_series(frame, "AnnualReturn", cfg, risk_free_override=0.5)
    assert result.iloc[0] == 0.5


def test_call_metric_series_skips_override_when_unsupported(monkeypatch):
    frame = _simple_frame(["Alpha Fund"])
    cfg = rs.RiskStatsConfig()

    def no_override(in_sample_df, metric_name, stats_cfg):
        return pd.Series([0.25], index=in_sample_df.columns)

    monkeypatch.setattr(rs, "_compute_metric_series", no_override)
    result = rs._call_metric_series(frame, "AnnualReturn", cfg, risk_free_override=0.5)
    assert result.iloc[0] == 0.25


def test_ensure_cov_payload_returns_existing():
    frame = _simple_frame(["Alpha Fund"])
    cfg = rs.RiskStatsConfig()
    payload = SimpleNamespace(cov=np.array([[1.0]]))
    bundle = rs.WindowMetricBundle(
        key=None,
        start="2020-01",
        end="2020-02",
        freq="M",
        stats_cfg_hash=rs._stats_cfg_hash(cfg),
        universe=tuple(frame.columns),
        in_sample_df=frame,
        _metrics=pd.DataFrame(index=frame.columns, dtype=float),
        cov_payload=payload,
    )
    assert rs._ensure_cov_payload(frame, bundle) is payload


def test_metric_from_cov_payload_cov_var_and_singleton():
    frame = _simple_frame(["Alpha Fund"])
    payload = SimpleNamespace(cov=np.array([[1.0]]))
    cov_var = rs._metric_from_cov_payload("__COV_VAR__", frame, payload)
    avg_corr = rs._metric_from_cov_payload("AvgCorr", frame, payload)
    assert cov_var.to_dict() == {"Alpha Fund": 1.0}
    assert avg_corr.to_dict() == {"Alpha Fund": 0.0}


def test_compute_metric_series_with_cache_avg_corr_multi(monkeypatch):
    frame = _simple_frame(["Alpha Fund", "BETA Growth"])
    cfg = rs.RiskStatsConfig()
    payload = SimpleNamespace(cov=np.array([[1.0, 0.5], [0.5, 1.0]]))
    monkeypatch.setattr(
        rs.importlib.import_module("trend_analysis.perf.cache"),
        "compute_cov_payload",
        lambda _frame: payload,
    )
    avg_corr = rs.compute_metric_series_with_cache(
        frame,
        "AvgCorr",
        cfg,
        enable_cache=False,
    )
    assert avg_corr.to_dict() == {"Alpha Fund": 0.5, "BETA Growth": 0.5}


def test_build_ui_requires_ipywidgets(monkeypatch):
    monkeypatch.setattr(rs.importlib.util, "find_spec", lambda _name: None)
    with pytest.raises(ImportError, match="ipywidgets is required"):
        rs.build_ui()
