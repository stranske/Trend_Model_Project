import numpy as np
import pandas as pd
import pytest

import trend_analysis.core.rank_selection as rs


def test_canonicalise_labels_handles_duplicates_and_blanks():
    labels = [" A", "A", " ", "B", "A "]
    canonical = rs._canonicalise_labels(labels)
    assert canonical == ["A", "A_2", "Unnamed_3", "B", "A_3"]


def test_ensure_canonical_columns_returns_same_frame_when_clean():
    frame = pd.DataFrame({"A": [1], "B": [2]})
    result = rs._ensure_canonical_columns(frame)
    assert result is frame
    assert list(result.columns) == ["A", "B"]


def test_ensure_canonical_columns_normalizes_and_copies_when_needed():
    frame = pd.DataFrame({" A": [1], "A ": [2]})
    result = rs._ensure_canonical_columns(frame)
    assert result is not frame
    assert list(result.columns) == ["A", "A_2"]


def test_json_default_handles_supported_types_and_errors():
    assert set(rs._json_default({1, 2})) == {1, 2}
    assert rs._json_default(("x", "y")) == ["x", "y"]
    assert rs._json_default(np.array([1, 2])) == [1, 2]
    assert rs._json_default(np.float64(1.5)) == pytest.approx(1.5)

    with pytest.raises(TypeError):
        rs._json_default({"unsupported": object()})


def test_make_window_key_canonicalizes_universe_and_stats_cfg():
    cfg = rs.RiskStatsConfig(metrics_to_run=["Sharpe"], risk_free=0.01)
    key_one = rs.make_window_key("2020-01", "2020-12", [" A", "B"], cfg)
    key_two = rs.make_window_key("2020-01", "2020-12", ["B", "A"], cfg)
    assert key_one == key_two


def test_hash_universe_is_order_insensitive_and_stable():
    baseline = rs._hash_universe(["b", "a", "c"])
    assert baseline == rs._hash_universe(["c", "b", "a"])

    changed = rs._hash_universe(["a", "c", "d"])
    assert baseline != changed


def test_stats_cfg_hash_tracks_extra_attributes():
    cfg = rs.RiskStatsConfig(metrics_to_run=["Sharpe"], risk_free=0.01)
    cfg.extra_field = {"alpha": 1}
    baseline = rs._stats_cfg_hash(cfg)

    cfg.extra_field["alpha"] = 2
    mutated = rs._stats_cfg_hash(cfg)
    assert baseline != mutated

    mirror = rs.RiskStatsConfig(metrics_to_run=["Sharpe"], risk_free=0.01)
    mirror.extra_field = {"alpha": 2}
    assert mutated == rs._stats_cfg_hash(mirror)


def test_window_metric_cache_scopes_and_limits_are_respected():
    rs.clear_window_metric_cache()
    frame = pd.DataFrame({"A": [1.0], "B": [2.0]})
    bundle = rs.WindowMetricBundle(
        key=None,
        start="2020-01",
        end="2020-02",
        freq="M",
        stats_cfg_hash="hash",
        universe=tuple(frame.columns),
        in_sample_df=frame,
        _metrics=pd.DataFrame(index=frame.columns),
    )
    key = ("2020-01", "2020-02", "u", "h")

    rs.store_window_metric_bundle(key, bundle)
    assert rs.get_window_metric_bundle(key) is bundle

    previous = rs.set_window_metric_cache_limit(0)
    assert rs.get_window_metric_bundle(key) is None
    rs.set_window_metric_cache_limit(previous)

    rs.clear_window_metric_cache()
    with rs.selector_cache_scope("alt"):
        rs.store_window_metric_bundle(key, bundle)
        assert rs.get_window_metric_bundle(key) is bundle

    assert rs.get_window_metric_bundle(key) is None
    stats = rs.selector_cache_stats()
    assert stats["entries"] == 0
