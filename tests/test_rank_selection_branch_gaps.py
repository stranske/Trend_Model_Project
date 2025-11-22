"""Targeted branch coverage for ``trend_analysis.core.rank_selection``.

These tests exercise edge paths that were previously unvisited by the
existing suite (error branches, rare helper logic, and alternate call
patterns).  The goal is to tighten coverage around configuration
validation and helper utilities without disturbing the broader
behavioural tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_analysis.core import rank_selection
from trend_analysis.core.rank_selection import (
    RiskStatsConfig,
    WindowMetricBundle,
    clear_window_metric_cache,
    default_quality_config,
    get_window_metric_bundle,
)
from trend_analysis.perf.cache import CovPayload


@pytest.fixture(autouse=True)
def _reset_cache_state() -> None:
    """Ensure selector caches do not leak between tests."""

    clear_window_metric_cache()
    yield
    clear_window_metric_cache()


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Alpha": [0.01, 0.02, 0.03],
            "Beta": [0.015, 0.012, 0.011],
        }
    )


def test_apply_transform_percentile_requires_rank_pct() -> None:
    series = pd.Series([0.1, 0.2, 0.3], index=["a", "b", "c"])
    with pytest.raises(ValueError, match="rank_pct must be set"):
        rank_selection._apply_transform(series, mode="percentile")


def test_rank_select_funds_does_not_store_bundle_when_disabled() -> None:
    df = _sample_frame()
    cfg = RiskStatsConfig()
    window_key = ("2020-01", "2020-02", "hash", "stats")

    rank_selection.rank_select_funds(
        df,
        cfg,
        inclusion_approach="top_n",
        n=1,
        window_key=window_key,
        store_bundle=False,
    )

    assert get_window_metric_bundle(window_key) is None


def test_rank_select_funds_updates_existing_bundle_frequency() -> None:
    df = _sample_frame()
    cfg = RiskStatsConfig()
    bundle = WindowMetricBundle(
        key=None,
        start="2020-01",
        end="2020-03",
        freq="ME",
        stats_cfg_hash=rank_selection._stats_cfg_hash(cfg),
        universe=tuple(df.columns),
        in_sample_df=df.copy(),
        _metrics=pd.DataFrame(index=df.columns, dtype=float),
    )

    rank_selection.rank_select_funds(
        df,
        cfg,
        inclusion_approach="top_n",
        n=1,
        bundle=bundle,
        freq="Q",
    )

    assert bundle.freq == "Q"


def test_rank_select_funds_blended_requires_weights() -> None:
    df = _sample_frame()
    cfg = RiskStatsConfig()

    with pytest.raises(ValueError, match="blended score requires blended_weights"):
        rank_selection.rank_select_funds(
            df,
            cfg,
            inclusion_approach="top_n",
            n=1,
            score_by="blended",
        )


def test_rank_select_funds_uses_compute_metric_series_when_no_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _sample_frame()
    cfg = RiskStatsConfig()
    called: dict[str, object] = {}

    def _fake_compute(
        frame: pd.DataFrame, metric: str, cfg_arg: RiskStatsConfig
    ) -> pd.Series:
        called["args"] = (frame, metric, cfg_arg)
        return pd.Series([1.0, 0.5], index=frame.columns)

    monkeypatch.setattr(rank_selection, "_compute_metric_series", _fake_compute)

    selected = rank_selection.rank_select_funds(
        df,
        cfg,
        inclusion_approach="top_n",
        n=1,
        score_by="AnnualReturn",
        window_key=None,
    )

    assert called["args"][1] == "AnnualReturn"
    assert selected == ["Alpha"]


def test_rank_select_funds_firm_key_branches_and_backfill() -> None:
    df = pd.DataFrame(
        {
            "Gamma Capital A": [0.30, 0.31, 0.32],
            "Gamma Capital B": [0.29, 0.30, 0.31],
            "AAA Fund": [0.25, 0.24, 0.26],
            "1234": [0.20, 0.19, 0.18],
        }
    )
    cfg = RiskStatsConfig()

    selected = rank_selection.rank_select_funds(
        df,
        cfg,
        inclusion_approach="top_n",
        n=4,
    )

    # First three entries cover unique firm keys, the fourth is a backfill of a
    # duplicate firm once the unique pool is exhausted.
    assert selected == ["Gamma Capital A", "AAA Fund", "1234", "Gamma Capital B"]


@pytest.mark.parametrize(
    "approach, kwargs, message",
    [
        ("top_n", {}, "top_n requires parameter n"),
        ("top_pct", {"pct": 1.5}, "top_pct requires 0 < pct <= 1"),
        ("threshold", {}, "threshold approach requires parameter threshold"),
    ],
)
def test_rank_select_funds_parameter_validation(
    approach: str, kwargs: dict[str, object], message: str
) -> None:
    df = _sample_frame()
    cfg = RiskStatsConfig()

    with pytest.raises(ValueError, match=message):
        rank_selection.rank_select_funds(
            df,
            cfg,
            inclusion_approach=approach,
            score_by="AnnualReturn",
            **kwargs,
        )


def test_canonical_metric_list_alias_resolution() -> None:
    all_metrics = rank_selection.canonical_metric_list()
    assert all_metrics, "registry should not be empty"

    canonical = rank_selection.canonical_metric_list(["annual_return", "Custom"])
    assert canonical[0] == "AnnualReturn"
    assert canonical[1] == "Custom"


def test_quality_filters_cover_ratio_and_limits() -> None:
    dates = pd.date_range("2022-01-31", periods=3, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "A": [0.1, np.nan, 0.2],  # missing count == 1 (passes first check)
            "B": [
                np.nan,
                0.0,
                np.nan,
            ],  # ratio of missing values triggers max_missing_ratio
            "C": [0.0, 5.0, 0.0],  # implausible value triggers limit check
            "D": [0.0, 0.1, 0.2],
        }
    )
    cfg = default_quality_config(
        max_missing_months=2,
        max_missing_ratio=0.3,
        implausible_value_limit=1.0,
    )

    eligible = rank_selection.quality_filter(df, cfg)
    assert eligible == ["D"]

    # The internal _quality_filter should mirror the behaviour over the window slice.
    window = rank_selection._quality_filter(
        df,
        ["A", "B", "C", "D"],
        in_sdate="2022-01",
        out_edate="2022-03",
        cfg=cfg,
    )
    assert window == ["D"]


def test_avg_corr_metric_single_column_returns_zero() -> None:
    frame = pd.DataFrame({"Only": [0.01, 0.02, 0.03]})
    token = rank_selection._METRIC_CONTEXT.set({"frame": frame})
    try:
        result = rank_selection._avg_corr_metric(frame["Only"])
        assert result == 0.0
    finally:
        rank_selection._METRIC_CONTEXT.reset(token)


def test_compute_metric_series_unknown_metric_raises() -> None:
    df = _sample_frame()
    cfg = RiskStatsConfig()
    with pytest.raises(ValueError, match="not registered"):
        rank_selection._compute_metric_series(df, "DoesNotExist", cfg)


def test_ensure_cov_payload_attaches_to_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _sample_frame()
    cfg = RiskStatsConfig()
    bundle = WindowMetricBundle(
        key=None,
        start="2020-01",
        end="2020-03",
        freq="ME",
        stats_cfg_hash=rank_selection._stats_cfg_hash(cfg),
        universe=tuple(df.columns),
        in_sample_df=df.copy(),
        _metrics=pd.DataFrame(index=df.columns, dtype=float),
    )

    # Use a deterministic payload to avoid heavy computations.
    payload = CovPayload(
        cov=np.eye(len(df.columns), dtype=float),
        mean=np.zeros(len(df.columns)),
        std=np.ones(len(df.columns)),
        n=len(df),
        assets=tuple(df.columns),
    )

    monkeypatch.setattr(
        "trend_analysis.perf.cache.compute_cov_payload",
        lambda _df, materialise_aggregates=False: payload,
    )

    ensured = rank_selection._ensure_cov_payload(df, bundle)
    assert ensured is payload
    assert bundle.cov_payload is payload


def test_metric_from_cov_payload_single_asset_returns_zero() -> None:
    payload = CovPayload(
        cov=np.array([[0.04]]),
        mean=np.array([0.0]),
        std=np.array([0.2]),
        n=5,
        assets=("Solo",),
    )
    avg = rank_selection._metric_from_cov_payload(
        "AvgCorr", pd.DataFrame(columns=["Solo"]), payload
    )
    assert avg.iloc[0] == 0.0


def test_compute_metric_series_with_cache_non_cov_metric_delegates() -> None:
    df = _sample_frame()
    cfg = RiskStatsConfig()
    result = rank_selection.compute_metric_series_with_cache(df, "AnnualReturn", cfg)
    assert isinstance(result, pd.Series)


def test_compute_metric_series_with_cache_single_asset_avgcorr() -> None:
    df = pd.DataFrame({"Solo": [0.01, 0.02, 0.03]})
    cfg = RiskStatsConfig()
    series = rank_selection.compute_metric_series_with_cache(df, "AvgCorr", cfg)
    assert series.name == "AvgCorr"
    assert series.iloc[0] == 0.0


def test_blended_score_empty_weights_error() -> None:
    df = _sample_frame()
    cfg = RiskStatsConfig()
    with pytest.raises(ValueError, match="non-empty weights dict"):
        rank_selection.blended_score(df, {}, cfg)


def test_blended_score_without_bundle_uses_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _sample_frame()
    cfg = RiskStatsConfig()
    observed: list[str] = []

    def _fake_metric(
        frame: pd.DataFrame, metric: str, cfg_arg: RiskStatsConfig
    ) -> pd.Series:
        observed.append(metric)
        return pd.Series([0.1, 0.2], index=frame.columns)

    monkeypatch.setattr(rank_selection, "_compute_metric_series", _fake_metric)

    score = rank_selection.blended_score(
        df,
        {"AnnualReturn": 0.7, "MaxDrawdown": 0.3},
        cfg,
        bundle=None,
    )

    assert observed == ["AnnualReturn", "MaxDrawdown"]
    assert isinstance(score, pd.Series)


def test_select_funds_random_requires_n(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2021-01-31", periods=3, freq="ME"),
            "RF": [0.0, 0.0, 0.0],
            "A": [0.01, 0.02, 0.03],
        }
    )

    with pytest.raises(ValueError, match="random_n must be provided"):
        rank_selection.select_funds(df, "RF", mode="random", n=0)

    captured: dict[str, object] = {}

    def _fake_choice(
        eligible: list[str], size: int, replace: bool = False
    ) -> np.ndarray:
        captured["eligible"] = eligible
        captured["size"] = size
        return np.array(eligible[:size])

    monkeypatch.setattr(np.random, "choice", _fake_choice)

    selected = rank_selection.select_funds(df, "RF", mode="random", n=1)

    expected_pool = rank_selection.quality_filter(df, default_quality_config())
    assert captured["eligible"] == expected_pool
    assert selected == [expected_pool[0]]


def test_select_funds_extended_random_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.period_range("2022-01", periods=3, freq="M").to_timestamp("M")
    df = pd.DataFrame(
        {
            "Date": dates,
            "RF": [0.0, 0.0, 0.0],
            "FundA": [0.01, 0.02, 0.03],
            "FundB": [0.02, 0.01, 0.02],
        }
    )
    cfg = default_quality_config()

    monkeypatch.setattr(
        np.random, "choice", lambda eligible, n, replace=False: np.array(eligible[:n])
    )

    selected = rank_selection.select_funds_extended(
        df,
        "RF",
        ["FundA", "FundB"],
        "2022-01",
        "2022-02",
        "2022-03",
        "2022-03",
        cfg,
        selection_mode="random",
        random_n=1,
    )

    assert selected == ["FundA"]


def test_select_funds_extended_rank_injects_window_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.period_range("2021-01", periods=3, freq="M").to_timestamp("M")
    df = pd.DataFrame(
        {
            "Date": dates,
            "RF": [0.0, 0.0, 0.0],
            "FundA": [0.01, 0.02, 0.03],
            "FundB": [0.02, 0.01, 0.02],
        }
    )
    cfg = default_quality_config()

    captured: dict[str, object] = {}

    def _fake_rank(
        df_slice: pd.DataFrame, stats: RiskStatsConfig, **kwargs: object
    ) -> list[str]:
        captured.update(kwargs)
        return list(df_slice.columns)

    monkeypatch.setattr(rank_selection, "rank_select_funds", _fake_rank)

    result = rank_selection.select_funds_extended(
        df,
        "RF",
        ["FundA", "FundB"],
        "2021-01",
        "2021-02",
        "2021-03",
        "2021-03",
        cfg,
        selection_mode="rank",
        rank_kwargs={},
    )

    assert result == ["FundA", "FundB"]
    assert "window_key" in captured
    assert "bundle" in captured


def test_select_funds_extended_all_and_unknown_mode() -> None:
    dates = pd.period_range("2020-01", periods=2, freq="M").to_timestamp("M")
    df = pd.DataFrame(
        {
            "Date": dates,
            "RF": [0.0, 0.0],
            "FundA": [0.01, 0.02],
        }
    )
    cfg = default_quality_config()

    all_selected = rank_selection.select_funds_extended(
        df,
        "RF",
        ["FundA"],
        "2020-01",
        "2020-01",
        "2020-02",
        "2020-02",
        cfg,
        selection_mode="all",
    )
    assert all_selected == ["FundA"]

    with pytest.raises(ValueError, match="Unsupported selection_mode"):
        rank_selection.select_funds_extended(
            df,
            "RF",
            ["FundA"],
            "2020-01",
            "2020-01",
            "2020-02",
            "2020-02",
            cfg,
            selection_mode="unknown",
        )


def test_select_funds_extended_requires_rank_kwargs() -> None:
    dates = pd.period_range("2023-01", periods=3, freq="M").to_timestamp("M")
    df = pd.DataFrame(
        {
            "Date": dates,
            "RF": [0.0, 0.0, 0.0],
            "FundA": [0.01, 0.02, 0.03],
            "FundB": [0.02, 0.01, 0.02],
        }
    )
    cfg = default_quality_config()

    with pytest.raises(ValueError, match="rank mode requires rank_kwargs"):
        rank_selection.select_funds_extended(
            df,
            "RF",
            ["FundA", "FundB"],
            "2023-01",
            "2023-02",
            "2023-03",
            "2023-03",
            cfg,
            selection_mode="rank",
            rank_kwargs=None,
        )


def test_select_funds_extended_random_requires_parameter() -> None:
    dates = pd.period_range("2024-01", periods=3, freq="M").to_timestamp("M")
    df = pd.DataFrame(
        {
            "Date": dates,
            "RF": [0.0, 0.0, 0.0],
            "FundA": [0.01, 0.02, 0.03],
        }
    )
    cfg = default_quality_config()

    with pytest.raises(ValueError, match="random_n must be provided for random mode"):
        rank_selection.select_funds_extended(
            df,
            "RF",
            ["FundA"],
            "2024-01",
            "2024-02",
            "2024-03",
            "2024-03",
            cfg,
            selection_mode="random",
        )


def test_select_funds_allows_extended_call_through_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-31", periods=2, freq="ME"),
            "RF": [0.0, 0.0],
            "A": [0.01, 0.02],
        }
    )
    cfg = default_quality_config()

    called: dict[str, object] = {}

    def _fake_extended(*args: object, **kwargs: object) -> list[str]:
        called["args"] = args
        called["kwargs"] = kwargs
        return ["A"]

    monkeypatch.setattr(rank_selection, "select_funds_extended", _fake_extended)

    out = rank_selection.select_funds(
        df,
        "RF",
        ["A"],
        "2025-01",
        "2025-01",
        "2025-02",
        "2025-02",
        cfg,
        "rank",
        1,
        {"rank": {}},
    )

    assert out == ["A"]
    assert called["args"][1] == "RF"
