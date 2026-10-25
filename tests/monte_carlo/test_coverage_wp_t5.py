"""Hermetic coverage tests for Monte Carlo runner, aggregator, and registry (wp t5)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import trend_analysis.monte_carlo.aggregator as aggregator_module
import trend_analysis.monte_carlo.registry as registry_module
import trend_analysis.monte_carlo.runner as runner_module
from trend_analysis.api import RunResult
from trend_analysis.monte_carlo.aggregator import (
    build_breach_frame,
    build_expected_shortfall_frame,
    build_quantiles_frame,
)
from trend_analysis.monte_carlo.registry import (
    load_scenario_from_path,
    normalize_return_model,
)
from trend_analysis.monte_carlo.costs import CostProcessOutput
from trend_analysis.monte_carlo.results import StrategyEvaluation
from trend_analysis.monte_carlo.runner import (
    MonteCarloRunner,
    _PathContext,
    _coerce_optional_bool,
    _coerce_turnover_guard,
    _has_turnover_override,
    _is_number,
)
from trend_analysis.monte_carlo.scenario import MonteCarloScenario
from trend_analysis.monte_carlo.strategy import StrategyVariant
from tests.monte_carlo.test_runner import _base_config, _price_history


def _runner(
    mode: str = "two_layer",
    *,
    strategy_set: dict[str, Any] | None = None,
    monte_carlo: dict[str, Any] | None = None,
) -> MonteCarloRunner:
    mc = {
        "mode": mode,
        "n_paths": 2,
        "horizon_years": 1.0,
        "frequency": "M",
        "seed": 99,
        "jobs": 1,
    }
    if monte_carlo:
        mc.update(monte_carlo)
    scenario = MonteCarloScenario(
        name="wp_t5",
        base_config="config/defaults.yml",
        monte_carlo=mc,
        strategy_set=strategy_set or {"curated": [StrategyVariant(name="A")]},
        return_model={"kind": "stationary_bootstrap", "params": {"block_size": 2}},
    )
    return MonteCarloRunner(scenario, base_config=_base_config(), price_history=_price_history())


def _path_context(**overrides: Any) -> _PathContext:
    dates = pd.date_range("2024-01-31", periods=3, freq="ME")
    returns = pd.DataFrame({"AssetA": [0.01, 0.02, -0.01]}, index=dates)
    base = {
        "path_id": 0,
        "prices": pd.DataFrame(),
        "returns": returns,
        "score_frame": pd.DataFrame(),
        "path_hash": "h",
        "seed": 1,
    }
    base.update(overrides)
    return _PathContext(**base)


# --- runner module helpers ---


def test_is_number_accepts_float_rejects_bool() -> None:
    assert _is_number(1.5) is True
    assert _is_number(True) is False


def test_coerce_optional_bool_rejects_non_bool() -> None:
    with pytest.raises(ValueError, match="pooled_distributions must be a boolean"):
        _coerce_optional_bool("yes", "pooled_distributions")


def test_coerce_turnover_guard_rejects_bool() -> None:
    with pytest.raises(
        ValueError,
        match="strategy_set.guards.max_turnover must be numeric or a distribution mapping",
    ):
        _coerce_turnover_guard(True)


def test_coerce_turnover_guard_rejects_non_numeric_string() -> None:
    with pytest.raises(
        ValueError,
        match="strategy_set.guards.max_turnover must be numeric or a distribution mapping",
    ):
        _coerce_turnover_guard("not-a-number")


def test_coerce_turnover_guard_coerces_numeric_string() -> None:
    assert _coerce_turnover_guard("0.25") == 0.25


def test_has_turnover_override_detects_portfolio_max_turnover() -> None:
    variant = StrategyVariant(name="x", overrides={"portfolio": {"max_turnover": 0.1}})
    assert _has_turnover_override(variant) is True
    assert _has_turnover_override(StrategyVariant(name="y")) is False


def test_resolve_strategies_public_wrapper() -> None:
    runner = _runner(strategy_set={"curated": [StrategyVariant(name="Only")]})
    names = [v.name for v in runner.resolve_strategies()]
    assert names == ["Only"]


def test_run_mode_rejects_unknown_mode() -> None:
    runner = _runner()
    with pytest.raises(ValueError, match="Unsupported Monte Carlo mode 'bogus'"):
        runner._run_mode(
            mode="bogus",
            model=object(),
            n_periods=3,
            strategies=[StrategyVariant(name="A")],
            path_seeds=[1, 2],
            strategy_seeds=[1, 2],
            progress_callback=None,
            jobs=1,
            fold_id=None,
            fold_label=None,
        )


def test_compute_n_periods_requires_frequency() -> None:
    runner = _runner()
    settings = runner._settings()
    object.__setattr__(settings, "frequency", None)
    with pytest.raises(ValueError, match="monte_carlo.frequency is required"):
        runner._compute_n_periods()


def test_compute_n_periods_requires_horizon() -> None:
    runner = _runner()
    settings = runner._settings()
    object.__setattr__(settings, "horizon_years", None)
    with pytest.raises(ValueError, match="monte_carlo.horizon_years is required"):
        runner._compute_n_periods()


def test_resolve_jobs_coerces_invalid_to_one() -> None:
    runner = _runner()
    assert runner._resolve_jobs("nope") == 1
    assert runner._resolve_jobs(-3) == 1


def test_resolve_strategies_string_curated_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    monkeypatch.setattr(runner, "_strategy_set", lambda: {"curated": ["Named"]})
    assert [v.name for v in runner._resolve_strategies()] == ["Named"]


def test_returns_to_prices_rejects_empty_and_sub_minus_one() -> None:
    runner = _runner()
    with pytest.raises(ValueError, match="returns data must not be empty"):
        runner._returns_to_prices(pd.DataFrame())
    bad = pd.DataFrame({"x": [-1.0, 0.1]})
    with pytest.raises(ValueError, match="returns contain values <= -1"):
        runner._returns_to_prices(bad)


def test_returns_to_prices_cumprod_from_returns() -> None:
    runner = _runner()
    returns = pd.DataFrame({"x": [0.1, 0.0]})
    prices = runner._returns_to_prices(returns)
    assert prices["x"].tolist() == pytest.approx([110.0, 110.0])


def test_slice_history_raises_on_empty_window() -> None:
    runner = _runner()
    history = _price_history()
    fold = SimpleNamespace(calibration_start=history.index[0], calibration_end=history.index[0])
    sliced = history.loc[fold.calibration_start : fold.calibration_end]
    assert sliced.shape[0] == 1
    empty_fold = SimpleNamespace(
        calibration_start=history.index[-1] + pd.Timedelta(days=30),
        calibration_end=history.index[-1] + pd.Timedelta(days=60),
    )
    with pytest.raises(ValueError, match="fold calibration window produced no history data"):
        runner._slice_history(history, empty_fold)


def test_resolve_price_history_from_csv_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text("Date,AssetA\n2020-01-31,100\n", encoding="utf-8")
    config = _base_config()
    config["data"]["csv_path"] = str(csv_path)
    scenario = MonteCarloScenario(
        name="csv_hist",
        base_config="config/defaults.yml",
        monte_carlo={"mode": "two_layer", "n_paths": 1, "horizon_years": 1.0, "frequency": "M"},
        return_model={"kind": "stationary_bootstrap"},
    )
    captured: list[Path] = []

    def _fake_load(path: Path) -> pd.DataFrame:
        captured.append(path)
        return _price_history()

    runner = MonteCarloRunner(scenario, base_config=config, price_history=None)
    monkeypatch.setattr(runner, "_load_history_from_path", _fake_load)
    frame = runner._resolve_price_history()
    assert not frame.empty
    assert captured == [csv_path]


def test_load_history_from_path_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    sentinel = _price_history()

    def _fake(path: Path) -> pd.DataFrame:
        assert path.name == "hist.csv"
        return sentinel

    monkeypatch.setattr(runner_module, "load_price_history", _fake)
    pd.testing.assert_frame_equal(runner._load_history_from_path(Path("hist.csv")), sentinel)


def test_build_regime_proxy_series_uses_date_column() -> None:
    runner = _runner()
    returns = pd.DataFrame(
        {"Date": pd.date_range("2024-01-31", periods=2, freq="ME"), "proxy": [1.0, 2.0]}
    )
    series = runner._build_regime_proxy_series(returns, "proxy")
    assert series is not None
    assert series.tolist() == pytest.approx([1.0, 2.0])
    assert isinstance(series.index, pd.DatetimeIndex)


def test_resolve_regime_proxy_column_matches_benchmark_key() -> None:
    runner = _runner()
    returns = pd.DataFrame({"Vol": [1.0, 2.0]})
    col = runner._resolve_regime_proxy_column(returns, "bench", {"bench": "Vol"})
    assert col == "Vol"


def test_resolve_sampled_strategies_disabled_via_null_enabled() -> None:
    runner = _runner(
        strategy_set={
            "curated": [],
            "sampled": {"enabled": None, "n_strategies": 1, "sampling": {}},
        }
    )
    assert (
        runner._resolve_sampled_strategies(
            {"sampled": {"enabled": None, "n_strategies": 1, "sampling": {}}}, existing=[]
        )
        == []
    )


def test_resolve_sampled_strategies_rejects_non_bool_enabled() -> None:
    runner = _runner()
    with pytest.raises(ValueError, match="strategy_set.sampled.enabled must be a bool"):
        runner._resolve_sampled_strategies(
            {"sampled": {"enabled": "yes", "n_strategies": 1, "sampling": {}}}, existing=[]
        )


def test_build_price_model_rejects_unknown_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    monkeypatch.setattr(
        registry_module,
        "normalize_return_model",
        lambda _value: {"kind": "exotic", "params": {}},
    )
    with pytest.raises(ValueError, match="Unsupported return model 'exotic'"):
        runner._build_price_model()


class _BrokenModel:
    def sample_prices(self, **_kwargs: Any) -> Any:
        raise RuntimeError("sampling failed")


def test_two_layer_sample_prices_failure_records_all_paths() -> None:
    runner = _runner()
    evals, errors = runner._run_two_layer(
        model=_BrokenModel(),
        n_periods=3,
        strategies=[StrategyVariant(name="A")],
        path_seeds=[None, None],
        progress_callback=None,
        jobs=1,
    )
    assert evals == []
    assert len(errors) == 2
    assert {e.error_type for e in errors} == {"RuntimeError"}


def test_mixture_sample_prices_failure_records_all_paths() -> None:
    runner = _runner(mode="mixture")
    evals, errors = runner._run_mixture(
        model=_BrokenModel(),
        n_periods=3,
        strategies=[StrategyVariant(name="A"), StrategyVariant(name="B")],
        path_seeds=[None, None],
        strategy_seeds=[10, 20],
        progress_callback=None,
        jobs=1,
    )
    assert evals == []
    assert len(errors) == 2


def test_mixture_path_context_failure_records_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(mode="mixture")

    def _boom(*_args: Any, **_kwargs: Any) -> _PathContext:
        raise RuntimeError("context boom")

    monkeypatch.setattr(runner, "_generate_path_context", _boom)
    evals, errors = runner._run_mixture(
        model=_RecordingModel(),
        n_periods=3,
        strategies=[StrategyVariant(name="A")],
        path_seeds=[1],
        strategy_seeds=[9],
        progress_callback=None,
        jobs=1,
    )
    assert evals == []
    assert len(errors) == 1
    assert errors[0].message == "context boom"


def test_mixture_strategy_evaluation_failure_records_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(mode="mixture")
    monkeypatch.setattr(
        runner,
        "_generate_path_context",
        lambda **_kw: _path_context(),
    )

    def _boom(_strategy: StrategyVariant, _ctx: _PathContext) -> StrategyEvaluation:
        raise RuntimeError("eval boom")

    monkeypatch.setattr(runner, "_evaluate_strategy", _boom)
    evals, errors = runner._run_mixture(
        model=_RecordingModel(),
        n_periods=3,
        strategies=[StrategyVariant(name="A")],
        path_seeds=[1],
        strategy_seeds=[9],
        progress_callback=None,
        jobs=1,
    )
    assert evals == []
    assert len(errors) == 1
    assert errors[0].strategy_name == "A"
    assert errors[0].error_type == "RuntimeError"
    assert errors[0].message == "eval boom"


class _RecordingModel:
    def sample_prices(
        self,
        *,
        n_periods: int,
        n_paths: int,
        frequency: str,
        seed: int | None,
    ) -> Any:
        index = pd.date_range("2024-01-31", periods=n_periods, freq="ME")
        cols = pd.MultiIndex.from_product([range(n_paths), ["AssetA"]], names=["path", "asset"])
        prices = pd.DataFrame(100.0, index=index, columns=cols)
        log_returns = pd.DataFrame(0.0, index=index, columns=cols)
        return SimpleNamespace(prices=prices, log_returns=log_returns)


def test_log_path_error_fold_label_only(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("trend_analysis.monte_carlo.wp_t5")
    runner = _runner()
    runner._logger = logger
    with caplog.at_level(logging.ERROR):
        runner._log_path_error(1, None, RuntimeError("x"), fold_label="Jan")
    assert "fold Jan path 1" in caplog.records[-1].message


def test_execute_paths_parallel_catches_worker_failure() -> None:
    runner = _runner()

    def _boom(path_id: int, _seed: int | None) -> tuple[list[Any], list[Any]]:
        if path_id == 0:
            raise RuntimeError("parallel boom")
        return [], []

    results = list(runner._execute_paths([1, 2], _boom, jobs=2))
    by_path = {path_id: errs for path_id, _evals, errs in results}
    assert len(by_path) == 2
    assert by_path[0][0].error_type == "RuntimeError"
    assert by_path[0][0].message == "parallel boom"
    assert by_path[1] == []


def test_coerce_nav_index_coerces_non_datetime_index() -> None:
    runner = _runner()
    idx = pd.Index(["2024-01-31", "2024-02-29", "2024-03-31"])
    coerced = runner._coerce_nav_index(idx, SimpleNamespace(), _path_context(), 3)
    assert isinstance(coerced, pd.DatetimeIndex)
    assert len(coerced) == 3


def test_returns_with_date_from_non_datetime_index() -> None:
    runner = _runner()
    returns = pd.DataFrame({"x": [0.1, 0.2]}, index=[0, 1])
    out = runner._returns_with_date(returns)
    assert "Date" in out.columns
    assert len(out) == 2


def test_stats_to_metrics_reads_object_dict() -> None:
    stats = SimpleNamespace(cagr=0.1, note="skip")
    metrics = MonteCarloRunner._stats_to_metrics(stats)
    assert metrics == {"cagr": 0.1}


def test_extract_metrics_uses_first_row_when_no_known_index() -> None:
    runner = _runner()
    metrics_df = pd.DataFrame({"a": [1.0], "b": [2.0]}, index=["custom"])
    out, source = runner._extract_metrics(metrics_df)
    assert out == {"a": 1.0, "b": 2.0}
    assert source == "custom"


def test_fit_vol_adjust_window_caps_mapping_window() -> None:
    runner = _runner()
    merged = {
        "vol_adjust": {"enabled": True, "window": {"length": "bad"}},
    }
    runner._fit_vol_adjust_window(merged, 5)
    merged2 = {"vol_adjust": {"enabled": True, "window": {"length": 20}}}
    runner._fit_vol_adjust_window(merged2, 5)
    assert merged2["vol_adjust"]["window"]["length"] == 5


def test_fit_vol_adjust_window_caps_scalar_window() -> None:
    runner = _runner()
    merged = {"vol_adjust": {"enabled": True, "window": 30}}
    runner._fit_vol_adjust_window(merged, 4)
    assert merged["vol_adjust"]["window"] == 4


def test_fit_signal_window_caps_min_periods() -> None:
    runner = _runner()
    merged = {"signals": {"window": 20, "min_periods": 15}}
    runner._fit_signal_window(merged, 6)
    assert merged["signals"]["window"] == 6
    assert merged["signals"]["min_periods"] == 6


def test_apply_strategy_guards_coerces_numeric_guard() -> None:
    runner = _runner(strategy_set={"guards": {"max_turnover": "0.3"}})
    merged: dict[str, Any] = {"portfolio": {}}
    runner._apply_strategy_guards(merged)
    assert merged["portfolio"]["max_turnover"] == 0.3


def test_apply_strategy_guards_copies_mapping_guard() -> None:
    runner = _runner(strategy_set={"guards": {"max_turnover": {"calm": 0.2}}})
    merged: dict[str, Any] = {"portfolio": {}}
    runner._apply_strategy_guards(merged)
    assert merged["portfolio"]["max_turnover"] == {"calm": 0.2}


def test_resolve_turnover_guard_distribution_numeric_guard() -> None:
    runner = _runner(strategy_set={"guards": {"max_turnover": 0.5}})
    assert runner._resolve_turnover_guard_distribution() is None


def test_resolve_turnover_guard_distribution_invalid_string() -> None:
    runner = _runner(strategy_set={"guards": {"max_turnover": "bad"}})
    with pytest.raises(
        ValueError,
        match="strategy_set.guards.max_turnover must be numeric or a distribution mapping",
    ):
        runner._resolve_turnover_guard_distribution()


def test_base_metrics_settings_empty_when_metrics_not_mapping() -> None:
    config = _base_config()
    config["metrics"] = "not-a-mapping"
    scenario = MonteCarloScenario(
        name="m",
        base_config="config/defaults.yml",
        monte_carlo={"mode": "two_layer", "n_paths": 1, "horizon_years": 1.0, "frequency": "M"},
        return_model={"kind": "stationary_bootstrap"},
    )
    runner = MonteCarloRunner(scenario, base_config=config, price_history=_price_history())
    assert runner._base_metrics_settings() == {}


def test_coerce_nav_paths_columns_plain_sort() -> None:
    runner = _runner()
    frame = pd.DataFrame(
        {2: [1.0], 1: [2.0]}, index=pd.date_range("2024-01-31", periods=1, freq="ME")
    )
    out = runner._coerce_nav_paths_columns(frame)
    assert list(out.columns) == [1, 2]


def test_coerce_nav_paths_columns_multiindex_asset_normalization() -> None:
    runner = _runner()
    arrays = [
        pd.Index([0, 1]),
        pd.Index(["EQ", "EQ"]),
    ]
    cols = pd.MultiIndex.from_arrays(arrays, names=["path", "asset"])
    frame = pd.DataFrame(
        [[1.0, 2.0]], index=pd.date_range("2024-01-31", periods=1, freq="ME"), columns=cols
    )
    out = runner._coerce_nav_paths_columns(frame)
    assets = out.columns.get_level_values("asset")
    assert assets.tolist() == ["NAV", "NAV"]


def test_turnover_cap_binding_scalar_non_positive() -> None:
    runner = _runner()
    turnover = pd.Series([0.2, 0.3])
    binding = runner._turnover_cap_binding_indicator(turnover, 0.0)
    assert binding is not None
    assert binding.tolist() == [False, False]


def test_resolve_turnover_cap_from_parsed_float_passthrough() -> None:
    runner = _runner()
    assert runner._resolve_turnover_cap_from_parsed(0.15, None, object()) == 0.15


def test_resolve_turnover_cap_from_parsed_lookup_fallbacks() -> None:
    runner = _runner()
    parsed = {"default": 0.1, "calm": 0.2}
    settings = SimpleNamespace(default_label="calm")
    assert runner._resolve_turnover_cap_from_parsed(parsed, "missing", settings) == 0.2
    assert runner._resolve_turnover_cap_from_parsed({"all": 0.05}, None, settings) == 0.05


def test_resolve_turnover_cap_series_from_parsed_scalar_broadcast() -> None:
    runner = _runner()
    idx = pd.date_range("2024-01-31", periods=2, freq="ME")
    series = runner._resolve_turnover_cap_series_from_parsed(0.2, None, idx)
    assert isinstance(series, pd.Series)
    assert series.tolist() == pytest.approx([0.2, 0.2])


def test_resolve_turnover_cap_regimes_canonical_alias() -> None:
    runner = _runner()
    idx = pd.date_range("2024-01-31", periods=1, freq="ME")
    regimes = pd.Series(["calm"], index=idx)
    out = runner._resolve_turnover_cap_regimes({"calm": 0.1}, regimes, idx)
    assert out is not None
    assert out.iloc[0] == "riskon"


# --- aggregator private paths ---


def test_ensure_path_columns_rejects_legacy_fold() -> None:
    frame = pd.DataFrame({"path_id": [1], "fold": [0], "metric": [1.0]})
    with pytest.raises(ValueError, match="legacy fold column is not supported"):
        aggregator_module._ensure_path_columns(frame)


def test_ensure_path_columns_fills_from_path_hash() -> None:
    frame = pd.DataFrame({"path_hash": ["h1", "h2"], "fold_id": [0, 0], "metric": [1.0, 2.0]})
    out = aggregator_module._ensure_path_columns(frame)
    assert out.loc[0, "path"] == "h1"
    assert out.loc[0, "path_id"] == "h1"
    assert out.loc[1, "path"] == "h2"
    assert out.loc[1, "path_id"] == "h2"


def test_coerce_quantiles_bool_scalar_raises() -> None:
    with pytest.raises(TypeError, match="Quantiles must be numeric values"):
        aggregator_module._coerce_quantiles(True)


def test_coerce_quantiles_skips_none_and_invalid_numeric() -> None:
    out = aggregator_module._coerce_quantiles([None, "nope", 0.5])
    assert out == [0.5]


def test_coerce_quantiles_percent_only_punctuation() -> None:
    out = aggregator_module._coerce_quantiles(["%"])
    assert out == pytest.approx([0.05, 0.5, 0.95])


def test_build_breach_frame_zero_paths_probability_nan() -> None:
    frame = pd.DataFrame(
        [
            {"strategy": "A", "fold_id": 0, "path": 1, "metric": float("nan")},
            {"strategy": "A", "fold_id": 0, "path": 2, "metric": float("inf")},
        ]
    )
    breach = build_breach_frame(frame, {"metric": [1.0]})
    assert breach.loc[0, "paths"] == 0
    assert np.isnan(breach.loc[0, "breach_probability"])


def test_build_expected_shortfall_all_non_finite_paths_zero() -> None:
    frame = pd.DataFrame(
        [
            {"strategy": "A", "fold_id": 0, "path": 1, "metric": float("nan")},
            {"strategy": "A", "fold_id": 0, "path": 2, "metric": float("inf")},
        ]
    )
    shortfall = build_expected_shortfall_frame(frame, {"metric": 0.5})
    assert shortfall.loc[0, "paths"] == 0
    assert np.isnan(shortfall.loc[0, "expected_shortfall"])


def test_coerce_breach_specs_scalar_invalid_returns_empty() -> None:
    specs = aggregator_module._coerce_breach_specs("nope", ["metric"])
    assert specs == []


def test_coerce_shortfall_specs_invalid_scalar_returns_empty() -> None:
    specs = aggregator_module._coerce_shortfall_specs("nope", ["metric"])
    assert specs == []


def test_coerce_shortfall_parse_none_alpha_falls_back_to_defaults() -> None:
    specs = aggregator_module._coerce_shortfall_specs({"metric": {"alpha": "  "}}, ["metric"])
    assert specs == [("metric", 0.05, "lower")]


def test_sort_frame_mixed_types_uses_string_key() -> None:
    frame = pd.DataFrame({"a": [1, "b"]})
    sorted_frame = aggregator_module._sort_frame(frame, ["a"])
    assert len(sorted_frame) == 2


def test_build_quantiles_injects_single_quantile_column() -> None:
    path_frame = pd.DataFrame(
        {
            "strategy": ["A", "A"],
            "fold_id": [0, 0],
            "path": [1, 2],
            "metric": [1.0, 3.0],
        }
    )
    quantiles = build_quantiles_frame(path_frame, [0.5])
    assert quantiles.loc[0, "value"] == pytest.approx(2.0)


# --- registry ---


def test_ensure_mapping_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="label must be a mapping"):
        registry_module._ensure_mapping([], label="label")


def test_normalize_return_model_regime_must_be_mapping() -> None:
    with pytest.raises(ValueError, match="return_model.regime must be a mapping"):
        normalize_return_model({"kind": "regime_bootstrap", "regime": "bad"})


def test_normalize_return_model_unknown_regime_field() -> None:
    with pytest.raises(ValueError, match="return_model.regime has unknown settings"):
        normalize_return_model(
            {
                "kind": "regime_bootstrap",
                "regime": {"enabled": True, "extra": 1},
            }
        )


def test_load_yaml_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.yml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="is empty"):
        registry_module._load_yaml(path)


def test_load_strategy_pack_requires_list_curated(tmp_path: Path) -> None:
    pack = tmp_path / "pack.yml"
    pack.write_text("curated: not-list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="curated.*list"):
        registry_module._load_strategy_pack(pack)


def test_load_strategy_pack_requires_strategies_or_curated(tmp_path: Path) -> None:
    pack = tmp_path / "pack.yml"
    pack.write_text("other: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must define 'strategies' or 'curated'"):
        registry_module._load_strategy_pack(pack)


def test_coerce_tags_non_sequence_returns_empty() -> None:
    assert registry_module._coerce_tags(42) == ()


def test_list_scenarios_blank_tag_filter_returns_all(tmp_path: Path) -> None:
    alpha = tmp_path / "a.yml"
    alpha.write_text("{}", encoding="utf-8")
    registry = tmp_path / "index.yml"
    registry.write_text("scenarios:\n  - name: alpha\n    path: a.yml\n", encoding="utf-8")
    entries = registry_module.list_scenarios(tags=["", "  "], registry_path=registry)
    assert [e.name for e in entries] == ["alpha"]


def test_resolve_path_rejects_directory(tmp_path: Path) -> None:
    folder = tmp_path / "dir.yml"
    folder.mkdir()
    with pytest.raises(IsADirectoryError, match="must be a file"):
        registry_module._resolve_path("dir.yml", base_dir=tmp_path)


def test_resolve_base_config_rejects_outside_roots(tmp_path: Path) -> None:
    outside = tmp_path / "outside.yml"
    outside.write_text("{}", encoding="utf-8")
    source = tmp_path / "nested" / "scenario.yml"
    source.parent.mkdir()
    source.write_text("name: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="base_config must resolve under"):
        registry_module._resolve_base_config(str(outside.resolve()), source_path=source)


def test_resolve_strategy_pack_rejects_bad_suffix(tmp_path: Path) -> None:
    pack = tmp_path / "pack.json"
    pack.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="must use one of"):
        registry_module._resolve_strategy_pack(str(pack.resolve()), source_path=tmp_path)


def test_load_scenario_from_path_rejects_missing_and_directory(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yml"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_scenario_from_path(missing)
    folder = tmp_path / "folder.yml"
    folder.mkdir()
    with pytest.raises(IsADirectoryError, match="must be a file"):
        load_scenario_from_path(folder)


def test_load_scenario_from_path_rejects_bad_suffix(tmp_path: Path) -> None:
    bad = tmp_path / "scenario.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="must use one of"):
        load_scenario_from_path(bad)


def test_load_scenario_from_path_reads_name_from_scenario_block(tmp_path: Path) -> None:
    base = tmp_path / "base.yml"
    base.write_text("version: '0.1.0'\ndata: {}\n", encoding="utf-8")
    scenario_file = tmp_path / "case.yml"
    scenario_file.write_text(
        "scenario:\n"
        "  name: from_block\n"
        "base_config: base.yml\n"
        "monte_carlo:\n"
        "  mode: two_layer\n"
        "  n_paths: 1\n"
        "  horizon_years: 1.0\n"
        "  frequency: M\n"
        "return_model:\n"
        "  kind: stationary_bootstrap\n",
        encoding="utf-8",
    )
    loaded = load_scenario_from_path(scenario_file)
    assert loaded.name == "from_block"


def test_parse_scenario_curated_pack_merges_inline(tmp_path: Path) -> None:
    base = tmp_path / "base.yml"
    base.write_text("version: '0.1.0'\ndata: {}\n", encoding="utf-8")
    pack = tmp_path / "pack.yml"
    pack.write_text("strategies:\n  - name: packed\n", encoding="utf-8")
    scenario_file = tmp_path / "merge.yml"
    scenario_file.write_text(
        "name: merge_case\n"
        "base_config: base.yml\n"
        "monte_carlo:\n"
        "  mode: two_layer\n"
        "  n_paths: 1\n"
        "  horizon_years: 1.0\n"
        "  frequency: M\n"
        "strategy_set:\n"
        "  curated_pack: pack.yml\n"
        "  curated:\n"
        "    - name: inline\n"
        "return_model:\n"
        "  kind: stationary_bootstrap\n",
        encoding="utf-8",
    )
    loaded = registry_module._parse_scenario(
        "merge_case", registry_module._load_yaml(scenario_file), source_path=scenario_file
    )
    curated = loaded.strategy_set["curated"]
    names = [item["name"] if isinstance(item, dict) else item.name for item in curated]
    assert names == ["packed", "inline"]


def test_parse_scenario_enable_fold_runs_passthrough(tmp_path: Path) -> None:
    base = tmp_path / "base.yml"
    base.write_text("version: '0.1.0'\ndata: {}\n", encoding="utf-8")
    scenario_file = tmp_path / "folds.yml"
    scenario_file.write_text(
        "name: fold_flag\n"
        "base_config: base.yml\n"
        "enable_fold_runs: false\n"
        "monte_carlo:\n"
        "  mode: two_layer\n"
        "  n_paths: 1\n"
        "  horizon_years: 1.0\n"
        "  frequency: M\n"
        "return_model:\n"
        "  kind: stationary_bootstrap\n",
        encoding="utf-8",
    )
    loaded = registry_module._parse_scenario(
        "fold_flag", registry_module._load_yaml(scenario_file), source_path=scenario_file
    )
    assert loaded.enable_fold_runs is False


def test_resolve_regime_proxy_column_case_insensitive_column() -> None:
    runner = _runner()
    returns = pd.DataFrame({"MyProxy": [1.0]})
    assert runner._resolve_regime_proxy_column(returns, "myproxy", None) == "MyProxy"


def test_resolve_sampled_strategies_disabled_flag() -> None:
    runner = _runner()
    out = runner._resolve_sampled_strategies(
        {
            "sampled": {
                "enabled": False,
                "n_strategies": 2,
                "sampling": {"kind": "fixed", "variants": [{"name": "x"}]},
            }
        },
        existing=[],
    )
    assert out == []


def test_resolve_sampled_strategies_requires_integer_n() -> None:
    runner = _runner()
    with pytest.raises(ValueError, match="n_strategies must be an integer"):
        runner._resolve_sampled_strategies(
            {"sampled": {"n_strategies": "x", "sampling": {}}}, existing=[]
        )


def test_resolve_sampled_strategies_requires_sampling_mapping() -> None:
    runner = _runner()
    with pytest.raises(ValueError, match="sampling must be a mapping"):
        runner._resolve_sampled_strategies(
            {"sampled": {"n_strategies": 1, "sampling": []}}, existing=[]
        )


def test_build_seeds_requires_n_paths() -> None:
    runner = _runner()
    settings = runner._settings()
    object.__setattr__(settings, "n_paths", None)
    with pytest.raises(ValueError, match="monte_carlo.n_paths is required"):
        runner._build_seeds()


def test_align_path_windows_invalid_ratio_uses_default() -> None:
    runner = _runner()
    merged: dict[str, Any] = {"sample_split": {"method": "ratio", "ratio": "bad"}}
    runner._align_path_windows(merged, _path_context())
    assert merged["sample_split"]["ratio"] == 0.7


def test_align_path_windows_honors_valid_ratio() -> None:
    runner = _runner()
    merged: dict[str, Any] = {"sample_split": {"method": "ratio", "ratio": 0.4}}
    runner._align_path_windows(merged, _path_context())
    assert merged["sample_split"]["ratio"] == 0.4


def test_multi_period_in_sample_cap_counts_rows() -> None:
    runner = _runner()
    dates = pd.date_range("2024-01-31", periods=6, freq="ME")
    returns = pd.DataFrame({"Date": dates, "AssetA": np.zeros(6)})
    ctx = _path_context(returns=returns)
    period = SimpleNamespace(in_start=dates[0], in_end=dates[2])
    cap = runner._multi_period_in_sample_cap(ctx, [period])
    assert cap == 3


def test_period_results_regimes_builds_series() -> None:
    runner = _runner()
    idx = pd.date_range("2024-01-31", periods=2, freq="ME")
    period_results = [
        {
            "period": (idx[0], idx[1], idx[0]),
            "regime_labels_out": pd.Series(["stress"], index=[idx[0]]),
        }
    ]
    series = runner._period_results_regimes(period_results, idx)
    assert series is not None
    assert series.iloc[0] == "stress"


def test_resolve_nav_return_series_prefers_analysis_returns() -> None:
    runner = _runner()
    series = pd.Series([0.01, 0.02], index=pd.date_range("2024-01-31", periods=2, freq="ME"))
    run_result = SimpleNamespace(
        portfolio=None,
        analysis=SimpleNamespace(returns=series),
        details={},
    )
    out = runner._resolve_nav_return_series(run_result, prefer_user=False)
    pd.testing.assert_series_equal(out, series)


def test_resolve_cost_index_from_period_results() -> None:
    runner = _runner()
    idx = pd.date_range("2024-01-31", periods=2, freq="ME")
    run_result = SimpleNamespace(
        details={
            "period_results": [
                {"period": (idx[0], idx[1], idx[0])},
                {"period": (idx[1], idx[1], idx[1])},
            ]
        }
    )
    out = runner._resolve_cost_index(run_result, _path_context())
    assert out is not None
    assert len(out) == 2


def test_resolve_turnover_series_broadcasts_scalar_series() -> None:
    runner = _runner()
    idx = pd.date_range("2024-01-31", periods=3, freq="ME")
    run_result = SimpleNamespace(turnover=pd.Series([0.2], index=[idx[0]]), details={})
    out = runner._resolve_turnover_series(run_result, idx)
    assert isinstance(out, pd.Series)
    assert out.tolist() == pytest.approx([0.2, 0.2, 0.2])


def test_resolve_turnover_cap_series_delegates() -> None:
    runner = _runner()
    idx = pd.date_range("2024-01-31", periods=2, freq="ME")
    out = runner._resolve_turnover_cap_series(0.25, None, idx, regime_settings=None)
    assert isinstance(out, pd.Series)
    assert out.iloc[0] == 0.25


def test_lookup_turnover_cap_returns_nan_for_unknown_label() -> None:
    runner = _runner()
    idx = pd.date_range("2024-01-31", periods=1, freq="ME")
    regimes = pd.Series(["unknown"], index=idx)
    caps = runner._resolve_turnover_cap_series_from_parsed({"calm": 0.1}, regimes, idx)
    assert np.isnan(caps.iloc[0])


def test_coerce_turnover_series_scalar_broadcast() -> None:
    runner = _runner()
    idx = pd.date_range("2024-01-31", periods=2, freq="ME")
    out = runner._coerce_turnover_series(0.15, idx)
    assert out.tolist() == pytest.approx([0.15, 0.15])


def test_period_results_index_skips_non_mapping_entries() -> None:
    runner = _runner()
    assert runner._period_results_index(["not-a-mapping"]) is None


def test_build_nav_paths_frame_skips_non_normalized_nav(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    idx = pd.date_range("2024-01-31", periods=2, freq="ME")
    bad_nav = pd.Series([2.0, 2.1], index=idx, name="NAV")
    evaluation = StrategyEvaluation(
        fold_id=None,
        path_id=0,
        strategy_name="A",
        metrics={},
        metric_source="x",
        path_hash="h",
        seed=1,
        nav_series=bad_nav,
    )
    frame = runner._build_nav_paths_frame([evaluation], primary_strategy_name="A", fold_id=None)
    assert frame.empty


def test_coerce_base_config_loads_yaml_file(tmp_path: Path) -> None:
    config_path = tmp_path / "base.yml"
    config_path.write_text(
        "version: '0.1.0'\n"
        "data: {frequency: M}\n"
        "preprocessing: {}\n"
        "vol_adjust: {}\n"
        "sample_split: {}\n"
        "portfolio: {}\n"
        "metrics: {}\n"
        "export: {}\n"
        "run: {}\n"
        "benchmarks: {}\n",
        encoding="utf-8",
    )
    scenario = MonteCarloScenario(
        name="from_file",
        base_config=config_path,
        monte_carlo={"mode": "two_layer", "n_paths": 1, "horizon_years": 1.0, "frequency": "M"},
        return_model={"kind": "stationary_bootstrap"},
    )
    runner = MonteCarloRunner(scenario, base_config=None, price_history=_price_history())
    assert runner._base_config["data"]["frequency"] == "M"


def test_base_config_path_requires_path_type() -> None:
    scenario = MonteCarloScenario(
        name="bad_base",
        base_config="config/defaults.yml",
        monte_carlo={"mode": "two_layer", "n_paths": 1, "horizon_years": 1.0, "frequency": "M"},
        return_model={"kind": "stationary_bootstrap"},
    )
    runner = MonteCarloRunner(scenario, base_config=_base_config(), price_history=_price_history())
    object.__setattr__(runner.scenario, "base_config", 42)
    with pytest.raises(TypeError, match="base_config must be a path"):
        runner._base_config_path()


def test_evaluate_strategy_records_run_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    diagnostic = SimpleNamespace(reason_code="RC", message="msg")

    def _fake_run(_config: Any, _returns: pd.DataFrame) -> RunResult:
        return RunResult(
            metrics=pd.DataFrame({"x": [1.0]}, index=["user_weight"]),
            details={},
            seed=0,
            environment={},
            diagnostic=diagnostic,
        )

    monkeypatch.setattr(runner_module, "run_simulation", _fake_run)
    evaluation = runner._evaluate_strategy(StrategyVariant(name="A"), _path_context())
    assert evaluation.diagnostic is not None
    assert evaluation.diagnostic["reason_code"] == "RC"


def test_evaluate_strategy_cost_date_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    idx = pd.date_range("2024-01-31", periods=2, freq="ME")
    gross = pd.Series([0.01, 0.02], index=idx)

    @dataclass
    class _CostOut:
        transaction_costs: pd.Series
        total_cost_drag: float = 0.0
        turnover: pd.Series | None = None
        cost_drag: pd.Series | None = None

    def _fake_run(_config: Any, _returns: pd.DataFrame) -> RunResult:
        return RunResult(
            metrics=pd.DataFrame({"x": [1.0]}, index=["user_weight"]),
            details={"portfolio_user_weight": gross},
            seed=0,
            environment={},
        )

    def _fake_costs(_run_result: Any, _ctx: _PathContext, _strategy: StrategyVariant) -> _CostOut:
        bad_idx = pd.date_range("2025-01-31", periods=2, freq="ME")
        return _CostOut(transaction_costs=pd.Series([0.001, 0.001], index=bad_idx))

    monkeypatch.setattr(runner_module, "run_simulation", _fake_run)
    monkeypatch.setattr(runner, "_maybe_sample_costs", _fake_costs)
    with pytest.raises(ValueError, match="Scenario cost dates do not match portfolio returns"):
        runner._evaluate_strategy(StrategyVariant(name="A"), _path_context())


def test_aggregator_fill_missing_merge_keys_inserts_sentinel() -> None:
    frame = pd.DataFrame({"strategy": [None, "A"], "fold_id": [0, 0], "metric": ["m", "m"]})
    sentinel = object()
    out = aggregator_module._fill_missing_merge_keys(frame, ["strategy"], sentinel)
    assert out.loc[0, "strategy"] is sentinel


def test_aggregator_select_strategy_column_prefers_name_for_numeric() -> None:
    frame = pd.DataFrame({"strategy": [1, 2], "strategy_name": ["A", "B"]})
    selected = aggregator_module._select_strategy_column(frame)
    assert selected.tolist() == ["A", "B"]


def test_aggregator_is_numeric_like_object_strings() -> None:
    series = pd.Series(["1.5", "2.5"])
    assert aggregator_module._is_numeric_like(series) is True


def test_aggregator_coerce_breach_percent_threshold_skips_blank() -> None:
    specs = aggregator_module._coerce_breach_specs({"metric": ["%"]}, ["metric"])
    assert specs == []


def test_aggregator_coerce_breach_mapping_empty_threshold_list() -> None:
    specs = aggregator_module._coerce_breach_specs({"metric": {"thresholds": []}}, ["metric"])
    assert specs == []


def test_aggregator_coerce_shortfall_bool_alpha_raises() -> None:
    with pytest.raises(TypeError, match="Expected shortfall alpha must be numeric"):
        aggregator_module._coerce_shortfall_specs(True, ["metric"])


def test_aggregator_ensure_path_columns_path_from_path_id() -> None:
    frame = pd.DataFrame({"path_id": [3], "fold_id": [0], "metric": [1.0]})
    out = aggregator_module._ensure_path_columns(frame)
    assert out.loc[0, "path"] == 3
    assert out.loc[0, "path_id"] == 3


def test_normalize_return_model_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="return_model must be a mapping"):
        normalize_return_model("bad")


def test_load_strategy_pack_reads_strategies_list(tmp_path: Path) -> None:
    pack = tmp_path / "pack.yml"
    pack.write_text("strategies:\n  - name: one\n", encoding="utf-8")
    loaded = registry_module._load_strategy_pack(pack)
    assert loaded[0]["name"] == "one"


def test_resolve_path_rejects_unsupported_suffix(tmp_path: Path) -> None:
    bad = tmp_path / "x.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="must use one of"):
        registry_module._resolve_path("x.json", base_dir=tmp_path)


def test_extract_scenario_metadata_conflicting_name(tmp_path: Path) -> None:
    raw = {
        "name": "top",
        "scenario": {"name": "block"},
        "base_config": "b.yml",
    }
    with pytest.raises(ValueError, match="conflicting 'name' values"):
        registry_module._extract_scenario_metadata("reg", raw, source_path=tmp_path / "s.yml")


def test_parse_scenario_rejects_null_monte_carlo(tmp_path: Path) -> None:
    base = tmp_path / "base.yml"
    base.write_text("{}", encoding="utf-8")
    scenario_file = tmp_path / "null_mc.yml"
    scenario_file.write_text(
        "name: null_mc\nbase_config: base.yml\nmonte_carlo: null\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="monte_carlo.*null provided"):
        registry_module._parse_scenario(
            "null_mc", registry_module._load_yaml(scenario_file), source_path=scenario_file
        )


def test_load_scenario_from_path_rejects_non_mapping_scenario_block(tmp_path: Path) -> None:
    path = tmp_path / "bad.yml"
    path.write_text("scenario: 1\nname: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="scenario.*must be a mapping"):
        load_scenario_from_path(path)


def test_evaluate_strategy_applies_net_cost_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    idx = pd.date_range("2024-01-31", periods=2, freq="ME")
    gross = pd.Series([0.10, -0.05], index=idx)

    def _fake_run(_config: Any, _returns: pd.DataFrame) -> RunResult:
        return RunResult(
            metrics=pd.DataFrame({"cagr": [0.2]}, index=["user_weight"]),
            details={
                "portfolio_user_weight": gross,
                "periods_per_year": 12.0,
                "risk_free_out_sample": pd.Series([0.0, 0.0], index=idx),
            },
            seed=0,
            environment={},
        )

    def _fake_costs(
        _run_result: Any, _ctx: _PathContext, _strategy: StrategyVariant
    ) -> CostProcessOutput:
        debits = pd.Series([0.01, 0.02], index=idx)
        zeros = pd.Series([0.0, 0.0], index=idx)
        return CostProcessOutput(
            regimes=pd.Series(["calm", "calm"], index=idx),
            cost_bps=zeros,
            slippage_multiplier=pd.Series([1.0, 1.0], index=idx),
            turnover=zeros,
            transaction_costs=debits,
            cost_drag=debits,
            total_cost_drag=float(debits.sum()),
        )

    monkeypatch.setattr(runner_module, "run_simulation", _fake_run)
    monkeypatch.setattr(runner, "_maybe_sample_costs", _fake_costs)
    evaluation = runner._evaluate_strategy(StrategyVariant(name="A"), _path_context())
    assert evaluation.metrics["total_cost_drag"] == pytest.approx(0.03)
    assert evaluation.diagnostic is not None
    assert "net_returns" in evaluation.diagnostic["costs"]
    assert evaluation.metric_source == "scenario_cost_net"


def test_evaluate_strategy_cost_missing_values_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    idx = pd.date_range("2024-01-31", periods=2, freq="ME")
    gross = pd.Series([0.01, np.nan], index=idx)

    def _fake_run(_config: Any, _returns: pd.DataFrame) -> RunResult:
        return RunResult(
            metrics=pd.DataFrame({"x": [1.0]}, index=["user_weight"]),
            details={"portfolio_user_weight": gross},
            seed=0,
            environment={},
        )

    monkeypatch.setattr(runner_module, "run_simulation", _fake_run)
    debits = pd.Series([0.0, 0.0], index=idx)
    monkeypatch.setattr(
        runner,
        "_maybe_sample_costs",
        lambda *_a: CostProcessOutput(
            regimes=pd.Series(["calm", "calm"], index=idx),
            cost_bps=debits,
            slippage_multiplier=pd.Series([1.0, 1.0], index=idx),
            turnover=debits,
            transaction_costs=debits,
            cost_drag=debits,
            total_cost_drag=0.0,
        ),
    )
    with pytest.raises(ValueError, match="missing values"):
        runner._evaluate_strategy(StrategyVariant(name="A"), _path_context())


def test_apply_regime_turnover_caps_resolves_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    from trend_analysis.config import Config

    runner = _runner()
    dates = pd.date_range("2024-01-31", periods=3, freq="ME")
    returns = pd.DataFrame({"Date": dates, "AssetA": [0.01, 0.02, 0.01], "proxy": [0.1, 0.2, 0.1]})
    ctx = _path_context(returns=returns)
    payload = _base_config()
    payload["portfolio"]["max_turnover"] = {"riskon": 0.15, "riskoff": 0.05}
    payload["regime"] = {
        "enabled": True,
        "proxy": "proxy",
        "lookback": 2,
        "threshold_percentile": 50,
    }
    config = Config(**payload)
    regimes = pd.Series(["riskon", "riskon", "riskon"], index=dates)

    def _fake_regimes(*_args: Any, **_kwargs: Any) -> pd.Series:
        return regimes

    monkeypatch.setattr(runner_module, "compute_regimes", _fake_regimes)
    runner._apply_regime_turnover_caps(config, ctx)
    assert config.portfolio["max_turnover"] == 0.15


def test_resolve_price_history_requires_csv_or_inline() -> None:
    scenario = MonteCarloScenario(
        name="no_hist",
        base_config="config/defaults.yml",
        monte_carlo={"mode": "two_layer", "n_paths": 1, "horizon_years": 1.0, "frequency": "M"},
        return_model={"kind": "stationary_bootstrap"},
    )
    config = _base_config()
    config["data"].pop("csv_path", None)
    runner = MonteCarloRunner(scenario, base_config=config, price_history=None)
    with pytest.raises(ValueError, match="price_history must be provided"):
        runner._resolve_price_history()


def test_resolve_sampled_strategies_draws_variants() -> None:
    runner = _runner()
    sampled = runner._resolve_sampled_strategies(
        {
            "sampled": {
                "n_strategies": 1,
                "seed": 7,
                "sampling": {
                    "portfolio.max_turnover": {"dist": "uniform", "low": 0.1, "high": 0.2},
                },
            }
        },
        existing=[],
    )
    assert len(sampled) == 1
    assert sampled[0].name == "sampled_001"
    assert sampled[0].overrides == {"portfolio": {"max_turnover": 0.13238327648331624}}


def test_resolve_sampled_strategies_invalid_seed_raises() -> None:
    runner = _runner()
    with pytest.raises(ValueError, match="seed must be an integer"):
        runner._resolve_sampled_strategies(
            {
                "sampled": {
                    "n_strategies": 1,
                    "seed": "bad",
                    "sampling": {"kind": "fixed", "variants": [{"name": "x"}]},
                }
            },
            existing=[],
        )


def test_resolve_sampled_strategies_invalid_max_rejection_attempts() -> None:
    runner = _runner()
    with pytest.raises(ValueError, match="max_rejection_attempts must be an integer"):
        runner._resolve_sampled_strategies(
            {
                "sampled": {
                    "n_strategies": 1,
                    "max_rejection_attempts": "x",
                    "sampling": {"kind": "fixed", "variants": [{"name": "x"}]},
                }
            },
            existing=[],
        )


def test_path_period_span_reads_date_column() -> None:
    runner = _runner()
    dates = pd.date_range("2024-01-31", periods=4, freq="ME")
    ctx = _path_context(returns=pd.DataFrame({"Date": dates, "AssetA": np.zeros(4)}))
    span = runner._path_period_span(ctx)
    assert span is not None
    assert span[0] == dates[0].strftime("%Y-%m")


def test_fit_vol_adjust_invalid_scalar_window_length() -> None:
    runner = _runner()
    merged = {"vol_adjust": {"enabled": True, "window": "nope"}}
    runner._fit_vol_adjust_window(merged, 3)
    assert merged["vol_adjust"]["window"] == "nope"


def test_fit_signal_window_invalid_window_skips() -> None:
    runner = _runner()
    merged = {"signals": {"window": "nope", "min_periods": 2}}
    runner._fit_signal_window(merged, 3)
    assert merged["signals"]["window"] == "nope"


def test_apply_strategy_guards_skips_none_guard() -> None:
    runner = _runner(strategy_set={"guards": {"max_turnover": None}})
    merged: dict[str, Any] = {"portfolio": {}}
    runner._apply_strategy_guards(merged)
    assert "max_turnover" not in merged["portfolio"]


def test_apply_turnover_guard_distribution_skips_when_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(
        strategy_set={
            "guards": {"max_turnover": {"dist": "uniform", "low": 0.1, "high": 0.2}},
        }
    )
    merged: dict[str, Any] = {"portfolio": {}}
    strategy = StrategyVariant(name="s", overrides={"portfolio": {"max_turnover": 0.5}})
    runner._apply_turnover_guard_distribution(merged, strategy, seed=1)
    assert "max_turnover" not in merged["portfolio"]


def test_stats_to_metrics_reads_mapping() -> None:
    metrics = MonteCarloRunner._stats_to_metrics({"cagr": 0.1, "note": "x"})
    assert metrics == {"cagr": 0.1}


def test_extract_nav_series_invalid_starting_nav(caplog: pytest.LogCaptureFixture) -> None:
    runner = _runner()
    idx = pd.date_range("2024-01-31", periods=2, freq="ME")
    returns = pd.Series([-1.0, 0.0], index=idx)
    run_result = SimpleNamespace(details={})
    out = runner._extract_nav_series(run_result, _path_context(), returns=returns)
    assert out is None


def test_coerce_nav_index_uses_cost_index_fallback() -> None:
    runner = _runner()
    idx = pd.Index(["not-a-date", "also-bad"])
    run_result = SimpleNamespace(
        details={
            "out_sample_scaled": pd.DataFrame(
                {"x": [1.0, 2.0]},
                index=pd.date_range("2024-01-31", periods=2, freq="ME"),
            )
        }
    )
    coerced = runner._coerce_nav_index(idx, run_result, _path_context(), 2)
    assert isinstance(coerced, pd.DatetimeIndex)


def test_aggregator_coerce_quantiles_percent_value() -> None:
    assert aggregator_module._coerce_quantiles(["25%"]) == [0.25]


def test_aggregator_coerce_quantiles_skips_non_finite() -> None:
    assert aggregator_module._coerce_quantiles([float("nan"), 0.4]) == [0.4]


def test_aggregator_build_breach_accepts_percent_threshold_string() -> None:
    frame = pd.DataFrame(
        [
            {"strategy": "A", "fold_id": 0, "path": 1, "metric": 0.2},
            {"strategy": "A", "fold_id": 0, "path": 2, "metric": 0.3},
        ]
    )
    breach = build_breach_frame(frame, {"metric": ["10%"]})
    assert breach.loc[0, "threshold"] == pytest.approx(0.1)


def test_aggregator_shortfall_scalar_invalid_returns_empty_frame() -> None:
    frame = pd.DataFrame(
        [{"strategy": "A", "fold_id": 0, "path": 1, "metric": 1.0}],
    )
    shortfall = build_expected_shortfall_frame(frame, "not-a-number")
    assert shortfall.empty


def test_aggregator_ensure_path_columns_fills_missing_with_na() -> None:
    frame = pd.DataFrame({"metric": [1.0, 2.0]})
    out = aggregator_module._ensure_path_columns(frame)
    assert list(out["metric"]) == [1.0, 2.0]
    assert pd.isna(out.loc[0, "strategy"])
    assert pd.isna(out.loc[0, "fold_id"])
    assert pd.isna(out.loc[0, "path"])
    assert pd.isna(out.loc[0, "path_id"])
