import time

import pandas as pd
import pytest

from trend_analysis.monte_carlo.cache import PathContext, PathContextCache
from trend_analysis.monte_carlo.runner import evaluate_strategies_for_path


def _evaluate_strategies_naively(
    rebalance_dates: list[str],
    compute_score_frame,
    strategies: dict[str, callable],
    *,
    columns_by_strategy: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    results: dict[str, object] = {}
    for name, strategy in strategies.items():
        columns = columns_by_strategy.get(name) if columns_by_strategy else None
        frames: dict[str, pd.DataFrame] = {}
        for date in rebalance_dates:
            frame = compute_score_frame(date)
            if columns:
                keep = [col for col in columns if col in frame.columns]
                frame = frame.loc[:, keep].copy()
            frames[date] = frame
        results[name] = strategy(frames)
    return results


def test_path_context_get_or_compute_caches_once() -> None:
    context = PathContext(path_id="path-1")
    calls: list[str] = []

    def compute() -> pd.DataFrame:
        calls.append("called")
        return pd.DataFrame({"Sharpe": [1.0, 2.0]}, index=["A", "B"])

    first = context.get_or_compute("2024-01-31", compute)
    second = context.get_or_compute("2024-01-31", compute)

    assert calls == ["called"]
    assert first is second
    assert context.has_score_frame("2024-01-31")


def test_path_context_select_columns_returns_copy() -> None:
    context = PathContext(path_id="path-2")
    frame = pd.DataFrame({"Sharpe": [1.0, 2.0], "Volatility": [0.1, 0.2]}, index=["A", "B"])
    context.set_score_frame("2024-01-31", frame)

    filtered = context.select_columns("2024-01-31", ["Sharpe"])
    filtered.loc[:, "Sharpe"] = 99.0

    cached = context.get_score_frame("2024-01-31")
    assert cached is not None
    assert cached["Sharpe"].iloc[0] == 1.0


def test_cache_reuses_score_frames_across_strategies_and_clears() -> None:
    cache = PathContextCache()
    calls: list[str] = []

    def compute_score_frame(date: str) -> pd.DataFrame:
        calls.append(date)
        return pd.DataFrame(
            {
                "Sharpe": [1.0, 2.0],
                "Volatility": [0.1, 0.2],
            },
            index=["A", "B"],
        )

    def sharpe_strategy(frames: dict[str, pd.DataFrame]) -> dict[str, float]:
        for frame in frames.values():
            assert set(frame.columns) == {"Sharpe"}
        return {date: float(frame["Sharpe"].sum()) for date, frame in frames.items()}

    def vol_strategy(frames: dict[str, pd.DataFrame]) -> dict[str, float]:
        for frame in frames.values():
            assert set(frame.columns) == {"Volatility"}
        return {date: float(frame["Volatility"].sum()) for date, frame in frames.items()}

    rebalance_dates = ["2024-01-31", "2024-02-29"]
    results = evaluate_strategies_for_path(
        "path-42",
        rebalance_dates,
        compute_score_frame,
        {"sharpe": sharpe_strategy, "vol": vol_strategy},
        columns_by_strategy={"sharpe": ["Sharpe"], "vol": ["Volatility"]},
        cache=cache,
    )

    assert calls == rebalance_dates
    assert results["sharpe"]["2024-01-31"] == 3.0
    assert results["vol"]["2024-02-29"] == pytest.approx(0.3)
    assert not cache.has_path("path-42")


def test_cache_computes_once_per_date_with_many_strategies() -> None:
    cache = PathContextCache()
    calls: list[str] = []

    def compute_score_frame(date: str) -> pd.DataFrame:
        calls.append(date)
        return pd.DataFrame(
            {
                "Sharpe": [1.0, 2.0],
                "Volatility": [0.1, 0.2],
                "Return": [0.05, 0.07],
            },
            index=["A", "B"],
        )

    def make_strategy(column: str, idx: int) -> tuple[str, callable]:
        def strategy(frames: dict[str, pd.DataFrame]) -> float:
            return float(sum(frame[column].sum() for frame in frames.values()))

        return f"{column}-strategy-{idx}", strategy

    columns = [
        "Sharpe",
        "Volatility",
        "Return",
        "Sharpe",
        "Volatility",
        "Return",
        "Sharpe",
        "Volatility",
        "Return",
        "Sharpe",
    ]
    strategies = dict(
        [make_strategy(column, idx) for idx, column in enumerate(columns, start=1)],
    )
    assert len(strategies) == 10
    columns_by_strategy = {name: [name.split("-")[0]] for name in strategies}

    rebalance_dates = ["2024-01-31", "2024-02-29", "2024-03-29"]
    results = evaluate_strategies_for_path(
        "path-99",
        rebalance_dates,
        compute_score_frame,
        strategies,
        columns_by_strategy=columns_by_strategy,
        cache=cache,
    )

    assert calls == rebalance_dates
    assert len(results) == 10
    assert not cache.has_path("path-99")


def test_cache_compute_score_frames_prefills_once() -> None:
    cache = PathContextCache()
    calls: list[str] = []

    def compute_score_frame(date: str) -> pd.DataFrame:
        calls.append(date)
        return pd.DataFrame({"Sharpe": [1.0]}, index=["A"])

    rebalance_dates = ["2024-01-31", "2024-02-29"]
    cache.compute_score_frames("path-prefill", rebalance_dates, compute_score_frame)
    cache.compute_score_frames("path-prefill", ["2024-01-31"], compute_score_frame)

    assert calls == rebalance_dates
    assert cache.get_score_frame("path-prefill", "2024-02-29") is not None


def test_cached_results_match_naive_evaluation() -> None:
    cache = PathContextCache()

    def compute_score_frame(date: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Sharpe": [1.0, 2.0],
                "Volatility": [0.1, 0.2],
                "Return": [0.05, 0.07],
            },
            index=["A", "B"],
        )

    def sharpe_strategy(frames: dict[str, pd.DataFrame]) -> float:
        return float(sum(frame["Sharpe"].sum() for frame in frames.values()))

    def vol_strategy(frames: dict[str, pd.DataFrame]) -> float:
        return float(sum(frame["Volatility"].sum() for frame in frames.values()))

    rebalance_dates = ["2024-01-31", "2024-02-29", "2024-03-29"]
    strategies = {"sharpe": sharpe_strategy, "vol": vol_strategy}
    columns_by_strategy = {"sharpe": ["Sharpe"], "vol": ["Volatility"]}

    cached_results = evaluate_strategies_for_path(
        "path-compare",
        rebalance_dates,
        compute_score_frame,
        strategies,
        columns_by_strategy=columns_by_strategy,
        cache=cache,
    )
    naive_results = _evaluate_strategies_naively(
        rebalance_dates,
        compute_score_frame,
        strategies,
        columns_by_strategy=columns_by_strategy,
    )

    assert cached_results == naive_results
    assert not cache.has_path("path-compare")


def test_cached_results_match_naive_without_column_filter() -> None:
    cache = PathContextCache()

    def compute_score_frame(date: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Sharpe": [1.0, 2.0],
                "Volatility": [0.1, 0.2],
                "Return": [0.05, 0.07],
                "Noise": [0.9, 0.8],
            },
            index=["A", "B"],
        )

    def combined_strategy(frames: dict[str, pd.DataFrame]) -> float:
        return float(sum(frame[["Sharpe", "Return"]].sum().sum() for frame in frames.values()))

    rebalance_dates = ["2024-01-31", "2024-02-29", "2024-03-29"]
    strategies = {"combined": combined_strategy}

    cached_results = evaluate_strategies_for_path(
        "path-compare-no-filter",
        rebalance_dates,
        compute_score_frame,
        strategies,
        cache=cache,
    )
    naive_results = _evaluate_strategies_naively(
        rebalance_dates,
        compute_score_frame,
        strategies,
    )

    assert cached_results == naive_results
    assert not cache.has_path("path-compare-no-filter")


@pytest.mark.performance
@pytest.mark.runtime
def test_cached_strategies_are_materially_faster() -> None:
    base_frame = pd.DataFrame(
        {
            "Sharpe": [1.0, 2.0],
            "Volatility": [0.1, 0.2],
            "Return": [0.05, 0.07],
        },
        index=["A", "B"],
    )

    def make_compute(delay: float, calls: list[str]):
        def compute_score_frame(date: str) -> pd.DataFrame:  # noqa: ARG001
            calls.append(date)
            time.sleep(delay)
            return base_frame.copy()

        return compute_score_frame

    def make_strategy(column: str, idx: int) -> tuple[str, callable]:
        def strategy(frames: dict[str, pd.DataFrame]) -> float:
            return float(sum(frame[column].sum() for frame in frames.values()))

        return f"{column}-strategy-{idx}", strategy

    columns = [
        "Sharpe",
        "Volatility",
        "Return",
        "Sharpe",
        "Volatility",
        "Return",
        "Sharpe",
        "Volatility",
        "Return",
        "Sharpe",
    ]
    strategies = dict(
        [make_strategy(column, idx) for idx, column in enumerate(columns, start=1)],
    )
    columns_by_strategy = {name: [name.split("-")[0]] for name in strategies}
    rebalance_dates = ["2024-01-31", "2024-02-29", "2024-03-29"]

    # Use a noticeable delay so compute cost dominates strategy overhead and
    # timing assertions remain stable across machines (40ms per frame).
    delay = 0.04
    cached_calls: list[str] = []
    naive_calls: list[str] = []
    cached_compute = make_compute(delay, cached_calls)
    naive_compute = make_compute(delay, naive_calls)

    # Warm up to avoid one-time overhead skewing the timing ratio assertions.
    single_strategy_name = next(iter(strategies))
    single_strategy = {single_strategy_name: strategies[single_strategy_name]}
    single_columns = {single_strategy_name: columns_by_strategy[single_strategy_name]}
    evaluate_strategies_for_path(
        "path-perf-warmup",
        rebalance_dates,
        make_compute(0.0, []),
        single_strategy,
        columns_by_strategy=single_columns,
        cache=PathContextCache(),
    )

    start = time.perf_counter()
    evaluate_strategies_for_path(
        "path-perf",
        rebalance_dates,
        cached_compute,
        strategies,
        columns_by_strategy=columns_by_strategy,
        cache=PathContextCache(),
    )
    cached_time = time.perf_counter() - start

    start = time.perf_counter()
    _evaluate_strategies_naively(
        rebalance_dates,
        naive_compute,
        strategies,
        columns_by_strategy=columns_by_strategy,
    )
    naive_time = time.perf_counter() - start

    assert cached_calls == rebalance_dates
    assert len(naive_calls) == len(rebalance_dates) * len(strategies)
    assert cached_time * 2 < naive_time


def _real_runner(strategy_count=1, *, multi_period=False):
    from tests.monte_carlo.test_runner import _base_config, _price_history, _scenario
    from trend_analysis.monte_carlo.runner import MonteCarloRunner
    from trend_analysis.monte_carlo.strategy import StrategyVariant

    scenario = _scenario("two_layer")
    scenario.monte_carlo.n_paths = 1
    scenario.strategy_set["curated"] = [
        StrategyVariant(name=f"same-{number}") for number in range(strategy_count)
    ]
    config = _base_config()
    if multi_period:
        config["multi_period"] = {
            "frequency": "M",
            "in_sample_len": 6,
            "out_sample_len": 1,
            "start": "2023-01",
            "end": "2023-12",
        }
    return MonteCarloRunner(scenario, base_config=config, price_history=_price_history())


@pytest.mark.parametrize("multi_period", [False, True])
def test_runner_kernel_calls_do_not_grow_with_strategies(monkeypatch, multi_period):
    from unittest.mock import Mock
    from trend_analysis.core import rank_selection

    spy = Mock(wraps=rank_selection._compute_metric_series)
    monkeypatch.setattr(rank_selection, "_compute_metric_series", spy)
    one = _real_runner(1, multi_period=multi_period).run(jobs=1)
    first = spy.call_count
    spy.reset_mock()
    three = _real_runner(3, multi_period=multi_period).run(jobs=1)
    assert not one.errors and not three.errors
    assert len(three.results_frame) == 3
    assert first > 0
    assert spy.call_count == first


@pytest.mark.parametrize("multi_period", [False, True])
def test_cached_and_uncached_runs_match(monkeypatch, multi_period):
    from contextlib import nullcontext
    import trend_analysis.monte_carlo.runner as runner_module

    captured = []
    original = runner_module.run_simulation

    def capture(config, returns):
        result = original(config, returns)
        captured.append(result)
        return result

    monkeypatch.setattr(runner_module, "run_simulation", capture)
    cached = _real_runner(3, multi_period=multi_period).run(jobs=1)
    cached_runs = captured[:]
    captured.clear()
    monkeypatch.setattr(runner_module, "use_metric_cache", lambda cache: nullcontext())
    uncached = _real_runner(3, multi_period=multi_period).run(jobs=1)
    assert not cached.errors and not uncached.errors
    assert len(cached_runs) == len(captured) == 3
    pd.testing.assert_frame_equal(cached.results_frame, uncached.results_frame)
    for left, right in zip(cached_runs, captured, strict=True):
        pd.testing.assert_frame_equal(left.metrics, right.metrics)
        left_periods = left.details.get("period_results") or [left.details]
        right_periods = right.details.get("period_results") or [right.details]
        assert len(left_periods) == len(right_periods) > 0
        for a_period, b_period in zip(left_periods, right_periods, strict=True):
            assert a_period["selected_funds"]
            assert a_period["fund_weights"]
            for key in [
                "selected_funds",
                "fund_weights",
                "portfolio_equal_weight",
                "portfolio_user_weight",
            ]:
                a, b = a_period[key], b_period[key]
                if isinstance(a, pd.Series):
                    pd.testing.assert_series_equal(a, b)
                else:
                    assert a == b


def test_incompatible_windows_do_not_share_cache():
    from trend_analysis.core.path_metrics import use_metric_cache
    from trend_analysis.core.rank_selection import RiskStatsConfig
    from trend_analysis.stages.selection import single_period_run

    frame = pd.DataFrame(
        {"Date": pd.date_range("2024-01-31", periods=4, freq="ME"), "A": [0.01, -0.02, 0.03, 0.04]}
    )
    cache = {}
    cfg = RiskStatsConfig(metrics_to_run=["Sharpe"])
    with use_metric_cache(cache):
        first = single_period_run(frame, "2024-01", "2024-03", stats_cfg=cfg, risk_free=0.0)
        assert len(cache) == 1
        first.iloc[0, 0] = -999.0
        repeated = single_period_run(frame, "2024-01", "2024-03", stats_cfg=cfg, risk_free=0.0)
        assert repeated.iloc[0, 0] != -999.0
        assert len(cache) == 1
        single_period_run(frame, "2024-02", "2024-04", stats_cfg=cfg, risk_free=0.0)
        single_period_run(frame, "2024-01", "2024-03", stats_cfg=cfg, risk_free=0.01)
        changed = frame.copy()
        changed.loc[0, "A"] = 0.5
        single_period_run(changed, "2024-01", "2024-03", stats_cfg=cfg, risk_free=0.0)
        assert len(cache) == 4


def test_runner_releases_path_entries(monkeypatch):
    runner = _real_runner(3)
    contexts = []
    generate = runner._generate_path_context

    def record(**kwargs):
        context = generate(**kwargs)
        contexts.append(context)
        return context

    monkeypatch.setattr(runner, "_generate_path_context", record)
    result = runner.run(jobs=1)
    assert not result.errors
    assert contexts and all(not context.metric_cache for context in contexts)


@pytest.mark.parametrize("fail", [False, True])
def test_distinct_paths_and_folds_have_separate_released_caches(monkeypatch, fail):
    from tests.monte_carlo.test_runner import _base_config, _price_history, _scenario_with_folds
    from trend_analysis.monte_carlo.runner import MonteCarloRunner

    scenario = _scenario_with_folds(
        mode="two_layer",
        folds={
            "mode": "rolling",
            "start": "2021-03-15",
            "end": "2021-09-30",
            "step_months": 3,
            "calibration_lookback_years": 1.0,
        },
        enable_fold_runs=True,
    )
    runner = MonteCarloRunner(scenario, base_config=_base_config(), price_history=_price_history())
    contexts = []
    generate = runner._generate_path_context

    def record(**kwargs):
        context = generate(**kwargs)
        contexts.append(context)
        return context

    monkeypatch.setattr(runner, "_generate_path_context", record)
    if fail:

        def fail_strategy(*args, **kwargs):
            raise ValueError("synthetic evaluation failure")

        monkeypatch.setattr(runner, "_evaluate_strategy", fail_strategy)
    result = runner.run(jobs=1)
    assert bool(result.errors) == fail
    assert len(contexts) > 2
    assert len({context.fold_id for context in contexts}) > 1
    assert len({id(context.metric_cache) for context in contexts}) == len(contexts)
    assert all(not context.metric_cache for context in contexts)


def test_metric_union_and_effective_inputs_are_keyed(monkeypatch):
    from unittest.mock import Mock
    from trend_analysis.core import rank_selection
    from trend_analysis.core.path_metrics import compute_metric_series, use_metric_cache
    from trend_analysis.core.rank_selection import RiskStatsConfig

    frame = pd.DataFrame(
        {"A": [0.01, -0.02, 0.03, 0.04]}, index=pd.date_range("2024-01-31", periods=4, freq="ME")
    )
    spy = Mock(wraps=rank_selection._compute_metric_series)
    monkeypatch.setattr(rank_selection, "_compute_metric_series", spy)
    cache = {}
    with use_metric_cache(cache):
        for metrics in [["Sharpe"], ["Sharpe", "Volatility"]]:
            cfg = RiskStatsConfig(metrics_to_run=metrics)
            for metric in metrics:
                compute_metric_series(frame, metric, cfg, risk_free_override=0.0)
        assert spy.call_count == 2
        compute_metric_series(
            frame, "Sharpe", RiskStatsConfig(periods_per_year=4), risk_free_override=0.0
        )
        compute_metric_series(
            frame, "Sharpe", cfg, risk_free_override=pd.Series(0.0, index=frame.index)
        )
        compute_metric_series(frame, "InformationRatio", cfg, benchmark=frame["A"])
        compute_metric_series(frame, "InformationRatio", cfg, benchmark=frame["A"] * 0.5)
        assert spy.call_count == 6
    # Scope restoration means normal API calls do not reuse path-owned values.
    compute_metric_series(frame, "Sharpe", cfg, risk_free_override=0.0)
    assert spy.call_count == 7
