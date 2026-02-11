"""Monte Carlo runner for evaluating strategy variants.

Canonical ``nav_paths`` schema (used by Monte Carlo reporting/export):

- Type: ``pandas.DataFrame`` with one column per simulated path.
- Index: datetime-like period-end labels (``DatetimeIndex`` preferred). Each row
  represents a simulation period end date and should be aligned across paths.
- Values: floating-point NAV values normalized to start at ``1.0`` for every
  path (first row equals 1.0 per column).
- Columns: either plain path identifiers (e.g. ``0, 1, 2, ...``) or a
  ``MultiIndex`` with levels ``path`` and ``asset``.
  - ``path``: the integer path identifier.
  - ``asset``: the asset label. For NAV paths this must be ``"NAV"`` (non-NAV
    labels are coerced to ``"NAV"`` during normalization).

Example (plain columns):
```
index = DatetimeIndex(["2024-01-31", "2024-02-29"], name="period")
columns = Index([0, 1], name="path")
```

Example (MultiIndex columns):
```
columns = MultiIndex.from_product([[0, 1], ["NAV"]], names=["path", "asset"])
```
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, SupportsFloat, cast

import numpy as np
import pandas as pd

from trend_analysis.api import run_simulation
from trend_analysis.config.models import Config, ConfigType
from trend_analysis.constants import NUMERICAL_TOLERANCE_LOW
from trend_analysis.core.rank_selection import RiskStatsConfig, canonical_metric_list
from trend_analysis.io.market_data import (
    MarketDataMode,
    load_market_data_csv,
    load_market_data_parquet,
)
from trend_analysis.monte_carlo.config import resolve_risk_free_source
from trend_analysis.monte_carlo.costs import CostProcess, CostProcessOutput
from trend_analysis.monte_carlo.models import (
    RegimeConditionedBootstrapModel,
    StationaryBootstrapModel,
)
from trend_analysis.monte_carlo.models.base import normalize_price_frequency
from trend_analysis.monte_carlo.scenario import MonteCarloScenario, MonteCarloSettings
from trend_analysis.monte_carlo.seed import SeedManager
from trend_analysis.monte_carlo.strategy import StrategyVariant
from trend_analysis.monte_carlo.strategy.sampler import (
    parse_distribution,
    sample_strategy_variants,
)
from trend_analysis.pipeline import _resolve_sample_split
from trend_analysis.pipeline_helpers import parse_regime_turnover_caps
from trend_analysis.regime_utils import alias_regime_key, normalize_regime_key
from trend_analysis.regimes import compute_regimes, normalise_settings
from trend_analysis.risk import periods_per_year_from_code
from trend_analysis.stages.selection import single_period_run

from .aggregator import aggregate_monte_carlo_results
from .cache import PathContextCache
from .export import export_aggregation_results
from .folds import Fold, FoldGenerator
from .results import (
    MonteCarloPathError,
    MonteCarloResults,
    StrategyEvaluation,
    build_cross_fold_summary_frame,
    build_diagnostics_frame,
    build_pooled_distribution_frame,
    build_pooled_summary_frame,
    build_results_frame,
    build_summary_frame,
    export_results,
)

__all__ = ["MonteCarloRunner", "evaluate_strategies_for_path"]

_STRATEGY_SELECTION_SEED_TAG = "__strategy_selection__"
_TURNOVER_GUARD_PATH = "strategy_set.guards.max_turnover"
# nav_paths schema requirements:
# - DataFrame index: datetime-like period ends (DatetimeIndex preferred).
# - Values: float NAVs normalized to start at 1.0 for each path.
# - Columns: plain path ids or MultiIndex with levels (path, asset="NAV").
#   When using a MultiIndex, the path id lives in the "path" level.


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _coerce_optional_bool(value: Any, field: str) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field} must be a boolean")


def _coerce_turnover_guard(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{_TURNOVER_GUARD_PATH} must be numeric or a distribution mapping")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{_TURNOVER_GUARD_PATH} must be numeric or a distribution mapping"
        ) from exc


def _is_distribution_mapping(value: Mapping[str, Any]) -> bool:
    dist = value.get("dist")
    return isinstance(dist, str) and bool(dist.strip())


def _has_turnover_override(variant: StrategyVariant) -> bool:
    overrides = variant.overrides
    if not isinstance(overrides, Mapping):
        return False
    portfolio = overrides.get("portfolio")
    if not isinstance(portfolio, Mapping):
        return False
    return "max_turnover" in portfolio


@dataclass(frozen=True)
class _PathContext:
    path_id: int
    prices: pd.DataFrame
    returns: pd.DataFrame
    score_frame: pd.DataFrame
    path_hash: str
    seed: int | None
    fold_id: int | None = None
    fold_label: str | None = None


def evaluate_strategies_for_path(
    path_id: str,
    rebalance_dates: Sequence[str],
    compute_score_frame: Callable[[str], pd.DataFrame],
    strategies: Mapping[str, Callable[[dict[str, pd.DataFrame]], object]],
    *,
    columns_by_strategy: Mapping[str, Sequence[str]] | None = None,
    cache: PathContextCache | None = None,
) -> dict[str, object]:
    """Compute strategy results for a single path using cached score frames."""
    context_cache = cache or PathContextCache()
    results: dict[str, object] = {}
    columns_lookup = columns_by_strategy or {}
    try:
        context_cache.compute_score_frames(path_id, rebalance_dates, compute_score_frame)
        base_frames = {
            date: context_cache.select_score_frame(path_id, date, None) for date in rebalance_dates
        }
        for name, strategy in strategies.items():
            columns = columns_lookup.get(name)
            if columns:
                frames = {
                    date: context_cache.select_score_frame(path_id, date, columns)
                    for date in rebalance_dates
                }
            else:
                frames = dict(base_frames)
            results[name] = strategy(frames)
    finally:
        context_cache.clear(path_id)
    return results


class MonteCarloRunner:
    """Run Monte Carlo path simulations across strategy variants."""

    def __init__(
        self,
        scenario: MonteCarloScenario,
        *,
        base_config: Mapping[str, Any] | None = None,
        price_history: pd.DataFrame | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.scenario = scenario
        self._base_config = self._coerce_base_config(base_config)
        self._price_history = price_history
        self._logger = logger or logging.getLogger("trend_analysis.monte_carlo")
        self._cash_injection_warned = False
        self._seed_manager: SeedManager | None = None
        self._seed_manager_init = False
        self._cost_process: CostProcess | None = None
        self._cost_process_init = False
        self._regime_cache: dict[tuple[object, ...], pd.Series] = {}

    def run(
        self,
        *,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        jobs: int | None = None,
    ) -> MonteCarloResults:
        """Run the Monte Carlo simulation for the configured scenario."""

        settings = self._settings()
        strategies = self._resolve_strategies()
        history = self._resolve_price_history()
        folds = self._resolve_folds(history)
        n_periods = self._compute_n_periods()
        path_seeds, strategy_seeds = self._build_seeds()
        worker_count = self._resolve_jobs(jobs)

        mode = settings.mode
        assert mode is not None
        evaluations: list[StrategyEvaluation] = []
        errors: list[MonteCarloPathError] = []
        primary_strategy = None
        if mode != "mixture" and strategies:
            primary_strategy = strategies[0].name
        if folds:
            for fold in folds:
                model = self._build_price_model(
                    history,
                    calibration_start=fold.calibration_start,
                    calibration_end=fold.calibration_end,
                )
                fold_evals, fold_errors = self._run_mode(
                    mode=mode,
                    model=model,
                    n_periods=n_periods,
                    strategies=strategies,
                    path_seeds=path_seeds,
                    strategy_seeds=strategy_seeds,
                    progress_callback=progress_callback,
                    jobs=worker_count,
                    fold_id=fold.fold_id,
                    fold_label=fold.label,
                )
                evaluations.extend(fold_evals)
                errors.extend(fold_errors)
        else:
            model = self._build_price_model(history)
            evaluations, errors = self._run_mode(
                mode=mode,
                model=model,
                n_periods=n_periods,
                strategies=strategies,
                path_seeds=path_seeds,
                strategy_seeds=strategy_seeds,
                progress_callback=progress_callback,
                jobs=worker_count,
                fold_id=None,
                fold_label=None,
            )

        results_frame = build_results_frame(evaluations)
        summary_frame = build_summary_frame(results_frame)
        diagnostics_frame = build_diagnostics_frame(evaluations)
        nav_paths_by_fold: dict[int, pd.DataFrame] | None = None
        nav_paths: pd.DataFrame | None = None
        if folds:
            nav_paths_by_fold = {}
            for fold in folds:
                fold_nav = self._build_nav_paths_frame(
                    evaluations,
                    primary_strategy_name=primary_strategy,
                    fold_id=fold.fold_id,
                )
                nav_paths_by_fold[int(fold.fold_id)] = fold_nav
        else:
            nav_paths = self._build_nav_paths_frame(
                evaluations,
                primary_strategy_name=primary_strategy,
                fold_id=None,
            )
        metadata: dict[str, Any] = {
            "scenario": self.scenario.name,
            "mode": mode,
            "n_paths": settings.n_paths,
            "n_strategies": len(strategies),
            "seed": settings.seed,
        }
        if nav_paths is not None:
            metadata["nav_paths"] = nav_paths
        if nav_paths_by_fold is not None:
            metadata["nav_paths_by_fold"] = nav_paths_by_fold
        if folds:
            metadata["n_folds"] = len(folds)
            metadata["folds"] = [fold.as_dict() for fold in folds]
        outputs = self.scenario.outputs or {}
        pooled_distributions = False
        if isinstance(outputs, Mapping):
            pooled_distributions = _coerce_optional_bool(
                outputs.get("pooled_distributions"),
                "outputs.pooled_distributions",
            )
        cross_fold_summary_frame = build_cross_fold_summary_frame(results_frame) if folds else None
        pooled_summary_frame = (
            build_pooled_summary_frame(results_frame) if folds and pooled_distributions else None
        )
        pooled_distribution_frame = (
            build_pooled_distribution_frame(results_frame)
            if folds and pooled_distributions
            else None
        )
        if folds:
            metadata["pooled_distributions"] = pooled_distributions
            if pooled_distributions:
                metadata["pooled_scope"] = "summary+distribution"
                metadata["pooled_outputs"] = ["summary", "distribution"]
            else:
                metadata["pooled_scope"] = "none"
                metadata["pooled_outputs"] = []
        results = MonteCarloResults(
            mode=mode,
            evaluations=evaluations,
            errors=errors,
            results_frame=results_frame,
            summary_frame=summary_frame,
            diagnostics_frame=diagnostics_frame,
            cross_fold_summary_frame=cross_fold_summary_frame,
            pooled_summary_frame=pooled_summary_frame,
            pooled_distribution_frame=pooled_distribution_frame,
            metadata=metadata,
        )
        self._maybe_export(results)
        return results

    @property
    def base_config(self) -> Mapping[str, Any]:
        """Return the resolved base configuration used by the runner.

        This is intended for validation and diagnostics; treat the returned
        mapping as read-only to avoid mutating the runner's internal state.
        """

        return self._base_config

    def resolve_strategies(self) -> list[StrategyVariant]:
        """Return the resolved strategy variants for the scenario."""

        return self._resolve_strategies()

    def _run_mode(
        self,
        *,
        mode: str,
        model: Any,
        n_periods: int,
        strategies: Sequence[StrategyVariant],
        path_seeds: Sequence[int | None],
        strategy_seeds: Sequence[int | None],
        progress_callback: Callable[[Mapping[str, Any]], None] | None,
        jobs: int,
        fold_id: int | None,
        fold_label: str | None,
    ) -> tuple[list[StrategyEvaluation], list[MonteCarloPathError]]:
        if mode == "two_layer":
            return self._run_two_layer(
                model=model,
                n_periods=n_periods,
                strategies=strategies,
                path_seeds=path_seeds,
                progress_callback=progress_callback,
                jobs=jobs,
                fold_id=fold_id,
                fold_label=fold_label,
            )
        if mode == "mixture":
            return self._run_mixture(
                model=model,
                n_periods=n_periods,
                strategies=strategies,
                path_seeds=path_seeds,
                strategy_seeds=strategy_seeds,
                progress_callback=progress_callback,
                jobs=jobs,
                fold_id=fold_id,
                fold_label=fold_label,
            )
        raise ValueError(f"Unsupported Monte Carlo mode '{mode}'")

    def _run_two_layer(
        self,
        *,
        model: Any,
        n_periods: int,
        strategies: Sequence[StrategyVariant],
        path_seeds: Sequence[int | None],
        progress_callback: Callable[[Mapping[str, Any]], None] | None,
        jobs: int,
        fold_id: int | None = None,
        fold_label: str | None = None,
    ) -> tuple[list[StrategyEvaluation], list[MonteCarloPathError]]:
        total = len(path_seeds)
        evaluations: list[StrategyEvaluation] = []
        errors: list[MonteCarloPathError] = []
        base_seed = self._settings().seed
        seed_manager = SeedManager(base_seed) if base_seed is not None else None
        seeds_match_base = False
        if seed_manager is not None:
            seeds_match_base = all(
                seed == seed_manager.get_path_seed(path_id)
                for path_id, seed in enumerate(path_seeds)
                if seed is not None
            )
        shared_paths = seeds_match_base or all(seed is None for seed in path_seeds)

        path_result = None
        if shared_paths:
            try:
                path_result = model.sample_prices(
                    n_periods=n_periods,
                    n_paths=total,
                    frequency=self.scenario.simulation_frequency(),
                    seed=base_seed,
                )
            except Exception as exc:
                for path_id in range(total):
                    self._log_path_error(path_id, None, exc, fold_id=fold_id, fold_label=fold_label)
                    errors.append(
                        self._error_record(
                            path_id, None, exc, fold_id=fold_id, fold_label=fold_label
                        )
                    )
                return evaluations, errors

        def _evaluate_path(
            path_id: int, seed: int | None
        ) -> tuple[list[StrategyEvaluation], list[MonteCarloPathError]]:
            try:
                context = self._generate_path_context(
                    path_id=path_id,
                    seed=seed,
                    model=model,
                    n_periods=n_periods,
                    path_result=path_result,
                    path_index=path_id,
                    fold_id=fold_id,
                    fold_label=fold_label,
                )
            except Exception as exc:
                self._log_path_error(path_id, None, exc, fold_id=fold_id, fold_label=fold_label)
                return [], [
                    self._error_record(path_id, None, exc, fold_id=fold_id, fold_label=fold_label)
                ]

            path_evals: list[StrategyEvaluation] = []
            path_errors: list[MonteCarloPathError] = []
            for strategy in strategies:
                try:
                    evaluation = self._evaluate_strategy(strategy, context)
                    path_evals.append(evaluation)
                except Exception as exc:
                    self._log_path_error(
                        path_id,
                        strategy.name,
                        exc,
                        fold_id=fold_id,
                        fold_label=fold_label,
                    )
                    path_errors.append(
                        self._error_record(
                            path_id,
                            strategy.name,
                            exc,
                            fold_id=fold_id,
                            fold_label=fold_label,
                        )
                    )
            return path_evals, path_errors

        completed = 0
        for path_id, path_eval, path_err in self._execute_paths(path_seeds, _evaluate_path, jobs):
            evaluations.extend(path_eval)
            errors.extend(path_err)
            completed += 1
            self._emit_progress(progress_callback, completed, total, path_id, "two_layer")

        return evaluations, errors

    def _run_mixture(
        self,
        *,
        model: Any,
        n_periods: int,
        strategies: Sequence[StrategyVariant],
        path_seeds: Sequence[int | None],
        strategy_seeds: Sequence[int | None],
        progress_callback: Callable[[Mapping[str, Any]], None] | None,
        jobs: int,
        fold_id: int | None = None,
        fold_label: str | None = None,
    ) -> tuple[list[StrategyEvaluation], list[MonteCarloPathError]]:
        if len(strategy_seeds) != len(path_seeds):
            raise ValueError("strategy_seeds must align with path_seeds")
        total = len(path_seeds)
        evaluations: list[StrategyEvaluation] = []
        errors: list[MonteCarloPathError] = []
        base_seed = self._settings().seed
        seed_manager = SeedManager(base_seed) if base_seed is not None else None
        seeds_match_base = False
        if seed_manager is not None:
            seeds_match_base = all(
                seed == seed_manager.get_path_seed(path_id)
                for path_id, seed in enumerate(path_seeds)
                if seed is not None
            )
        shared_paths = seeds_match_base or all(seed is None for seed in path_seeds)

        path_result = None
        if shared_paths:
            try:
                path_result = model.sample_prices(
                    n_periods=n_periods,
                    n_paths=total,
                    frequency=self.scenario.simulation_frequency(),
                    seed=base_seed,
                )
            except Exception as exc:
                for path_id in range(total):
                    self._log_path_error(path_id, None, exc, fold_id=fold_id, fold_label=fold_label)
                    errors.append(
                        self._error_record(
                            path_id, None, exc, fold_id=fold_id, fold_label=fold_label
                        )
                    )
                return evaluations, errors
        else:
            self._logger.debug(
                "Mixture mode selected per-path generation because path seeds diverge from the base SeedManager"
            )

        def _evaluate_path(
            path_id: int, seed: int | None
        ) -> tuple[list[StrategyEvaluation], list[MonteCarloPathError]]:
            strategy = self._sample_strategy(strategies, strategy_seeds[path_id])
            try:
                context = self._generate_path_context(
                    path_id=path_id,
                    seed=seed,
                    model=model,
                    n_periods=n_periods,
                    path_result=path_result,
                    path_index=path_id,
                    fold_id=fold_id,
                    fold_label=fold_label,
                )
            except Exception as exc:
                self._log_path_error(path_id, None, exc, fold_id=fold_id, fold_label=fold_label)
                return [], [
                    self._error_record(path_id, None, exc, fold_id=fold_id, fold_label=fold_label)
                ]

            try:
                evaluation = self._evaluate_strategy(strategy, context)
                return [evaluation], []
            except Exception as exc:
                self._log_path_error(
                    path_id,
                    strategy.name,
                    exc,
                    fold_id=fold_id,
                    fold_label=fold_label,
                )
                return [], [
                    self._error_record(
                        path_id,
                        strategy.name,
                        exc,
                        fold_id=fold_id,
                        fold_label=fold_label,
                    )
                ]

        completed = 0
        for path_id, path_eval, path_err in self._execute_paths(path_seeds, _evaluate_path, jobs):
            evaluations.extend(path_eval)
            errors.extend(path_err)
            completed += 1
            self._emit_progress(progress_callback, completed, total, path_id, "mixture")

        return evaluations, errors

    def _generate_path_context(
        self,
        *,
        path_id: int,
        seed: int | None,
        model: Any,
        n_periods: int,
        path_result: Any | None = None,
        path_index: int = 0,
        fold_id: int | None,
        fold_label: str | None,
    ) -> _PathContext:
        if path_result is None:
            result = model.sample_prices(
                n_periods=n_periods,
                n_paths=1,
                frequency=self.scenario.simulation_frequency(),
                seed=seed,
            )
            path_index = 0
        else:
            result = path_result
        prices = self._extract_path_frame(result.prices, path_index)
        log_returns = self._extract_path_frame(result.log_returns, path_index)
        returns = np.expm1(log_returns)
        returns_df = self._returns_with_date(returns)
        returns_df = self._inject_cash_returns(returns_df)
        score_frame = self._compute_score_frame(returns_df)
        path_hash = self._hash_frame(prices)
        return _PathContext(
            fold_id=fold_id,
            fold_label=fold_label,
            path_id=path_id,
            prices=prices,
            returns=returns_df,
            score_frame=score_frame,
            path_hash=path_hash,
            seed=seed,
        )

    def _evaluate_strategy(
        self,
        strategy: StrategyVariant,
        context: _PathContext,
    ) -> StrategyEvaluation:
        strategy_seed = self._strategy_seed(context.path_id, strategy.name)
        config = self._build_strategy_config(strategy, strategy_seed)
        self._apply_regime_turnover_caps(config, context)
        run_result = run_simulation(config, context.returns)
        metrics, source = self._extract_metrics(run_result.metrics)
        cost_payload = self._maybe_sample_costs(run_result, context, strategy)
        if cost_payload is not None:
            metrics = dict(metrics)
            metrics["total_cost_drag"] = float(cost_payload.total_cost_drag)
        if (
            not metrics
            and isinstance(context.score_frame, pd.DataFrame)
            and not context.score_frame.empty
        ):
            fallback = context.score_frame.mean(numeric_only=True)
            metrics = {str(k): float(v) for k, v in fallback.items()}
            source = "score_frame_mean"
        diagnostic: dict[str, Any] | None = None
        if run_result.diagnostic is not None:
            diagnostic = {
                "reason_code": run_result.diagnostic.reason_code,
                "message": run_result.diagnostic.message,
            }
        portfolio_cfg = getattr(config, "portfolio", {})
        max_turnover = None
        if isinstance(portfolio_cfg, Mapping):
            max_turnover = portfolio_cfg.get("max_turnover")
        regime_cfg = getattr(config, "regime", None)
        regime_settings = (
            normalise_settings(regime_cfg) if isinstance(regime_cfg, Mapping) else None
        )
        turnover_series, binding, cap_series, cap_regimes = self._resolve_turnover_diagnostics(
            run_result,
            context,
            max_turnover=max_turnover,
            regime_settings=regime_settings,
        )
        if turnover_series is not None or binding is not None:
            if diagnostic is None:
                diagnostic = {}
            if turnover_series is not None:
                diagnostic["turnover"] = turnover_series
            if binding is not None:
                diagnostic["turnover_cap_binding"] = binding
            if cap_series is not None:
                diagnostic["turnover_cap"] = cap_series
            if cap_regimes is not None:
                diagnostic["turnover_cap_regime"] = cap_regimes
        if cost_payload is not None:
            if diagnostic is None:
                diagnostic = {}
            diagnostic["costs"] = self._cost_payload_dict(cost_payload)
        nav_series = None
        try:
            nav_series = self._extract_nav_series(run_result, context)
        except Exception as exc:
            self._logger.debug(
                "Failed to extract NAV series for path %s (%s): %s",
                context.path_id,
                strategy.name,
                exc,
            )
        return StrategyEvaluation(
            fold_id=context.fold_id,
            fold_label=context.fold_label,
            path_id=context.path_id,
            strategy_name=strategy.name,
            metrics=metrics,
            metric_source=source,
            path_hash=context.path_hash,
            seed=context.seed,
            diagnostic=diagnostic,
            turnover=turnover_series,
            turnover_cap_binding=binding,
            nav_series=nav_series,
        )

    def _apply_regime_turnover_caps(self, config: ConfigType, context: _PathContext) -> None:
        """Apply turnover-cap regime overrides to ``portfolio.max_turnover``.

        Supported shapes for ``portfolio.max_turnover`` are a numeric scalar or a
        regime mapping. Regime key aliases are normalized so ``calm``/``stress`` map
        to the canonical ``riskon``/``riskoff`` labels. Caps are applied when a
        regime mapping is provided and regime detection is enabled; they are
        bypassed when regime detection is disabled or no mapping can be resolved.
        Numeric coercion happens only in parse_regime_turnover_caps.
        """
        portfolio = getattr(config, "portfolio", None)
        if not isinstance(portfolio, Mapping):
            return
        max_turnover = portfolio.get("max_turnover")
        # Supported shapes for ``portfolio.max_turnover``:
        # - numeric scalar (int/float/np.number): apply as-is without regime lookup. Numeric
        #   coercion/validation happens only inside ``parse_regime_turnover_caps``.
        # - mapping: regime-conditioned caps; keys are normalized/aliased (case-insensitive,
        #   punctuation-stripped, calm/stress -> riskon/riskoff) via
        #   ``parse_regime_turnover_caps`` to canonical regime labels.
        # Caps are applied only when a mapping is provided and regime detection is enabled.
        # Single-period runs resolve the mapping to a scalar for the current path; if no
        # regime label can be resolved or the mapping is empty, the cap is left unchanged.
        # Multi-period runs keep the mapping so the engine can apply per-period caps.
        if max_turnover is None or not isinstance(max_turnover, Mapping):
            return

        multi_period_cfg = getattr(config, "multi_period", None)
        multi_period_enabled = isinstance(multi_period_cfg, Mapping)
        # Control flow:
        # - Single-period runs resolve the mapping to a scalar for the current path.
        # - Multi-period runs keep the mapping so the engine can apply per-period caps.
        if multi_period_enabled:
            resolved: float | Mapping[str, Any] | None = max_turnover
        else:
            resolved = self._resolve_turnover_cap_for_context(config, context, max_turnover)
        if resolved is None or resolved is max_turnover:
            return
        if isinstance(portfolio, dict):
            portfolio["max_turnover"] = resolved

    def _resolve_turnover_cap_for_context(
        self,
        config: ConfigType,
        context: _PathContext,
        max_turnover: Mapping[str, Any],
    ) -> float | Mapping[str, Any] | None:
        regime_cfg = getattr(config, "regime", None)
        if not isinstance(regime_cfg, Mapping):
            return max_turnover
        settings = normalise_settings(regime_cfg)
        if not settings.enabled or not settings.proxy:
            return max_turnover
        proxy_col = self._resolve_regime_proxy_column(
            context.returns,
            settings.proxy,
            getattr(config, "benchmarks", None),
        )
        if proxy_col is None:
            return max_turnover
        proxy_series = self._build_regime_proxy_series(context.returns, proxy_col)
        if proxy_series is None or proxy_series.empty:
            return max_turnover
        data_cfg = getattr(config, "data", None)
        freq = None
        if isinstance(data_cfg, Mapping):
            freq = data_cfg.get("frequency")
        freq_label = str(freq or "M")
        periods_per_year = periods_per_year_from_code(freq_label)
        proxy_window = (proxy_series.index[0], proxy_series.index[-1], len(proxy_series))
        cache_key = self._regime_cache_key(
            context,
            proxy_col,
            proxy_window,
            freq_label,
            periods_per_year,
            settings,
        )
        regimes = self._regime_cache.get(cache_key)
        if regimes is None:
            regimes = compute_regimes(
                proxy_series,
                settings,
                freq=freq_label,
                periods_per_year=periods_per_year,
            )
            self._regime_cache[cache_key] = regimes
        if regimes.empty:
            return max_turnover
        regime_label = str(regimes.iloc[-1])
        parsed = parse_regime_turnover_caps(max_turnover, settings)
        resolved = self._resolve_turnover_cap_from_parsed(parsed, regime_label, settings)
        if resolved is None:
            return max_turnover
        return resolved

    def _resolve_turnover_cap_from_parsed(
        self,
        parsed: dict[str, float] | float | None,
        regime_label: str | None,
        settings: Any,
    ) -> float | None:
        if parsed is None:
            return None
        if not isinstance(parsed, Mapping):
            return parsed

        def _lookup(label: str | None) -> float | None:
            normalized = normalize_regime_key(label)
            if not normalized:
                return None
            if normalized in parsed:
                return parsed[normalized]
            return None

        if regime_label:
            resolved = _lookup(regime_label)
            if resolved is not None:
                return resolved

        default_label = getattr(settings, "default_label", None)
        resolved = _lookup(default_label)
        if resolved is not None:
            return resolved

        for fallback in ("default", "all", "any"):
            if fallback in parsed:
                return parsed[fallback]
        return None

    def _regime_cache_key(
        self,
        context: _PathContext,
        proxy_col: str,
        proxy_window: tuple[object, object, int],
        freq_label: str,
        periods_per_year: float,
        settings: Any,
    ) -> tuple[object, ...]:
        return (
            context.path_hash,
            proxy_col,
            proxy_window,
            freq_label,
            periods_per_year,
            getattr(settings, "method", None),
            getattr(settings, "lookback", None),
            getattr(settings, "smoothing", None),
            getattr(settings, "threshold", None),
            getattr(settings, "neutral_band", None),
            getattr(settings, "min_obs", None),
            getattr(settings, "risk_on_label", None),
            getattr(settings, "risk_off_label", None),
            getattr(settings, "default_label", None),
            getattr(settings, "annualise_volatility", None),
        )

    def _resolve_regime_proxy_column(
        self,
        returns: pd.DataFrame,
        proxy: str,
        benchmarks: Mapping[str, Any] | None,
    ) -> str | None:
        columns = returns.columns
        if proxy in columns:
            return proxy
        proxy_lower = str(proxy).lower()
        if isinstance(benchmarks, Mapping):
            for key, value in benchmarks.items():
                if proxy_lower == str(key).lower() and value in columns:
                    return str(value)
                if proxy_lower == str(value).lower() and value in columns:
                    return str(value)
        for col in columns:
            if proxy_lower == str(col).lower():
                return str(col)
        return None

    def _build_regime_proxy_series(self, returns: pd.DataFrame, proxy_col: str) -> pd.Series | None:
        if proxy_col not in returns.columns:
            return None
        if "Date" in returns.columns:
            index = pd.to_datetime(returns["Date"], errors="coerce")
        else:
            index = returns.index
        series = pd.Series(returns[proxy_col].to_numpy(), index=index)
        return series.dropna()

    def _compute_n_periods(self) -> int:
        settings = self._settings()
        frequency = settings.frequency
        horizon_years = settings.horizon_years
        if frequency is None:
            raise ValueError("monte_carlo.frequency is required")
        if horizon_years is None:
            raise ValueError("monte_carlo.horizon_years is required")
        periods_per_year = periods_per_year_from_code(frequency)
        n_periods = int(math.ceil(float(horizon_years) * periods_per_year))
        return max(n_periods, 1)

    def _resolve_jobs(self, jobs: int | None) -> int:
        requested = jobs if jobs is not None else self._settings().jobs
        if requested is None:
            return 1
        try:
            count = int(requested)
        except (TypeError, ValueError):
            return 1
        return max(count, 1)

    def _resolve_strategies(self) -> list[StrategyVariant]:
        strategy_set = self._strategy_set()
        variants: list[StrategyVariant] = []
        curated = strategy_set.get("curated")
        if isinstance(curated, list) and curated:
            for item in curated:
                if isinstance(item, StrategyVariant):
                    variants.append(item)
                elif isinstance(item, str):
                    variants.append(StrategyVariant(name=item))
        sampled = self._resolve_sampled_strategies(strategy_set, existing=variants)
        if sampled:
            variants.extend(sampled)
        if not variants:
            variants = [StrategyVariant(name="base")]
        return variants

    def _resolve_sampled_strategies(
        self, strategy_set: Mapping[str, Any], *, existing: Sequence[StrategyVariant]
    ) -> list[StrategyVariant]:
        sampled = strategy_set.get("sampled")
        if not isinstance(sampled, Mapping):
            return []
        enabled = sampled.get("enabled", True)
        if isinstance(enabled, bool):
            if not enabled:
                return []
        elif enabled is None:
            return []
        else:
            raise ValueError("strategy_set.sampled.enabled must be a bool")

        n_value = sampled.get("n_strategies")
        if n_value is None:
            raise ValueError("strategy_set.sampled.n_strategies is required")
        try:
            n_strategies = int(n_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("strategy_set.sampled.n_strategies must be an integer") from exc
        if n_strategies < 1:
            raise ValueError("strategy_set.sampled.n_strategies must be >= 1")

        sampling = sampled.get("sampling")
        if not isinstance(sampling, Mapping):
            raise ValueError("strategy_set.sampled.sampling must be a mapping")

        name_prefix = sampled.get("name_prefix", "sampled")
        seed = sampled.get("seed", self._settings().seed)
        if seed is not None:
            try:
                seed = int(seed)
            except (TypeError, ValueError) as exc:
                raise ValueError("strategy_set.sampled.seed must be an integer") from exc

        max_rejection_attempts = sampled.get("max_rejection_attempts", 1000)
        try:
            max_rejection_attempts = int(max_rejection_attempts)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "strategy_set.sampled.max_rejection_attempts must be an integer"
            ) from exc
        if max_rejection_attempts < 0:
            raise ValueError("strategy_set.sampled.max_rejection_attempts must be >= 0")

        existing_names = [variant.name for variant in existing]
        return sample_strategy_variants(
            sampling,
            n_strategies,
            seed=seed,
            max_rejection_attempts=max_rejection_attempts,
            name_prefix=str(name_prefix),
            existing_names=existing_names,
        )

    def _build_price_model(
        self,
        history: pd.DataFrame | None = None,
        *,
        calibration_start: pd.Timestamp | None = None,
        calibration_end: pd.Timestamp | None = None,
    ) -> Any:
        model_spec_raw = self.scenario.return_model
        model_spec: Mapping[str, Any] = (
            model_spec_raw if isinstance(model_spec_raw, Mapping) else {}
        )
        kind = str(model_spec.get("kind") or "stationary_bootstrap").lower()
        params = model_spec.get("params") or {}
        frequency = self.scenario.simulation_frequency()
        model: Any
        if kind in {"stationary_bootstrap", "bootstrap"}:
            mean_block_len = params.get("mean_block_len", params.get("block_size", 6))
            calibration_window = params.get("calibration_window")
            model = StationaryBootstrapModel(
                mean_block_len=mean_block_len,
                calibration_window=calibration_window,
                frequency=frequency,
            )
        elif kind in {"regime_bootstrap", "regime_conditioned"}:
            mean_block_len = params.get("mean_block_len", params.get("block_size", 6))
            calibration_window = params.get("calibration_window")
            model = RegimeConditionedBootstrapModel(
                mean_block_len=mean_block_len,
                calibration_window=calibration_window,
                frequency=frequency,
                regime_proxy_column=params.get("regime_proxy_column"),
                threshold_percentile=params.get("threshold_percentile", 75.0),
                lookback=params.get("lookback", 20),
            )
        else:
            raise ValueError(f"Unsupported return model '{kind}'")

        resolved_history = history if history is not None else self._resolve_price_history()
        if calibration_start is not None or calibration_end is not None:
            normalized, _summary = normalize_price_frequency(resolved_history, frequency)
            windowed = normalized.loc[calibration_start:calibration_end]
            if windowed.empty:
                raise ValueError(
                    "fold calibration window produced no history data "
                    f"({calibration_start} to {calibration_end})"
                )
            resolved_history = windowed
        return model.fit(resolved_history, frequency=frequency)

    def _resolve_folds(self, history: pd.DataFrame) -> list[Fold]:
        if not self.scenario.enable_fold_runs:
            return []
        folds_config = cast(Mapping[str, Any] | None, self.scenario.folds)
        generator = FoldGenerator.from_config(folds_config)
        if generator is None:
            return []
        return generator.generate(history.index)

    def _slice_history(self, history: pd.DataFrame, fold: Fold) -> pd.DataFrame:
        sliced = history.loc[fold.calibration_start : fold.calibration_end]
        if sliced.empty:
            raise ValueError(
                "fold calibration window produced no history data "
                f"({fold.calibration_start} to {fold.calibration_end})"
            )
        return sliced

    def _resolve_price_history(self) -> pd.DataFrame:
        if self._price_history is not None:
            return self._price_history.copy()
        data_cfg = self._base_config.get("data", {})
        csv_path = data_cfg.get("csv_path")
        if csv_path:
            return self._load_history_from_path(Path(str(csv_path)))
        raise ValueError("price_history must be provided or data.csv_path must be configured")

    def _load_history_from_path(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            validated = load_market_data_parquet(str(path))
        else:
            validated = load_market_data_csv(str(path))
        frame = validated.frame.copy()
        mode = validated.metadata.mode
        if mode == MarketDataMode.RETURNS:
            return self._returns_to_prices(frame)
        return frame

    def _returns_to_prices(self, returns: pd.DataFrame) -> pd.DataFrame:
        if returns.empty:
            raise ValueError("returns data must not be empty")
        if (returns <= -1.0).any().any():
            raise ValueError("returns contain values <= -1; cannot convert to prices")
        prices = (1.0 + returns).cumprod() * 100.0
        return prices

    def _build_seeds(self) -> tuple[list[int | None], list[int | None]]:
        settings = self._settings()
        n_paths = settings.n_paths
        if n_paths is None:
            raise ValueError("monte_carlo.n_paths is required")
        seed_manager = self._get_seed_manager()
        if seed_manager is None:
            path_seeds: list[int | None] = [None] * n_paths
            strategy_seeds: list[int | None] = [None] * n_paths
            return path_seeds, strategy_seeds
        path_seeds = [seed_manager.get_path_seed(path_id) for path_id in range(n_paths)]
        strategy_seeds = [
            seed_manager.get_strategy_seed(path_id, _STRATEGY_SELECTION_SEED_TAG)
            for path_id in range(n_paths)
        ]
        return path_seeds, strategy_seeds

    def _build_strategy_config(self, strategy: StrategyVariant, seed: int | None) -> ConfigType:
        merged = strategy.apply_to(self._base_config)
        self._apply_strategy_guards(merged)
        self._apply_turnover_guard_distribution(merged, strategy, seed)
        if seed is not None:
            merged["seed"] = int(seed)
        return Config(**merged)

    def _apply_strategy_guards(self, merged: dict[str, Any]) -> None:
        strategy_set = self._strategy_set()
        guards = strategy_set.get("guards")
        if not isinstance(guards, Mapping):
            return
        portfolio = merged.setdefault("portfolio", {})
        if not isinstance(portfolio, dict):
            return
        if "max_turnover" in guards:
            guard_value = guards.get("max_turnover")
            if isinstance(guard_value, Mapping):
                if _is_distribution_mapping(guard_value):
                    return
                portfolio["max_turnover"] = dict(guard_value)
                return
            if guard_value is None:
                return
            portfolio["max_turnover"] = _coerce_turnover_guard(guard_value)

    def _apply_turnover_guard_distribution(
        self,
        merged: dict[str, Any],
        strategy: StrategyVariant,
        seed: int | None,
    ) -> None:
        distribution = self._resolve_turnover_guard_distribution()
        if distribution is None:
            return
        if _has_turnover_override(strategy):
            return
        portfolio = merged.setdefault("portfolio", {})
        if not isinstance(portfolio, dict):
            return
        rng = random.Random(seed) if seed is not None else random.Random()
        try:
            value = float(distribution.sample(rng))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{_TURNOVER_GUARD_PATH} distribution must sample numeric values"
            ) from exc
        portfolio["max_turnover"] = value

    def _resolve_turnover_guard_distribution(self) -> Any:
        strategy_set = self._strategy_set()
        guards = strategy_set.get("guards")
        if not isinstance(guards, Mapping):
            return None
        if "max_turnover" not in guards:
            return None
        guard_value = guards.get("max_turnover")
        if isinstance(guard_value, Mapping):
            if _is_distribution_mapping(guard_value):
                return parse_distribution(guard_value, path=_TURNOVER_GUARD_PATH)
            return None
        if guard_value is None:
            return None
        if _is_number(guard_value):
            return None
        _coerce_turnover_guard(guard_value)
        return None

    def _compute_score_frame(self, returns: pd.DataFrame) -> pd.DataFrame:
        try:
            config = Config(**self._base_config)
        except Exception:
            return pd.DataFrame()
        try:
            split = _resolve_sample_split(returns, config.sample_split)
            metrics_raw = config.metrics.get("registry")
            if not metrics_raw:
                metrics_raw = ["annual_return", "volatility", "sharpe_ratio"]
            metrics = canonical_metric_list(metrics_raw)
            data_settings = config.data or {}
            metrics_settings = config.metrics or {}
            use_resolver = bool(metrics_settings.get("rf_override_enabled", False))
            if data_settings.get("risk_free_column"):
                use_resolver = True
            if data_settings.get("allow_risk_free_fallback") is True:
                use_resolver = True
            risk_free_value: float | pd.Series | None = None
            if use_resolver:
                resolution = resolve_risk_free_source(returns, config)
                risk_free_value = resolution.risk_free
                if isinstance(risk_free_value, pd.Series):
                    date_col = str(data_settings.get("date_column") or "Date")
                    if date_col in returns.columns:
                        risk_free_value = pd.Series(
                            risk_free_value.to_numpy(),
                            index=pd.to_datetime(returns[date_col].values),
                            name=risk_free_value.name,
                        )
            stats_cfg = RiskStatsConfig(
                metrics_to_run=metrics,
                risk_free=(
                    float(risk_free_value) if isinstance(risk_free_value, (int, float)) else 0.0
                ),
                periods_per_year=int(periods_per_year_from_code(config.data.get("frequency"))),
            )
            return single_period_run(
                returns,
                split["in_start"],
                split["in_end"],
                stats_cfg=stats_cfg,
                risk_free=risk_free_value,
            )
        except Exception as exc:
            self._logger.debug("Failed to compute score frame: %s", exc)
            return pd.DataFrame()

    def _warn_cash_once(self, msg: str, *args: object) -> None:
        """Emit a CASH-injection warning at most once per runner lifetime."""
        if self._cash_injection_warned:
            return
        self._cash_injection_warned = True
        self._logger.warning(msg, *args)

    def _inject_cash_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        if "CASH" in returns.columns:
            return returns
        try:
            config = Config(**self._base_config)
        except Exception:
            self._warn_cash_once("CASH injection skipped: failed to parse base config")
            return returns
        data_settings = config.data or {}
        if data_settings.get("risk_free_column"):
            return returns
        if data_settings.get("allow_risk_free_fallback") is not True:
            self._warn_cash_once(
                "CASH injection skipped: data.allow_risk_free_fallback is not "
                "enabled (set to true in your scenario or base config to "
                "allow CASH column injection)"
            )
            return returns
        try:
            resolution = resolve_risk_free_source(returns, config)
        except Exception as exc:
            self._warn_cash_once("CASH injection failed: %s", exc)
            return returns
        risk_free = resolution.risk_free
        if not isinstance(risk_free, pd.Series):
            self._warn_cash_once(
                "CASH injection skipped: resolved risk-free source is not a Series (source=%s)",
                getattr(resolution, "source", "unknown"),
            )
            return returns
        cash_values = pd.to_numeric(risk_free, errors="coerce")
        if len(cash_values) != len(returns):
            self._warn_cash_once(
                "CASH injection skipped: risk-free length (%d) does not match "
                "returns length (%d)",
                len(cash_values),
                len(returns),
            )
            return returns
        if cash_values.isna().any():
            self._warn_cash_once(
                "CASH injection skipped: resolved risk-free series contains NaN values"
            )
            return returns
        out = returns.copy()
        out["CASH"] = cash_values.to_numpy(dtype=float, copy=False)
        return out

    def _extract_metrics(self, metrics_df: pd.DataFrame) -> tuple[dict[str, float], str | None]:
        if metrics_df is None or metrics_df.empty:
            return {}, None
        source = None
        if "user_weight" in metrics_df.index:
            row = metrics_df.loc["user_weight"]
            source = "user_weight"
        elif "equal_weight" in metrics_df.index:
            row = metrics_df.loc["equal_weight"]
            source = "equal_weight"
        else:
            row = metrics_df.iloc[0]
            try:
                source = str(metrics_df.index[0])
            except Exception:
                source = None
        return {str(k): float(v) for k, v in row.items()}, source

    def _maybe_sample_costs(
        self,
        run_result: Any,
        context: _PathContext,
        strategy: StrategyVariant,
    ) -> CostProcessOutput | None:
        cost_process = self._get_cost_process()
        if cost_process is None:
            return None
        out_index = self._resolve_cost_index(run_result, context)
        if out_index is None or len(out_index) == 0:
            return None
        regimes = self._resolve_regime_labels(run_result, out_index)
        turnover = self._resolve_turnover_series(run_result, out_index)
        rng = self._cost_rng(context.path_id, strategy.name)
        return cost_process.sample(
            regimes=regimes,
            turnover=turnover,
            index=out_index,
            rng=rng,
        )

    def _resolve_cost_index(self, run_result: Any, context: _PathContext) -> pd.Index | None:
        details = getattr(run_result, "details", None)
        if isinstance(details, Mapping):
            out_scaled = details.get("out_sample_scaled")
            if isinstance(out_scaled, pd.DataFrame):
                return out_scaled.index
            period_index = self._period_results_index(details.get("period_results"))
            if period_index is not None:
                return period_index
            turnover = details.get("turnover")
            if isinstance(turnover, pd.Series):
                return turnover.index
        turnover_attr = getattr(run_result, "turnover", None)
        if isinstance(turnover_attr, pd.Series):
            return turnover_attr.index
        if isinstance(context.returns, pd.DataFrame) and "Date" in context.returns.columns:
            return pd.DatetimeIndex(context.returns["Date"]).copy()
        return None

    def _resolve_regime_labels(self, run_result: Any, out_index: pd.Index) -> pd.Series | None:
        details = getattr(run_result, "details", None)
        if isinstance(details, Mapping):
            labels = details.get("regime_labels_out")
            if isinstance(labels, pd.Series):
                return labels.reindex(out_index)
            labels = details.get("regime_labels")
            if isinstance(labels, pd.Series):
                return labels.reindex(out_index)
            period_labels = self._period_results_regimes(details.get("period_results"), out_index)
            if period_labels is not None:
                return period_labels
        return None

    def _resolve_turnover_series(
        self, run_result: Any, out_index: pd.Index | None
    ) -> pd.Series | float | None:
        turnover_attr = getattr(run_result, "turnover", None)
        if isinstance(turnover_attr, pd.Series):
            if out_index is None:
                return turnover_attr
            if len(turnover_attr) == 1 and len(out_index) > 1:
                value = float(turnover_attr.iloc[0])
                return pd.Series(value, index=out_index, name=turnover_attr.name or "turnover")
            return turnover_attr.reindex(out_index).fillna(0.0)
        turnover_value = turnover_attr if isinstance(turnover_attr, (float, int)) else None
        details = getattr(run_result, "details", None)
        if isinstance(details, Mapping):
            turnover = details.get("turnover")
            if isinstance(turnover, pd.Series):
                if out_index is None:
                    return turnover
                if len(turnover) == 1 and len(out_index) > 1:
                    value = float(turnover.iloc[0])
                    return pd.Series(value, index=out_index, name=turnover.name or "turnover")
                return turnover.reindex(out_index).fillna(0.0)
            if isinstance(turnover, (float, int)):
                return float(turnover)
            risk_diag = details.get("risk_diagnostics")
            if isinstance(risk_diag, Mapping):
                turnover_value = risk_diag.get("turnover_value")
                if isinstance(turnover_value, (float, int)):
                    return float(turnover_value)
        if isinstance(turnover_value, (float, int)):
            return float(turnover_value)
        return None

    def _resolve_nav_return_series(self, run_result: Any) -> pd.Series | None:
        portfolio_attr = getattr(run_result, "portfolio", None)
        if isinstance(portfolio_attr, pd.Series):
            return portfolio_attr
        analysis = getattr(run_result, "analysis", None)
        if analysis is not None:
            candidate = getattr(analysis, "returns", None)
            if isinstance(candidate, pd.Series):
                return candidate
            candidate = getattr(analysis, "portfolio", None)
            if isinstance(candidate, pd.Series):
                return candidate
        details = getattr(run_result, "details", None)
        if isinstance(details, Mapping):
            for key in (
                "portfolio_user_weight_combined",
                "portfolio_user_weight",
                "portfolio_equal_weight_combined",
                "portfolio_equal_weight",
            ):
                candidate = details.get(key)
                if isinstance(candidate, pd.Series):
                    return candidate
        return None

    def _coerce_nav_index(
        self,
        index: pd.Index,
        run_result: Any,
        context: _PathContext,
        length: int,
    ) -> pd.DatetimeIndex | None:
        if isinstance(index, pd.DatetimeIndex):
            return index.copy()
        try:
            coerced = pd.to_datetime(index, errors="coerce")
        except Exception:
            coerced = None
        if coerced is not None:
            dt_index = pd.DatetimeIndex(coerced)
            if not dt_index.isna().all():
                return dt_index
        fallback = self._resolve_cost_index(run_result, context)
        if fallback is not None and len(fallback) == length:
            try:
                return pd.DatetimeIndex(fallback)
            except Exception:
                return None
        return None

    def _extract_nav_series(
        self,
        run_result: Any,
        context: _PathContext,
    ) -> pd.Series | None:
        returns = self._resolve_nav_return_series(run_result)
        if not isinstance(returns, pd.Series) or returns.empty:
            return None
        returns = returns.astype(float)
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if returns.empty:
            return None
        nav_index = self._coerce_nav_index(returns.index, run_result, context, len(returns))
        if nav_index is None:
            self._logger.debug(
                "nav_paths: non-datetime index for path %s; skipping NAV series",
                context.path_id,
            )
            return None
        returns = pd.Series(returns.to_numpy(), index=nav_index, name=returns.name)
        nav = (1.0 + returns.fillna(0.0)).cumprod()
        nav = nav.replace([np.inf, -np.inf], np.nan).dropna()
        if nav.empty:
            return None
        first = float(nav.iloc[0])
        if not np.isfinite(first) or first <= 0.0:
            self._logger.debug(
                "nav_paths: invalid starting NAV for path %s (value=%s)",
                context.path_id,
                first,
            )
            return None
        nav = nav / first
        nav.name = "NAV"
        return nav

    def _resolve_turnover_diagnostics(
        self,
        run_result: Any,
        context: _PathContext,
        *,
        max_turnover: object | None,
        regime_settings: Any | None,
    ) -> tuple[pd.Series | None, pd.Series | None, pd.Series | float | None, pd.Series | None]:
        out_index = self._resolve_cost_index(run_result, context)
        regimes = (
            self._resolve_regime_labels(run_result, out_index) if out_index is not None else None
        )
        turnover_raw = self._resolve_turnover_series(run_result, out_index)
        turnover_series = self._coerce_turnover_series(turnover_raw, out_index)
        parsed_caps = self._parse_turnover_cap_config(max_turnover, regime_settings)
        cap_series = self._resolve_turnover_cap_series_from_parsed(
            parsed_caps,
            regimes,
            out_index,
        )
        cap_regimes = self._resolve_turnover_cap_regimes(parsed_caps, regimes, out_index)
        binding = self._turnover_cap_binding_indicator(turnover_series, cap_series)
        return turnover_series, binding, cap_series, cap_regimes

    def _period_results_index(self, period_results: Any) -> pd.Index | None:
        if not isinstance(period_results, Sequence):
            return None
        values: list[Any] = []
        for entry in period_results:
            if not isinstance(entry, Mapping):
                continue
            period = entry.get("period")
            if isinstance(period, (list, tuple)) and len(period) > 2:
                values.append(period[2])
        if not values:
            return None
        return pd.Index(values, name="period")

    def _build_nav_paths_frame(
        self,
        evaluations: Sequence[StrategyEvaluation],
        *,
        primary_strategy_name: str | None,
        fold_id: int | None,
    ) -> pd.DataFrame:
        paths: dict[int, pd.Series] = {}
        index_ref: pd.Index | None = None
        for evaluation in evaluations:
            if fold_id is not None and evaluation.fold_id != fold_id:
                continue
            if (
                primary_strategy_name is not None
                and evaluation.strategy_name != primary_strategy_name
            ):
                continue
            nav = evaluation.nav_series
            if not isinstance(nav, pd.Series) or nav.empty:
                continue
            # Enforce canonical NAV normalization: each path must start at 1.0.
            if not np.isclose(nav.iloc[0], 1.0, atol=NUMERICAL_TOLERANCE_LOW):
                self._logger.debug(
                    "nav_paths: path %s did not normalize to 1.0; skipping", evaluation.path_id
                )
                continue
            if index_ref is None:
                index_ref = nav.index
            # All NAV series must share the same datetime-like index.
            elif not nav.index.equals(index_ref):
                self._logger.debug(
                    "nav_paths: index mismatch for path %s; skipping", evaluation.path_id
                )
                continue
            paths[int(evaluation.path_id)] = nav
        if not paths:
            return pd.DataFrame()
        frame = pd.DataFrame(paths)
        # Coerce to DatetimeIndex for canonical period-end semantics.
        frame.index = pd.DatetimeIndex(frame.index)
        frame.sort_index(inplace=True)
        # Ensure columns support either plain path ids or MultiIndex (path, asset="NAV").
        frame = self._coerce_nav_paths_columns(frame)
        return frame

    def _coerce_nav_paths_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Ensure nav_paths columns support both plain and MultiIndex path ids."""
        if isinstance(frame.columns, pd.MultiIndex):
            names = list(frame.columns.names)
            nlevels = frame.columns.nlevels
            # Ensure a "path" level exists (defaults to the first level).
            path_level = names.index("path") if "path" in names else 0
            if len(names) <= path_level:
                names.extend([None] * (path_level + 1 - len(names)))
            names[path_level] = "path"
            asset_level = names.index("asset") if "asset" in names else (1 if nlevels > 1 else None)
            if asset_level is None:
                # Append an "asset" level containing NAV when absent.
                arrays = [frame.columns.get_level_values(i) for i in range(nlevels)]
                arrays.append(pd.Index(["NAV"] * len(frame.columns)))
                names.append("asset")
                frame.columns = pd.MultiIndex.from_arrays(arrays, names=names)
                return frame
            if len(names) <= asset_level:
                names.extend([None] * (asset_level + 1 - len(names)))
            names[asset_level] = "asset"
            frame.columns = frame.columns.set_names(names)
            assets = frame.columns.get_level_values(asset_level)
            if not (assets == "NAV").all():
                # Force asset labels to NAV for fan chart compatibility.
                arrays = [frame.columns.get_level_values(i) for i in range(nlevels)]
                arrays[asset_level] = pd.Index(["NAV"] * len(frame.columns))
                frame.columns = pd.MultiIndex.from_arrays(arrays, names=frame.columns.names)
            try:
                # Keep paths ordered for deterministic fan chart exports.
                frame = frame.sort_index(axis=1, level=path_level)
            except TypeError:
                pass
            return frame
        frame.columns = pd.Index(frame.columns, name="path")
        try:
            frame = frame.sort_index(axis=1)
        except TypeError:
            pass
        return frame

    def _period_results_regimes(
        self,
        period_results: Any,
        out_index: pd.Index | None,
    ) -> pd.Series | None:
        if not isinstance(period_results, Sequence):
            return None
        values: list[Any] = []
        labels: list[Any] = []
        for entry in period_results:
            if not isinstance(entry, Mapping):
                continue
            period = entry.get("period")
            if not isinstance(period, (list, tuple)) or len(period) <= 2:
                continue
            values.append(period[2])
            label = None
            labels_out = entry.get("regime_labels_out")
            if isinstance(labels_out, pd.Series) and not labels_out.empty:
                label = labels_out.iloc[-1]
            else:
                labels_in = entry.get("regime_labels")
                if isinstance(labels_in, pd.Series) and not labels_in.empty:
                    label = labels_in.iloc[-1]
            labels.append(label)
        if not values:
            return None
        series = pd.Series(labels, index=pd.Index(values, name="period"), name="regime")
        try:
            series = series.astype("string")
        except Exception:
            pass
        if out_index is None:
            return series
        return series.reindex(out_index)

    def _coerce_turnover_series(
        self, turnover: pd.Series | SupportsFloat | None, out_index: pd.Index | None
    ) -> pd.Series | None:
        if turnover is None:
            return None
        if isinstance(turnover, pd.Series):
            if out_index is None:
                return turnover
            return turnover.reindex(out_index).fillna(0.0)
        if out_index is None:
            return None
        value = cast(SupportsFloat, turnover)
        return pd.Series(float(value), index=out_index, name="turnover")

    def _parse_turnover_cap_config(
        self,
        max_turnover: object | None,
        regime_settings: Any | None,
    ) -> dict[str, float] | float | None:
        return parse_regime_turnover_caps(max_turnover, regime_settings)

    def _resolve_turnover_cap_series(
        self,
        max_turnover: object | None,
        regimes: pd.Series | None,
        out_index: pd.Index | None,
        *,
        regime_settings: Any | None,
    ) -> pd.Series | float | None:
        parsed = self._parse_turnover_cap_config(max_turnover, regime_settings)
        return self._resolve_turnover_cap_series_from_parsed(parsed, regimes, out_index)

    def _resolve_turnover_cap_series_from_parsed(
        self,
        parsed: dict[str, float] | float | None,
        regimes: pd.Series | None,
        out_index: pd.Index | None,
    ) -> pd.Series | float | None:
        if parsed is None:
            return None
        if not isinstance(parsed, Mapping):
            if out_index is None:
                return parsed
            return pd.Series(parsed, index=out_index, name="turnover_cap")
        if out_index is None:
            return None
        if regimes is None:
            return pd.Series(np.nan, index=out_index, name="turnover_cap")
        labels = regimes.reindex(out_index).astype("string")

        def _lookup_turnover_cap(label: str | None) -> float:
            if not label:
                return np.nan
            normalized = normalize_regime_key(label)
            if not normalized:
                return np.nan
            if normalized in parsed:
                return parsed[normalized]
            return np.nan

        caps = labels.map(_lookup_turnover_cap)
        return pd.Series(caps, index=out_index, name="turnover_cap")

    def _resolve_turnover_cap_regimes(
        self,
        parsed: dict[str, float] | float | None,
        regimes: pd.Series | None,
        out_index: pd.Index | None,
    ) -> pd.Series | None:
        if not isinstance(parsed, Mapping):
            return None
        if regimes is None or out_index is None:
            return None
        labels = regimes.reindex(out_index).astype("string")

        def _canonical_label(label: str | None) -> str | None:
            if not label:
                return None
            normalized = normalize_regime_key(label)
            if not normalized:
                return None
            canonical = normalized
            if normalized in ("calm", "stress"):
                alias_key = alias_regime_key(normalized)
                if alias_key:
                    canonical = alias_key
            return canonical

        resolved = labels.map(_canonical_label)
        try:
            resolved = resolved.astype("string")
        except Exception:
            pass
        return resolved.rename("turnover_cap_regime")

    def _turnover_cap_binding_indicator(
        self,
        turnover: pd.Series | None,
        cap: pd.Series | float | None,
    ) -> pd.Series | None:
        if turnover is None or cap is None:
            return None
        if isinstance(cap, pd.Series):
            aligned_cap = cap.reindex(turnover.index)
            valid_cap = aligned_cap.where(aligned_cap > 0)
            binding = turnover >= (valid_cap - NUMERICAL_TOLERANCE_LOW)
            return binding.fillna(False).astype(bool).rename("turnover_cap_binding")
        if cap <= 0:
            return pd.Series(False, index=turnover.index, name="turnover_cap_binding")
        binding = turnover >= (cap - NUMERICAL_TOLERANCE_LOW)
        return binding.astype(bool).rename("turnover_cap_binding")

    def _cost_rng(self, path_id: int, strategy_name: str) -> np.random.Generator:
        manager = self._get_seed_manager()
        if manager is None:
            return np.random.default_rng()
        return manager.get_strategy_rng(path_id, strategy_name)

    def _cost_payload_dict(self, payload: CostProcessOutput) -> Mapping[str, Any]:
        return {
            "regimes": payload.regimes,
            "cost_bps": payload.cost_bps,
            "slippage_multiplier": payload.slippage_multiplier,
            "turnover": payload.turnover,
            "transaction_costs": payload.transaction_costs,
            "cost_drag": payload.cost_drag,
            "total_cost_drag": payload.total_cost_drag,
        }

    def _extract_path_frame(self, frame: pd.DataFrame, path_index: int = 0) -> pd.DataFrame:
        if isinstance(frame.columns, pd.MultiIndex) and "path" in frame.columns.names:
            return frame.xs(path_index, level="path", axis=1)
        return frame.copy()

    def _returns_with_date(self, returns: pd.DataFrame) -> pd.DataFrame:
        if isinstance(returns.index, pd.DatetimeIndex):
            out = returns.copy()
            out.insert(0, "Date", returns.index)
            return out.reset_index(drop=True)
        out = returns.copy()
        out.insert(0, "Date", pd.to_datetime(returns.index, errors="coerce"))
        return out.reset_index(drop=True)

    def _hash_frame(self, frame: pd.DataFrame) -> str:
        if frame.empty:
            return "empty"
        hashed = pd.util.hash_pandas_object(frame, index=True).sum()
        return f"{int(hashed):x}"

    def _sample_strategy(
        self, strategies: Sequence[StrategyVariant], seed: int | None
    ) -> StrategyVariant:
        if len(strategies) == 1:
            return strategies[0]
        rng = np.random.default_rng(seed)
        idx = int(rng.integers(0, len(strategies)))
        return strategies[idx]

    def _emit_progress(
        self,
        progress_callback: Callable[[Mapping[str, Any]], None] | None,
        completed: int,
        total: int,
        path_id: int,
        mode: str,
    ) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "completed": completed,
                "total": total,
                "path_id": path_id,
                "mode": mode,
            }
        )

    def _log_path_error(
        self,
        path_id: int,
        strategy_name: str | None,
        exc: Exception,
        *,
        fold_id: int | None = None,
        fold_label: str | None = None,
    ) -> None:
        label = f"path {path_id}"
        if fold_id is not None or fold_label:
            if fold_id is not None and fold_label:
                label = f"fold {fold_id} ({fold_label}) {label}"
            elif fold_id is not None:
                label = f"fold {fold_id} {label}"
            else:
                label = f"fold {fold_label} {label}"
        if strategy_name:
            label += f" strategy {strategy_name}"
        self._logger.exception("Monte Carlo evaluation failed for %s: %s", label, exc)

    def _error_record(
        self,
        path_id: int,
        strategy_name: str | None,
        exc: Exception,
        *,
        fold_id: int | None = None,
        fold_label: str | None = None,
    ) -> MonteCarloPathError:
        return MonteCarloPathError(
            fold_id=fold_id,
            fold_label=fold_label,
            path_id=path_id,
            strategy_name=strategy_name,
            error_type=type(exc).__name__,
            message=str(exc),
        )

    def _execute_paths(
        self,
        path_seeds: Sequence[int | None],
        fn: Callable[
            [int, int | None],
            tuple[list[StrategyEvaluation], list[MonteCarloPathError]],
        ],
        jobs: int,
    ) -> Iterable[tuple[int, list[StrategyEvaluation], list[MonteCarloPathError]]]:
        if jobs <= 1:
            for path_id, seed in enumerate(path_seeds):
                try:
                    evals, errs = fn(path_id, seed)
                except Exception as exc:
                    self._log_path_error(path_id, None, exc)
                    evals, errs = [], [self._error_record(path_id, None, exc)]
                yield (path_id, evals, errs)
            return

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(fn, path_id, seed): path_id
                for path_id, seed in enumerate(path_seeds)
            }
            for future in as_completed(futures):
                path_id = futures[future]
                try:
                    evals, errs = future.result()
                except Exception as exc:
                    self._log_path_error(path_id, None, exc)
                    evals, errs = [], [self._error_record(path_id, None, exc)]
                yield (path_id, evals, errs)

    def _maybe_export(self, results: MonteCarloResults) -> None:
        outputs = self.scenario.outputs or {}
        if not isinstance(outputs, Mapping):
            return
        directory = outputs.get("directory")
        if not directory:
            return
        output_dir = self._resolve_output_dir(str(directory))
        formats = outputs.get("formats", outputs.get("format"))
        export_results(results, output_dir, formats=formats)
        aggregation_config = outputs.get("aggregation", outputs.get("aggregations"))
        quantiles = None
        breach_spec = None
        expected_shortfall_spec = None
        if isinstance(aggregation_config, Mapping):
            quantiles = aggregation_config.get("quantiles")
            breach_spec = aggregation_config.get("breach")
            expected_shortfall_spec = aggregation_config.get("expected_shortfall")
        quantiles = outputs.get("quantiles", quantiles)
        breach_spec = outputs.get("breach", breach_spec)
        expected_shortfall_spec = outputs.get("expected_shortfall", expected_shortfall_spec)
        aggregation = aggregate_monte_carlo_results(
            results.results_frame,
            quantiles=quantiles,
            breach_spec=breach_spec,
            expected_shortfall_spec=expected_shortfall_spec,
        )
        export_aggregation_results(aggregation, output_dir, formats=formats)

    def _resolve_output_dir(self, template: str) -> Path:
        now = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        rendered = template.format(scenario_name=self.scenario.name, timestamp=now)
        return Path(rendered)

    def _coerce_base_config(self, base_config: Mapping[str, Any] | None) -> dict[str, Any]:
        if base_config is None:
            path = self._base_config_path()
            if not path.exists():
                raise FileNotFoundError(f"Base config not found: {path}")
            raw = path.read_text(encoding="utf-8")
            import yaml

            payload = yaml.safe_load(raw)
            if not isinstance(payload, dict):
                raise ValueError("Base config must be a mapping")
            payload = self._apply_scenario_overrides(payload)
            return self._ensure_required_sections(payload)
        if hasattr(base_config, "model_dump"):
            payload = base_config.model_dump()
        else:
            payload = dict(base_config)
        payload = self._apply_scenario_overrides(payload)
        return self._ensure_required_sections(payload)

    def _apply_scenario_overrides(self, config: dict[str, Any]) -> dict[str, Any]:
        """Merge scenario-level section overrides into the base config.

        Scenario YAML files may contain top-level keys like ``data`` that
        should override values inherited from ``base_config``.  Without this
        merge the overrides sit in ``scenario.raw`` but never reach the
        runtime config dict.
        """
        raw = getattr(self.scenario, "raw", None)
        if not isinstance(raw, Mapping):
            return config
        # Only merge keys that correspond to known config sections so we
        # don't accidentally inject scenario-only keys (e.g. ``costs``,
        # ``strategy_set``) into the pipeline config.
        mergeable_sections = (
            "data",
            "preprocessing",
            "vol_adjust",
            "sample_split",
            "portfolio",
            "metrics",
            "benchmarks",
            "export",
            "run",
            "regime",
        )
        updated = dict(config)
        for section in mergeable_sections:
            override = raw.get(section)
            if override is None or not isinstance(override, Mapping):
                continue
            existing = updated.get(section)
            if isinstance(existing, dict):
                merged = dict(existing)
                merged.update(override)
                updated[section] = merged
            else:
                updated[section] = dict(override)
        return updated

    def _ensure_required_sections(self, config: dict[str, Any]) -> dict[str, Any]:
        required = [
            "data",
            "preprocessing",
            "vol_adjust",
            "sample_split",
            "portfolio",
            "metrics",
            "export",
            "run",
            "benchmarks",
        ]
        updated = dict(config)
        for key in required:
            updated.setdefault(key, {})
        if "version" not in updated:
            updated["version"] = "0.1.0"
        return updated

    def _settings(self) -> MonteCarloSettings:
        settings = self.scenario.monte_carlo
        if not isinstance(settings, MonteCarloSettings):
            raise TypeError("monte_carlo settings are not resolved")
        return settings

    def _strategy_set(self) -> Mapping[str, Any]:
        strategy_set = self.scenario.strategy_set
        if isinstance(strategy_set, Mapping):
            return strategy_set
        return {}

    def _base_config_path(self) -> Path:
        base_config = self.scenario.base_config
        if isinstance(base_config, Path):
            return base_config
        if isinstance(base_config, str):
            return Path(base_config)
        raise TypeError("base_config must be a path")

    def _get_seed_manager(self) -> SeedManager | None:
        if self._seed_manager_init:
            return self._seed_manager
        self._seed_manager_init = True
        base_seed = self._settings().seed
        if base_seed is None:
            self._seed_manager = None
        else:
            self._seed_manager = SeedManager(int(base_seed))
        return self._seed_manager

    def _get_cost_process(self) -> CostProcess | None:
        if self._cost_process_init:
            return self._cost_process
        self._cost_process_init = True
        config = getattr(self.scenario, "costs", None)
        if isinstance(config, Mapping):
            self._cost_process = CostProcess.from_config(config)
        else:
            self._cost_process = None
        return self._cost_process

    def _strategy_seed(self, path_id: int, strategy_name: str) -> int | None:
        manager = self._get_seed_manager()
        if manager is None:
            return None
        return manager.get_strategy_seed(path_id, strategy_name)
