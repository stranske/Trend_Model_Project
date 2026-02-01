"""Monte Carlo runner for evaluating strategy variants."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from trend_analysis.api import run_simulation
from trend_analysis.config.models import Config
from trend_analysis.core.rank_selection import RiskStatsConfig, canonical_metric_list
from trend_analysis.monte_carlo.models import (
    RegimeConditionedBootstrapModel,
    StationaryBootstrapModel,
)
from trend_analysis.monte_carlo.scenario import MonteCarloScenario
from trend_analysis.monte_carlo.strategy import StrategyVariant
from trend_analysis.pipeline import _resolve_sample_split
from trend_analysis.risk import periods_per_year_from_code
from trend_analysis.stages.selection import single_period_run
from trend_analysis.io.market_data import (
    MarketDataMode,
    load_market_data_csv,
    load_market_data_parquet,
)

from .results import (
    MonteCarloPathError,
    MonteCarloResults,
    StrategyEvaluation,
    build_results_frame,
    build_summary_frame,
    export_results,
)

__all__ = ["MonteCarloRunner"]


@dataclass(frozen=True)
class _PathContext:
    path_id: int
    prices: pd.DataFrame
    returns: pd.DataFrame
    score_frame: pd.DataFrame
    path_hash: str
    seed: int | None


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

    def run(
        self,
        *,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        jobs: int | None = None,
    ) -> MonteCarloResults:
        """Run the Monte Carlo simulation for the configured scenario."""

        strategies = self._resolve_strategies()
        model = self._build_price_model()
        n_periods = self._compute_n_periods()
        path_seeds, strategy_seeds = self._build_seeds()
        worker_count = self._resolve_jobs(jobs)

        mode = self.scenario.monte_carlo.mode
        if mode == "two_layer":
            evaluations, errors = self._run_two_layer(
                model=model,
                n_periods=n_periods,
                strategies=strategies,
                path_seeds=path_seeds,
                progress_callback=progress_callback,
                jobs=worker_count,
            )
        elif mode == "mixture":
            evaluations, errors = self._run_mixture(
                model=model,
                n_periods=n_periods,
                strategies=strategies,
                path_seeds=path_seeds,
                strategy_seeds=strategy_seeds,
                progress_callback=progress_callback,
                jobs=worker_count,
            )
        else:
            raise ValueError(f"Unsupported Monte Carlo mode '{mode}'")

        results_frame = build_results_frame(evaluations)
        summary_frame = build_summary_frame(results_frame)
        metadata = {
            "scenario": self.scenario.name,
            "mode": mode,
            "n_paths": self.scenario.monte_carlo.n_paths,
            "n_strategies": len(strategies),
            "seed": self.scenario.monte_carlo.seed,
        }
        results = MonteCarloResults(
            mode=mode,
            evaluations=evaluations,
            errors=errors,
            results_frame=results_frame,
            summary_frame=summary_frame,
            metadata=metadata,
        )
        self._maybe_export(results)
        return results

    def _run_two_layer(
        self,
        *,
        model: Any,
        n_periods: int,
        strategies: Sequence[StrategyVariant],
        path_seeds: Sequence[int | None],
        progress_callback: Callable[[Mapping[str, Any]], None] | None,
        jobs: int,
    ) -> tuple[list[StrategyEvaluation], list[MonteCarloPathError]]:
        total = len(path_seeds)
        evaluations: list[StrategyEvaluation] = []
        errors: list[MonteCarloPathError] = []

        def _evaluate_path(path_id: int, seed: int | None) -> tuple[list[StrategyEvaluation], list[MonteCarloPathError]]:
            try:
                context = self._generate_path_context(
                    path_id=path_id,
                    seed=seed,
                    model=model,
                    n_periods=n_periods,
                )
            except Exception as exc:
                self._log_path_error(path_id, None, exc)
                return [], [self._error_record(path_id, None, exc)]

            path_evals: list[StrategyEvaluation] = []
            path_errors: list[MonteCarloPathError] = []
            for strategy in strategies:
                try:
                    evaluation = self._evaluate_strategy(strategy, context)
                    path_evals.append(evaluation)
                except Exception as exc:
                    self._log_path_error(path_id, strategy.name, exc)
                    path_errors.append(self._error_record(path_id, strategy.name, exc))
            return path_evals, path_errors

        completed = 0
        for path_id, path_eval, path_err in self._execute_paths(
            path_seeds, _evaluate_path, jobs
        ):
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
    ) -> tuple[list[StrategyEvaluation], list[MonteCarloPathError]]:
        total = len(path_seeds)
        evaluations: list[StrategyEvaluation] = []
        errors: list[MonteCarloPathError] = []

        def _evaluate_path(path_id: int, seed: int | None) -> tuple[list[StrategyEvaluation], list[MonteCarloPathError]]:
            strategy = self._sample_strategy(strategies, strategy_seeds[path_id])
            try:
                context = self._generate_path_context(
                    path_id=path_id,
                    seed=seed,
                    model=model,
                    n_periods=n_periods,
                )
            except Exception as exc:
                self._log_path_error(path_id, None, exc)
                return [], [self._error_record(path_id, None, exc)]

            try:
                evaluation = self._evaluate_strategy(strategy, context)
                return [evaluation], []
            except Exception as exc:
                self._log_path_error(path_id, strategy.name, exc)
                return [], [self._error_record(path_id, strategy.name, exc)]

        completed = 0
        for path_id, path_eval, path_err in self._execute_paths(
            path_seeds, _evaluate_path, jobs
        ):
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
    ) -> _PathContext:
        result = model.sample_prices(
            n_periods=n_periods,
            n_paths=1,
            frequency=self.scenario.simulation_frequency(),
            seed=seed,
        )
        prices = self._extract_path_frame(result.prices)
        log_returns = self._extract_path_frame(result.log_returns)
        returns = np.expm1(log_returns)
        returns_df = self._returns_with_date(returns)
        score_frame = self._compute_score_frame(returns_df)
        path_hash = self._hash_frame(prices)
        return _PathContext(
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
        config = self._build_strategy_config(strategy, context.seed)
        run_result = run_simulation(config, context.returns)
        metrics, source = self._extract_metrics(run_result.metrics)
        diagnostic = None
        if run_result.diagnostic is not None:
            diagnostic = {
                "reason_code": run_result.diagnostic.reason_code,
                "message": run_result.diagnostic.message,
            }
        return StrategyEvaluation(
            path_id=context.path_id,
            strategy_name=strategy.name,
            metrics=metrics,
            metric_source=source,
            path_hash=context.path_hash,
            seed=context.seed,
            diagnostic=diagnostic,
        )

    def _compute_n_periods(self) -> int:
        settings = self.scenario.monte_carlo
        periods_per_year = periods_per_year_from_code(settings.frequency)
        n_periods = int(math.ceil(settings.horizon_years * periods_per_year))
        return max(n_periods, 1)

    def _resolve_jobs(self, jobs: int | None) -> int:
        requested = jobs if jobs is not None else self.scenario.monte_carlo.jobs
        if requested is None:
            return 1
        try:
            count = int(requested)
        except (TypeError, ValueError):
            return 1
        return max(count, 1)

    def _resolve_strategies(self) -> list[StrategyVariant]:
        strategy_set = self.scenario.strategy_set or {}
        curated = strategy_set.get("curated")
        if isinstance(curated, list) and curated:
            variants: list[StrategyVariant] = []
            for item in curated:
                if isinstance(item, StrategyVariant):
                    variants.append(item)
                elif isinstance(item, str):
                    variants.append(StrategyVariant(name=item))
            if variants:
                return variants
        return [StrategyVariant(name="base")]

    def _build_price_model(self) -> Any:
        model_spec = self.scenario.return_model or {}
        kind = str(model_spec.get("kind") or "stationary_bootstrap").lower()
        params = model_spec.get("params") or {}
        frequency = self.scenario.simulation_frequency()
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

        history = self._resolve_price_history()
        return model.fit(history, frequency=frequency)

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
        n_paths = self.scenario.monte_carlo.n_paths
        base_seed = self.scenario.monte_carlo.seed
        if base_seed is None:
            return [None] * n_paths, [None] * n_paths
        seq = np.random.SeedSequence(int(base_seed))
        child_seeds = seq.spawn(2)
        path_rng = np.random.default_rng(child_seeds[0])
        strategy_rng = np.random.default_rng(child_seeds[1])
        path_seeds = path_rng.integers(0, 2**32 - 1, size=n_paths, dtype=np.uint32).tolist()
        strategy_seeds = strategy_rng.integers(
            0, 2**32 - 1, size=n_paths, dtype=np.uint32
        ).tolist()
        return path_seeds, strategy_seeds

    def _build_strategy_config(self, strategy: StrategyVariant, seed: int | None) -> Config:
        merged = strategy.apply_to(self._base_config)
        self._apply_strategy_guards(merged)
        if seed is not None:
            merged["seed"] = int(seed)
        return Config(**merged)

    def _apply_strategy_guards(self, merged: dict[str, Any]) -> None:
        strategy_set = self.scenario.strategy_set or {}
        guards = strategy_set.get("guards")
        if not isinstance(guards, Mapping):
            return
        portfolio = merged.setdefault("portfolio", {})
        if not isinstance(portfolio, dict):
            return
        if "max_turnover" in guards:
            portfolio["max_turnover"] = guards.get("max_turnover")

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
            stats_cfg = RiskStatsConfig(
                metrics_to_run=metrics,
                risk_free=float(config.metrics.get("rf_rate_annual", 0.0) or 0.0),
                periods_per_year=int(periods_per_year_from_code(config.data.get("frequency"))),
            )
            return single_period_run(
                returns,
                split["in_start"],
                split["in_end"],
                stats_cfg=stats_cfg,
                risk_free=stats_cfg.risk_free,
            )
        except Exception as exc:
            self._logger.debug("Failed to compute score frame: %s", exc)
            return pd.DataFrame()

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

    def _extract_path_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if isinstance(frame.columns, pd.MultiIndex) and "path" in frame.columns.names:
            return frame.xs(0, level="path", axis=1)
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

    def _log_path_error(self, path_id: int, strategy_name: str | None, exc: Exception) -> None:
        label = f"path {path_id}"
        if strategy_name:
            label += f" strategy {strategy_name}"
        self._logger.exception("Monte Carlo evaluation failed for %s: %s", label, exc)

    def _error_record(
        self, path_id: int, strategy_name: str | None, exc: Exception
    ) -> MonteCarloPathError:
        return MonteCarloPathError(
            path_id=path_id,
            strategy_name=strategy_name,
            error_type=type(exc).__name__,
            message=str(exc),
        )

    def _execute_paths(
        self,
        path_seeds: Sequence[int | None],
        fn: Callable[[int, int | None], tuple[list[StrategyEvaluation], list[MonteCarloPathError]]],
        jobs: int,
    ) -> Iterable[tuple[int, list[StrategyEvaluation], list[MonteCarloPathError]]]:
        if jobs <= 1:
            for path_id, seed in enumerate(path_seeds):
                yield (path_id, *fn(path_id, seed))
            return

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(fn, path_id, seed): path_id
                for path_id, seed in enumerate(path_seeds)
            }
            for future in as_completed(futures):
                path_id = futures[future]
                yield (path_id, *future.result())

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

    def _resolve_output_dir(self, template: str) -> Path:
        now = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        rendered = template.format(scenario_name=self.scenario.name, timestamp=now)
        return Path(rendered)

    def _coerce_base_config(self, base_config: Mapping[str, Any] | None) -> dict[str, Any]:
        if base_config is None:
            path = Path(self.scenario.base_config)
            if not path.exists():
                raise FileNotFoundError(f"Base config not found: {path}")
            raw = path.read_text(encoding="utf-8")
            import yaml

            payload = yaml.safe_load(raw)
            if not isinstance(payload, dict):
                raise ValueError("Base config must be a mapping")
            return self._ensure_required_sections(payload)
        if hasattr(base_config, "model_dump"):
            payload = base_config.model_dump()
        else:
            payload = dict(base_config)
        return self._ensure_required_sections(payload)

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
