from __future__ import annotations

import logging
import random
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - for static type checking only
    from .config.models import ConfigProtocol as ConfigType
else:  # Runtime: avoid importing typing-only names
    from typing import Any as ConfigType

from .pipeline import _run_analysis

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Container for simulation output.

    Attributes
    ----------
    metrics : pd.DataFrame
        Summary metrics table.
    details : dict[str, Any]
        Full result payload returned by the pipeline.
    seed : int
        Random seed used for reproducibility.
    environment : dict[str, Any]
        Environment metadata (python/numpy/pandas versions).
    fallback_info : dict[str, Any] | None
        Present when a requested weight engine failed and the system
        reverted to equal weights.  Includes keys: ``engine``,
        ``error_type`` and ``error``.
    """

    metrics: pd.DataFrame
    details: dict[str, Any]
    seed: int
    environment: dict[str, Any]
    fallback_info: dict[str, Any] | None = None


def run_simulation(config: ConfigType, returns: pd.DataFrame) -> RunResult:
    """Execute the analysis pipeline using pre-loaded returns data.

    Parameters
    ----------
    config : Config
        Configuration object controlling the run.
    returns : pd.DataFrame
        DataFrame of returns including a ``Date`` column.

    Returns
    -------
    RunResult
        Structured results with the summary metrics and detailed payload.
    """
    logger.info("run_simulation start")

    seed = getattr(config, "seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    split = config.sample_split
    metrics_list = config.metrics.get("registry")
    stats_cfg = None
    if metrics_list:
        from .core.rank_selection import RiskStatsConfig, canonical_metric_list

        stats_cfg = RiskStatsConfig(
            metrics_to_run=canonical_metric_list(metrics_list),
            risk_free=0.0,
        )

    res = _run_analysis(
        returns,
        str(split.get("in_start")),
        str(split.get("in_end")),
        str(split.get("out_start")),
        str(split.get("out_end")),
        config.vol_adjust.get("target_vol", 1.0),
        getattr(config, "run", {}).get("monthly_cost", 0.0),
        selection_mode=config.portfolio.get("selection_mode", "all"),
        random_n=config.portfolio.get("random_n", 8),
        custom_weights=config.portfolio.get("custom_weights"),
        rank_kwargs=config.portfolio.get("rank"),
        manual_funds=config.portfolio.get("manual_list"),
        indices_list=config.portfolio.get("indices_list"),
        benchmarks=config.benchmarks,
        seed=seed,
        weighting_scheme=config.portfolio.get("weighting_scheme", "equal"),
        constraints=config.portfolio.get("constraints"),
        stats_cfg=stats_cfg,
    )
    if res is None:
        logger.warning("run_simulation produced no result")
        env = {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
        }
        return RunResult(pd.DataFrame(), {}, seed, env)

    stats_obj = res["out_sample_stats"]
    if isinstance(stats_obj, dict):
        stats_items = list(stats_obj.items())
    else:
        stats_items = list(getattr(stats_obj, "items", lambda: [])())
    metrics_df = pd.DataFrame({k: vars(v) for k, v in stats_items}).T
    bench_ir_obj = res.get("benchmark_ir", {})
    bench_ir_items = bench_ir_obj.items() if isinstance(bench_ir_obj, dict) else []
    for label, ir_map in bench_ir_items:
        col = f"ir_{label}"
        metrics_df[col] = pd.Series(
            {
                k: v
                for k, v in ir_map.items()
                if k not in {"equal_weight", "user_weight"}
            }
        )

    env = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }

    fallback_raw = res.get("weight_engine_fallback") if isinstance(res, dict) else None
    fallback_info: dict[str, Any] | None = (
        fallback_raw if isinstance(fallback_raw, dict) else None
    )
    logger.info("run_simulation end")
    # Construct portfolio series for bundle export (equal-weight baseline)
    try:
        in_scaled = res.get("in_sample_scaled")  # type: ignore[index]
        out_scaled = res.get("out_sample_scaled")  # type: ignore[index]
        ew_weights = res.get("ew_weights")  # type: ignore[index]
        if (
            isinstance(in_scaled, pd.DataFrame)
            and isinstance(out_scaled, pd.DataFrame)
            and isinstance(ew_weights, dict)
        ):
            # Build one continuous portfolio series across IS + OS
            import numpy as _np

            from .pipeline import calc_portfolio_returns as _cpr

            cols = list(in_scaled.columns)
            w = _np.array([ew_weights.get(c, 0.0) for c in cols])
            port_is = _cpr(w, in_scaled)
            port_os = _cpr(w, out_scaled)
            portfolio_series = pd.concat([port_is, port_os])
            res["portfolio_equal_weight_combined"] = portfolio_series
    except (
        KeyError,
        AttributeError,
        TypeError,
        IndexError,
    ):  # pragma: no cover - defensive
        pass

    rr = RunResult(
        metrics=metrics_df,
        details=res,
        seed=seed,
        environment=env,
        fallback_info=fallback_info,
    )
    # Ensure details dict is JSON-friendly (no Timestamp / non-primitive keys)
    try:  # pragma: no cover - lightweight sanitation (non-destructive)
        from pandas import DataFrame as _DataFrame
        from pandas import Series as _Series

        def _sanitize_keys(obj):  # type: ignore[override]
            if isinstance(obj, _Series):
                return {
                    (
                        str(getattr(i, "isoformat", lambda: i)())
                        if not isinstance(i, (str, int, float, bool, type(None)))
                        else i
                    ): _sanitize_keys(v)
                    for i, v in obj.items()
                }
            if isinstance(obj, _DataFrame):
                return {col: _sanitize_keys(obj[col]) for col in obj.columns}
            if isinstance(obj, dict):
                new = {}
                for k, v in obj.items():
                    # Leave DataFrame/Series values untouched (they will be sanitized recursively if needed)
                    if not isinstance(k, (str, int, float, bool, type(None))):
                        try:
                            sk = str(getattr(k, "isoformat", lambda: k)())
                        except Exception:  # pragma: no cover
                            sk = str(k)
                    else:
                        sk = k  # type: ignore[assignment]
                    new[sk] = _sanitize_keys(v)
                return new
            if isinstance(obj, (list, tuple)):
                return [_sanitize_keys(x) for x in obj]
            return obj

        # Store a parallel sanitized view for hashing/export without mutating original
        rr.details_sanitized = _sanitize_keys(rr.details)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        pass
    return rr
