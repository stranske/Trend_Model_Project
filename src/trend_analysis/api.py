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

from .pipeline import _run_analysis, _resolve_weight_engine_params

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

    weighting_scheme = config.portfolio.get("weighting_scheme", "equal")
    weight_engine_params = _resolve_weight_engine_params(
        weighting_scheme, config.portfolio
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
        weighting_scheme=weighting_scheme,
        constraints=config.portfolio.get("constraints"),
        stats_cfg=stats_cfg,
        weight_engine_params=weight_engine_params,
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
    return RunResult(
        metrics=metrics_df,
        details=res,
        seed=seed,
        environment=env,
        fallback_info=fallback_info,
    )
