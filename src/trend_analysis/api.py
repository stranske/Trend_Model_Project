from __future__ import annotations

import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - for static type checking only
    from .config.models import ConfigProtocol as ConfigType
else:  # Runtime: avoid importing typing-only names
    from typing import Any as ConfigType

from .logging import RunLogMetadata, start_run_logger
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
    log: RunLogMetadata | None = None


def run_simulation(
    config: ConfigType,
    returns: pd.DataFrame,
    *,
    run_id: str | None = None,
    log_dir: str | Path | None = None,
) -> RunResult:
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
    with start_run_logger(run_id=run_id, log_dir=Path(log_dir) if log_dir else None) as run_logger:
        log_meta = RunLogMetadata(run_logger.run_id, run_logger.path)
        run_logger.log("initialise", "run_simulation start")

        seed = getattr(config, "seed", 42)
        run_logger.log("seed", "Seeding RNG", seed=seed)
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
            run_logger.log(
                "metrics",
                "Initialised risk metrics configuration",
                metrics=list(metrics_list),
            )

        try:
            run_logger.log("pipeline", "Invoking core pipeline")
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
            run_logger.log("pipeline", "Pipeline invocation complete")
        except Exception as exc:
            run_logger.log("pipeline", f"Pipeline execution failed: {exc}", level="ERROR")
            raise
    if res is None:
        logger.warning("run_simulation produced no result")
        run_logger.log(
            "pipeline",
            "Core pipeline returned no result payload",
            level="WARNING",
        )
        env = {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
        }
        run_logger.log("complete", "run_simulation end (empty result)")
        return RunResult(pd.DataFrame(), {}, seed, env, log=log_meta)

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
    if fallback_info:
        run_logger.log("fallback", "Weight engine fallback triggered", **fallback_info)
        run_logger.log("complete", "run_simulation end")
    logger.info("run_simulation end")
    return RunResult(
        metrics=metrics_df,
        details=res,
        seed=seed,
        environment=env,
        fallback_info=fallback_info,
        log=log_meta,
    )
