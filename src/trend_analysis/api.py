from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
import random
import sys

import numpy as np
import pandas as pd

from .config import Config
from .pipeline import _run_analysis

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Container for simulation output."""

    metrics: pd.DataFrame
    details: dict[str, Any]
    seed: int
    environment: dict[str, Any]


def run_simulation(config: Config, returns: pd.DataFrame) -> RunResult:
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
    # Set random seeds for deterministic behavior
    # Note: PYTHONHASHSEED must be set before Python starts, so we don't set it here
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

    stats = res["out_sample_stats"]
    metrics_df = pd.DataFrame({k: vars(v) for k, v in stats.items()}).T
    for label, ir_map in res.get("benchmark_ir", {}).items():
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

    logger.info("run_simulation end")
    return RunResult(metrics=metrics_df, details=res, seed=seed, environment=env)
