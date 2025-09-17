"""Utilities for the Streamlit "Run demo" workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

import pandas as pd
import yaml

from trend_analysis.api import run_simulation
from trend_analysis.config import Config
from trend_portfolio_app.data_schema import infer_benchmarks, load_and_validate_file
from trend_portfolio_app.policy_engine import MetricSpec, PolicyConfig

from .result_adapter import adapt_run_result

DEFAULT_PRESET = "Balanced"

_UI_METRIC_ALIASES: Dict[str, str] = {
    "sharpe_ratio": "sharpe",
    "sharpe": "sharpe",
    "return_ann": "return_ann",
    "annual_return": "return_ann",
    "max_drawdown": "drawdown",
    "drawdown": "drawdown",
    "volatility": "vol",
    "vol": "vol",
}

_PIPELINE_METRIC_ALIASES: Dict[str, str] = {
    "sharpe_ratio": "sharpe_ratio",
    "sharpe": "sharpe_ratio",
    "return_ann": "annual_return",
    "annual_return": "annual_return",
    "max_drawdown": "max_drawdown",
    "drawdown": "max_drawdown",
    "volatility": "volatility",
    "vol": "volatility",
}


class DemoRunError(RuntimeError):
    """Raised when the demo workflow cannot complete."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_demo_dataset() -> Path:
    root = _repo_root()
    for name in ("demo_returns.csv", "demo_returns.xlsx"):
        candidate = root / "demo" / name
        if candidate.exists():
            return candidate
    raise DemoRunError(
        "Demo dataset not found. Run 'python scripts/generate_demo.py' first."
    )


def _load_returns(path: Path) -> tuple[pd.DataFrame, Dict[str, Any]]:
    with path.open("rb") as handle:
        df, meta = load_and_validate_file(handle)
    return df, dict(meta)


def _load_preset(preset_name: str) -> Dict[str, Any]:
    root = _repo_root()
    path = root / "config" / "presets" / f"{preset_name.lower()}.yml"
    if not path.exists():
        raise DemoRunError(f"Preset '{preset_name}' not found in config/presets.")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise DemoRunError(f"Preset '{preset_name}' is not a mapping.")
    return data


def _map_metric_weights(raw: Mapping[str, Any]) -> tuple[Dict[str, float], list[str]]:
    ui_weights: Dict[str, float] = {}
    pipeline_metrics: list[str] = []
    for key, value in raw.items():
        try:
            weight = float(value)
        except Exception:
            continue
        if weight <= 0:
            continue
        ui_key = _UI_METRIC_ALIASES.get(key, key)
        pipe_key = _PIPELINE_METRIC_ALIASES.get(key, key)
        ui_weights[ui_key] = weight
        if pipe_key not in pipeline_metrics:
            pipeline_metrics.append(pipe_key)
    if not ui_weights:
        ui_weights = {"sharpe": 0.5, "return_ann": 0.5}
        pipeline_metrics = ["sharpe_ratio", "annual_return"]
    total = sum(ui_weights.values())
    if total > 0:
        ui_weights = {k: float(v) / float(total) for k, v in ui_weights.items()}
    return ui_weights, pipeline_metrics


def _choose_dates(df: pd.DataFrame, lookback_months: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    if df.empty:
        raise DemoRunError("Demo dataset is empty.")
    end = pd.Timestamp(df.index.max())
    candidate = end - pd.DateOffset(months=11)
    min_start = pd.Timestamp(df.index.min()) + pd.DateOffset(months=lookback_months)
    if candidate < min_start:
        candidate = min_start
    start_period = pd.Period(candidate, freq="M")
    end_period = pd.Period(end, freq="M")
    start = start_period.to_timestamp("M")
    end_ts = end_period.to_timestamp("M")
    return start, end_ts


def run_one_click_demo(
    session_state: MutableMapping[str, Any],
    preset_name: str = DEFAULT_PRESET,
):
    """Populate session state with demo data, configuration, and results."""

    dataset_path = _resolve_demo_dataset()
    returns_df, meta = _load_returns(dataset_path)
    preset = _load_preset(preset_name)

    benchmark_candidates = infer_benchmarks(list(returns_df.columns))
    benchmark_col = benchmark_candidates[0] if benchmark_candidates else None
    return_cols = [c for c in returns_df.columns if c != benchmark_col]

    column_mapping = {
        "date_column": "Date",
        "return_columns": return_cols,
        "benchmark_column": benchmark_col,
        "risk_free_column": None,
        "column_display_names": {},
        "column_tickers": {},
    }

    raw_metric_cfg = preset.get("metrics", {})
    if not isinstance(raw_metric_cfg, dict):
        raw_metric_cfg = {}
    ui_weights, pipeline_metrics = _map_metric_weights(raw_metric_cfg)

    overrides = {
        "lookback_months": int(preset.get("lookback_months", 36)),
        "rebalance_frequency": str(preset.get("rebalance_frequency", "monthly")),
        "min_track_months": int(preset.get("min_track_months", 24)),
        "selection_count": int(preset.get("selection_count", 10)),
        "risk_target": float(preset.get("risk_target", 0.10)),
        "cooldown_months": int(preset.get("portfolio", {}).get("cooldown_months", 3)),
        "selected_metrics": list(ui_weights.keys()),
        "metric_weights": ui_weights,
        "weighting_scheme": str(
            preset.get("portfolio", {}).get("weighting", {}).get("name", "equal")
        ),
    }
    if not overrides["selected_metrics"]:
        overrides["selected_metrics"] = ["sharpe", "return_ann"]
        overrides["metric_weights"] = {"sharpe": 0.5, "return_ann": 0.5}

    policy = PolicyConfig(
        top_k=overrides["selection_count"],
        bottom_k=0,
        cooldown_months=overrides["cooldown_months"],
        min_track_months=overrides["min_track_months"],
        max_active=100,
        max_weight=float(preset.get("portfolio", {}).get("max_weight", 0.15)),
        metrics=[
            MetricSpec(name=name, weight=overrides["metric_weights"][name])
            for name in overrides["selected_metrics"]
        ],
    )

    start, end = _choose_dates(returns_df, overrides["lookback_months"])

    sim_config = {
        "start": start,
        "end": end,
        "freq": overrides["rebalance_frequency"],
        "lookback_months": overrides["lookback_months"],
        "benchmark": benchmark_col,
        "cash_rate": 0.0,
        "policy": policy.dict(),
        "rebalance": {
            "bayesian_only": True,
            "strategies": ["drift_band"],
            "params": {},
        },
        "risk_target": overrides["risk_target"],
        "column_mapping": column_mapping,
        "preset_name": preset_name,
        "portfolio": {"weighting_scheme": overrides["weighting_scheme"]},
    }

    config_state = {
        "preset_name": preset_name,
        "preset_config": preset,
        "column_mapping": column_mapping,
        "custom_overrides": overrides,
        "validation_errors": [],
        "is_valid": True,
    }

    returns_reset = returns_df.reset_index().rename(columns={returns_df.index.name or "index": "Date"})

    metrics_registry = pipeline_metrics or ["sharpe_ratio", "annual_return"]
    rank_metric = metrics_registry[0]
    cfg = Config(
        version="1",
        data={},
        preprocessing={},
        vol_adjust={"target_vol": overrides["risk_target"]},
        sample_split={
            "in_start": (start - pd.DateOffset(months=overrides["lookback_months"])).strftime("%Y-%m"),
            "in_end": (start - pd.DateOffset(months=1)).strftime("%Y-%m"),
            "out_start": start.strftime("%Y-%m"),
            "out_end": end.strftime("%Y-%m"),
        },
        portfolio={
            "selection_mode": "rank",
            "rank": {
                "inclusion_approach": "top_n",
                "n": overrides["selection_count"],
                "score_by": rank_metric,
            },
            "weighting_scheme": overrides["weighting_scheme"],
        },
        benchmarks={"spx": benchmark_col} if benchmark_col else {},
        metrics={"registry": metrics_registry},
        export={},
        run={},
    )

    try:
        result = run_simulation(cfg, returns_reset)
    except Exception as exc:  # pragma: no cover - runtime safety
        raise DemoRunError(f"Demo pipeline failed: {exc}") from exc
    adapted = adapt_run_result(result)

    session_state["returns_df"] = returns_df
    session_state["schema_meta"] = meta
    session_state["benchmark_candidates"] = benchmark_candidates
    session_state["upload_status"] = "success"
    session_state["config_state"] = config_state
    session_state["validation_messages"] = []
    session_state["sim_config"] = sim_config
    session_state["sim_results"] = adapted
    session_state["demo_metadata"] = {
        "preset": preset_name,
        "dataset": dataset_path.name,
    }

    return adapted


__all__ = ["run_one_click_demo", "DemoRunError"]
