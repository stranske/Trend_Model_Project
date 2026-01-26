"""Helpers to execute the Trend analysis pipeline from the Streamlit UI."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from trend_analysis.config.bridge import build_config_payload, validate_payload
from trend_analysis.config.legacy import Config
from trend_analysis.config.ui_mapping import METRIC_REGISTRY, build_config_from_ui_state
from trend_analysis.signals import TrendSpec as TrendSpecModel
from utils.paths import proj_path

from .data_cache import cache_key_for_frame
from .guardrails import infer_frequency
from .upload_guard import DEFAULT_UPLOAD_DIR


@dataclass(frozen=True)
class ModelSettings:
    """Compatibility shim for legacy demo code expecting ``ModelSettings``."""

    lookback_periods: int
    rebalance_frequency: str
    selection_count: int
    risk_target: float
    weighting_scheme: str
    cooldown_periods: int
    min_history_periods: int
    metric_weights: Mapping[str, float]
    trend_spec: Mapping[str, Any]
    benchmark: str | None = None


@dataclass
class AnalysisPayload:
    """Container describing the data required to run the analysis."""

    returns: pd.DataFrame
    model_state: Mapping[str, Any]
    benchmark: str | None


def _coerce_positive_int(value: Any, *, default: int, minimum: int = 1) -> int:
    try:
        as_int = int(value)
    except (TypeError, ValueError):
        return default
    return max(as_int, minimum)


def _coerce_positive_float(value: Any, *, default: float) -> float:
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return default
    return max(as_float, 0.0)


def _month_end(ts: pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(ts)
    period = stamp.to_period("M")
    return period.to_timestamp("M", how="end")


def _build_sample_split(
    index: pd.DatetimeIndex, config: Mapping[str, Any]
) -> dict[str, str]:
    if index.empty:
        raise ValueError("Dataset is empty")

    # Check if user specified explicit date mode
    date_mode = config.get("date_mode", "relative")

    if date_mode == "explicit":
        # User has specified explicit start/end dates
        user_start = config.get("start_date")
        user_end = config.get("end_date")

        if not user_start or not user_end:
            raise ValueError(
                "Explicit date mode requires both start_date and end_date to be specified"
            )

        # Parse user dates
        try:
            start_ts = pd.Timestamp(user_start)
            end_ts = pd.Timestamp(user_end)
        except (ValueError, TypeError):
            # Fall back to relative mode on parse error
            pass
        else:
            # Clamp to data boundaries
            data_start = index.min()
            data_end = index.max()
            start_ts = max(start_ts, data_start)
            end_ts = min(end_ts, data_end)

            # Convert periods to months based on frequency
            frequency = str(config.get("multi_period_frequency", "A") or "A")
            period_to_months = {"M": 1, "Q": 3, "A": 12}
            months_per_period = period_to_months.get(frequency, 12)

            lookback_periods = _coerce_positive_int(
                config.get("lookback_periods"),
                default=3 if frequency != "M" else 36,
                minimum=1,
            )
            lookback_months = lookback_periods * months_per_period

            # Calculate date boundaries for explicit mode
            # out_start and out_end come from user-specified dates
            out_start = _month_end(start_ts)
            out_end = _month_end(end_ts)

            # in_end is one month before out_start
            in_end = _month_end(out_start - pd.DateOffset(months=1))
            if in_end < index.min():
                in_end = _month_end(index.min())

            # in_start is lookback_months before in_end
            in_start = _month_end(in_end - pd.DateOffset(months=lookback_months - 1))
            if in_start < index.min():
                in_start = _month_end(index.min())

            return {
                "in_start": in_start.strftime("%Y-%m"),
                "in_end": in_end.strftime("%Y-%m"),
                "out_start": out_start.strftime("%Y-%m"),
                "out_end": out_end.strftime("%Y-%m"),
            }

    # Relative mode (default): compute from lookback/evaluation windows
    # Convert periods to months based on frequency
    frequency = str(config.get("multi_period_frequency", "A") or "A")
    period_to_months = {"M": 1, "Q": 3, "A": 12}
    months_per_period = period_to_months.get(frequency, 12)

    lookback_periods = _coerce_positive_int(
        config.get("lookback_periods"),
        default=3 if frequency != "M" else 36,
        minimum=1,
    )
    evaluation_periods = _coerce_positive_int(
        config.get("evaluation_periods"),
        default=1 if frequency != "M" else 12,
        minimum=1,
    )
    lookback_months = lookback_periods * months_per_period
    evaluation_months = evaluation_periods * months_per_period

    last = _month_end(index.max())
    first = _month_end(index.min())
    out_start = _month_end(last - pd.DateOffset(months=evaluation_months - 1))
    if out_start < first:
        out_start = first
    in_end = _month_end(out_start - pd.DateOffset(months=1))
    if in_end < first:
        in_end = first
    in_start = _month_end(in_end - pd.DateOffset(months=lookback_months - 1))
    if in_start < first:
        in_start = first

    return {
        "in_start": in_start.strftime("%Y-%m"),
        "in_end": in_end.strftime("%Y-%m"),
        "out_start": out_start.strftime("%Y-%m"),
        "out_end": last.strftime("%Y-%m"),
    }


def _build_signals_config(config: Mapping[str, Any]) -> dict[str, Any]:
    base = TrendSpecModel()
    window = _coerce_positive_int(config.get("window"), default=base.window)
    lag = _coerce_positive_int(config.get("lag"), default=base.lag)
    min_periods_raw = config.get("min_periods")
    try:
        min_periods = (
            int(min_periods_raw) if min_periods_raw not in (None, "") else None
        )
    except (TypeError, ValueError):
        min_periods = None
    if min_periods is not None and min_periods <= 0:
        min_periods = None
    if min_periods is not None and min_periods > window:
        min_periods = window

    vol_adjust = bool(config.get("vol_adjust", base.vol_adjust))
    vol_target_raw = config.get("vol_target")
    try:
        vol_target = float(vol_target_raw) if vol_target_raw is not None else None
    except (TypeError, ValueError):
        vol_target = None
    if vol_target is not None and vol_target <= 0:
        vol_target = None
    if not vol_adjust:
        vol_target = None

    zscore = bool(config.get("zscore", base.zscore))

    payload: dict[str, Any] = {
        "kind": base.kind,
        "window": window,
        "lag": lag,
        "vol_adjust": vol_adjust,
        "zscore": zscore,
    }
    if min_periods is not None:
        payload["min_periods"] = min_periods
    if vol_target is not None:
        payload["vol_target"] = vol_target
    return payload


def _normalise_metric_weights(raw: Mapping[str, Any]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for name, value in raw.items():
        if name not in METRIC_REGISTRY:
            continue
        try:
            weight = float(value)
        except (TypeError, ValueError):
            continue
        if weight <= 0:
            continue
        weights[name] = weight
    if not weights:
        default = 1.0 / 3
        return {
            "sharpe": default,
            "return_ann": default,
            "drawdown": default,
        }
    total = sum(weights.values())
    return {name: weight / total for name, weight in weights.items()}


def _build_portfolio_config(
    config: Mapping[str, Any], weights: Mapping[str, float]
) -> dict[str, Any]:
    selection_count = _coerce_positive_int(
        config.get("selection_count"), default=10, minimum=1
    )
    weighting_scheme = str(config.get("weighting_scheme", "equal") or "equal")
    registry_weights = {
        METRIC_REGISTRY.get(metric, metric): float(weight)
        for metric, weight in weights.items()
    }

    # Advanced settings
    max_weight = _coerce_positive_float(config.get("max_weight"), default=0.20)
    max_turnover = _coerce_positive_float(config.get("max_turnover"), default=1.0)
    transaction_cost_bps = _coerce_positive_int(
        config.get("transaction_cost_bps"), default=0, minimum=0
    )
    rebalance_freq = str(config.get("rebalance_freq", "M") or "M")

    # Fund holding rules (Phase 3)
    min_tenure_periods = _coerce_positive_int(
        config.get("min_tenure_periods"), default=0, minimum=0
    )
    max_changes_per_period = _coerce_positive_int(
        config.get("max_changes_per_period"), default=0, minimum=0
    )
    max_active_positions = _coerce_positive_int(
        config.get("max_active_positions"), default=0, minimum=0
    )

    # Phase 8: Selection approach settings (accept both naming conventions)
    selection_approach = str(
        config.get("inclusion_approach") or config.get("selection_approach") or "top_n"
    )
    # Buy & Hold mode uses a sub-selection method for initial/replacement selection
    is_buy_and_hold = selection_approach == "buy_and_hold"
    buy_hold_initial = str(config.get("buy_hold_initial", "top_n"))
    # Transform is now implicit: threshold mode uses zscore, ranking modes use none
    # For buy_and_hold, use the initial method's transform
    effective_approach = buy_hold_initial if is_buy_and_hold else selection_approach
    rank_transform = "zscore" if effective_approach == "threshold" else "raw"
    slippage_bps = _coerce_positive_int(
        config.get("slippage_bps"), default=0, minimum=0
    )
    bottom_k = _coerce_positive_int(config.get("bottom_k"), default=0, minimum=0)

    # Phase 9: Selection approach parameters
    rank_pct = _coerce_positive_float(config.get("rank_pct"), default=0.10)
    # For threshold mode, use z_entry_soft as the threshold
    rank_threshold = _coerce_positive_float(
        config.get("z_entry_soft") or config.get("rank_threshold"), default=1.0
    )

    # Phase 15: Constraints
    long_only = bool(config.get("long_only", True))

    # Determine selection mode based on approach
    is_random_mode = selection_approach == "random"
    # For buy_and_hold with random initial, also set random mode
    if is_buy_and_hold and buy_hold_initial == "random":
        is_random_mode = True
    selection_mode = "random" if is_random_mode else "rank"
    # Override selection_mode for buy_and_hold
    if is_buy_and_hold:
        selection_mode = "buy_and_hold"

    portfolio_cfg: dict[str, Any] = {
        "selection_mode": selection_mode,
        "rank": {
            "inclusion_approach": selection_approach,
            "n": selection_count,
            "pct": rank_pct,
            "threshold": rank_threshold,
            "score_by": "blended",
            "blended_weights": registry_weights,
            "transform": rank_transform,
        },
        "buy_and_hold": {
            "initial_method": buy_hold_initial,
            "n": selection_count,
            "pct": rank_pct,
            "threshold": rank_threshold,
            "blended_weights": registry_weights,
        },
        "random_n": selection_count,  # Used when selection_mode is "random"
        "weighting_scheme": weighting_scheme,
        "rebalance_freq": rebalance_freq,
        "max_turnover": max_turnover,
        "transaction_cost_bps": transaction_cost_bps,
        "constraints": {
            "long_only": long_only,
            "max_weight": max_weight,
        },
    }

    # Add slippage_bps to cost_model if specified
    if slippage_bps > 0:
        portfolio_cfg["cost_model"] = {
            "bps_per_trade": transaction_cost_bps,
            "slippage_bps": slippage_bps,
        }

    # Add bottom_k exclusion if specified
    if bottom_k > 0:
        portfolio_cfg["rank"]["bottom_k"] = bottom_k

    # Add fund holding rules if set (0 means unlimited/disabled)
    if min_tenure_periods > 0:
        portfolio_cfg["min_tenure_n"] = min_tenure_periods
    if max_changes_per_period > 0:
        portfolio_cfg["turnover_budget_max_changes"] = max_changes_per_period
    if max_active_positions > 0:
        portfolio_cfg.setdefault("constraints", {})
        portfolio_cfg["constraints"]["max_active_positions"] = max_active_positions

    return portfolio_cfg


def _build_config(payload: AnalysisPayload) -> Config:
    frequency = _resolve_frequency(payload.returns)
    csv_path = _ensure_validation_csv_path(payload.returns)
    return build_config_from_ui_state(
        returns=payload.returns,
        model_state=payload.model_state,
        benchmark=payload.benchmark,
        frequency=frequency,
        csv_path=csv_path,
    )


def _prepare_returns(df: pd.DataFrame) -> pd.DataFrame:
    reset = df.reset_index()
    index_name = df.index.name or "Date"
    return reset.rename(columns={index_name: "Date"})


def _resolve_frequency(returns: pd.DataFrame) -> str:
    meta = st.session_state.get("schema_meta")
    if isinstance(meta, Mapping):
        freq = meta.get("frequency_code") or meta.get("frequency")
        if isinstance(freq, str) and freq.strip():
            return freq.strip().upper()
    return infer_frequency(returns.index)


def _ensure_validation_csv_path(returns: pd.DataFrame) -> str | None:
    candidate = st.session_state.get("data_saved_path") or st.session_state.get(
        "uploaded_file_path"
    )
    if isinstance(candidate, str) and candidate:
        path = Path(candidate)
        if path.exists() and path.suffix.lower() == ".csv":
            return str(path)

    try:
        upload_dir = DEFAULT_UPLOAD_DIR
        upload_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(
            cache_key_for_frame(returns).encode("utf-8")
        ).hexdigest()[:12]
        target = upload_dir / f"streamlit-returns-{digest}.csv"
        if not target.exists():
            _prepare_returns(returns).to_csv(target, index=False)
        return str(target)
    except Exception:
        return None


def _validate_streamlit_payload(payload: AnalysisPayload) -> None:
    state = payload.model_state
    date_column = "Date"
    frequency = _resolve_frequency(payload.returns)
    csv_path = _ensure_validation_csv_path(payload.returns)
    rebalance_calendar = "NYSE"
    max_turnover = _coerce_positive_float(state.get("max_turnover"), default=1.0)
    transaction_cost_bps = _coerce_positive_float(
        state.get("transaction_cost_bps"), default=0.0
    )
    slippage_bps = _coerce_positive_float(state.get("slippage_bps"), default=0.0)
    target_vol = _coerce_positive_float(state.get("risk_target"), default=0.1)

    payload_dict = build_config_payload(
        csv_path=csv_path,
        universe_membership_path=None,
        managers_glob=None,
        date_column=date_column,
        frequency=frequency,
        rebalance_calendar=rebalance_calendar,
        max_turnover=max_turnover,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        target_vol=target_vol,
    )

    base_dir = Path(csv_path).parent if csv_path else proj_path()
    _, validation_error = validate_payload(payload_dict, base_path=base_dir)
    if validation_error:
        raise ValueError(f"Config validation failed:\n{validation_error}")


def _execute_analysis(payload: AnalysisPayload):
    from trend_analysis.api import run_simulation

    config = _build_config(payload)
    _validate_streamlit_payload(payload)
    returns = _prepare_returns(payload.returns)
    return run_simulation(config, returns)


def _hashable_model_state(state: Mapping[str, Any]) -> str:
    return json.dumps(state, sort_keys=True, default=str)


@st.cache_data(
    show_spinner="Running analysis…", hash_funcs={pd.DataFrame: cache_key_for_frame}
)
def run_cached_analysis(
    returns: pd.DataFrame,
    model_state_blob: str,
    benchmark: str | None,
    data_hash: str,
):
    """
    Run the analysis pipeline with caching.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame containing asset returns, indexed by date.
    model_state_blob : str
        JSON-serialized model state containing analysis configuration.
    benchmark : str or None
        Optional benchmark identifier for the analysis.

    Returns
    -------
    Any
        The result of the analysis pipeline, as returned by `run_simulation`.
    """
    model_state = json.loads(model_state_blob)
    payload = AnalysisPayload(
        returns=returns,
        model_state=model_state,
        benchmark=benchmark,
    )
    return _execute_analysis(payload)


def run_analysis(
    df: pd.DataFrame,
    model_state: Mapping[str, Any],
    benchmark: str | None,
    *,
    data_hash: str | None = None,
):
    """Execute the cached analysis pipeline."""

    blob = _hashable_model_state(model_state)
    effective_hash = data_hash or cache_key_for_frame(df)
    return run_cached_analysis(df, blob, benchmark, effective_hash)


def clear_cached_analysis() -> None:
    """Invalidate any cached analysis results."""

    clear_fn = getattr(run_cached_analysis, "clear", None)
    if callable(clear_fn):
        clear_fn()
