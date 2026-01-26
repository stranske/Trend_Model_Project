"""Shared mapping from Streamlit UI state to core Config."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from trend_analysis.config.legacy import Config
from trend_analysis.signals import TrendSpec as TrendSpecModel

METRIC_REGISTRY = {
    "sharpe": "Sharpe",
    "return_ann": "AnnualReturn",
    "sortino": "Sortino",
    "info_ratio": "InformationRatio",
    "drawdown": "MaxDrawdown",
    "vol": "Volatility",
}


def coerce_positive_int(value: Any, *, default: int, minimum: int = 1) -> int:
    try:
        as_int = int(value)
    except (TypeError, ValueError):
        return default
    return max(as_int, minimum)


def coerce_positive_float(value: Any, *, default: float) -> float:
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return default
    return max(as_float, 0.0)


def month_end(ts: pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(ts)
    period = stamp.to_period("M")
    return period.to_timestamp("M", how="end")


def build_sample_split(index: pd.DatetimeIndex, config: Mapping[str, Any]) -> dict[str, str]:
    if index.empty:
        raise ValueError("Dataset is empty")

    date_mode = config.get("date_mode", "relative")

    if date_mode == "explicit":
        user_start = config.get("start_date")
        user_end = config.get("end_date")

        if not user_start or not user_end:
            raise ValueError(
                "Explicit date mode requires both start_date and end_date to be specified"
            )

        try:
            start_ts = pd.Timestamp(user_start)
            end_ts = pd.Timestamp(user_end)
        except (ValueError, TypeError):
            pass
        else:
            data_start = index.min()
            data_end = index.max()
            start_ts = max(start_ts, data_start)
            end_ts = min(end_ts, data_end)

            frequency = str(config.get("multi_period_frequency", "A") or "A")
            period_to_months = {"M": 1, "Q": 3, "A": 12}
            months_per_period = period_to_months.get(frequency, 12)

            lookback_periods = coerce_positive_int(
                config.get("lookback_periods"),
                default=3 if frequency != "M" else 36,
                minimum=1,
            )
            lookback_months = lookback_periods * months_per_period

            out_start = month_end(start_ts)
            out_end = month_end(end_ts)

            in_end = month_end(out_start - pd.DateOffset(months=1))
            if in_end < index.min():
                in_end = month_end(index.min())

            in_start = month_end(in_end - pd.DateOffset(months=lookback_months - 1))
            if in_start < index.min():
                in_start = month_end(index.min())

            return {
                "in_start": in_start.strftime("%Y-%m"),
                "in_end": in_end.strftime("%Y-%m"),
                "out_start": out_start.strftime("%Y-%m"),
                "out_end": out_end.strftime("%Y-%m"),
            }

    frequency = str(config.get("multi_period_frequency", "A") or "A")
    period_to_months = {"M": 1, "Q": 3, "A": 12}
    months_per_period = period_to_months.get(frequency, 12)

    lookback_periods = coerce_positive_int(
        config.get("lookback_periods"),
        default=3 if frequency != "M" else 36,
        minimum=1,
    )
    evaluation_periods = coerce_positive_int(
        config.get("evaluation_periods"),
        default=1 if frequency != "M" else 12,
        minimum=1,
    )
    lookback_months = lookback_periods * months_per_period
    evaluation_months = evaluation_periods * months_per_period

    last = month_end(index.max())
    first = month_end(index.min())
    out_start = month_end(last - pd.DateOffset(months=evaluation_months - 1))
    if out_start < first:
        out_start = first
    in_end = month_end(out_start - pd.DateOffset(months=1))
    if in_end < first:
        in_end = first
    in_start = month_end(in_end - pd.DateOffset(months=lookback_months - 1))
    if in_start < first:
        in_start = first

    return {
        "in_start": in_start.strftime("%Y-%m"),
        "in_end": in_end.strftime("%Y-%m"),
        "out_start": out_start.strftime("%Y-%m"),
        "out_end": last.strftime("%Y-%m"),
    }


def build_signals_config(config: Mapping[str, Any]) -> dict[str, Any]:
    base = TrendSpecModel()
    window = coerce_positive_int(config.get("window"), default=base.window)
    lag = coerce_positive_int(config.get("lag"), default=base.lag)
    min_periods_raw = config.get("min_periods")
    min_periods: int | None
    if min_periods_raw in (None, ""):
        min_periods = None
    elif isinstance(min_periods_raw, (int, float, str)):
        try:
            min_periods = int(min_periods_raw)
        except (TypeError, ValueError):
            min_periods = None
    else:
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


def normalise_metric_weights(raw: Mapping[str, Any]) -> dict[str, float]:
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


def build_portfolio_config(
    config: Mapping[str, Any], weights: Mapping[str, float]
) -> dict[str, Any]:
    selection_count = coerce_positive_int(config.get("selection_count"), default=10, minimum=1)
    weighting_scheme = str(config.get("weighting_scheme", "equal") or "equal")
    registry_weights = {
        METRIC_REGISTRY.get(metric, metric): float(weight) for metric, weight in weights.items()
    }

    max_weight = coerce_positive_float(config.get("max_weight"), default=0.20)
    max_turnover = coerce_positive_float(config.get("max_turnover"), default=1.0)
    transaction_cost_bps = coerce_positive_int(
        config.get("transaction_cost_bps"), default=0, minimum=0
    )
    rebalance_freq = str(config.get("rebalance_freq", "M") or "M")

    min_tenure_periods = coerce_positive_int(config.get("min_tenure_periods"), default=0, minimum=0)
    max_changes_per_period = coerce_positive_int(
        config.get("max_changes_per_period"), default=0, minimum=0
    )
    max_active_positions = coerce_positive_int(
        config.get("max_active_positions"), default=0, minimum=0
    )

    selection_approach = str(
        config.get("inclusion_approach") or config.get("selection_approach") or "top_n"
    )
    is_buy_and_hold = selection_approach == "buy_and_hold"
    buy_hold_initial = str(config.get("buy_hold_initial", "top_n"))
    effective_approach = buy_hold_initial if is_buy_and_hold else selection_approach
    rank_transform = "zscore" if effective_approach == "threshold" else "raw"
    slippage_bps = coerce_positive_int(config.get("slippage_bps"), default=0, minimum=0)
    bottom_k = coerce_positive_int(config.get("bottom_k"), default=0, minimum=0)

    rank_pct = coerce_positive_float(config.get("rank_pct"), default=0.10)
    rank_threshold = coerce_positive_float(
        config.get("z_entry_soft") or config.get("rank_threshold"), default=1.0
    )

    long_only = bool(config.get("long_only", True))

    is_random_mode = selection_approach == "random"
    if is_buy_and_hold and buy_hold_initial == "random":
        is_random_mode = True
    selection_mode = "random" if is_random_mode else "rank"
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
        "random_n": selection_count,
        "weighting_scheme": weighting_scheme,
        "rebalance_freq": rebalance_freq,
        "max_turnover": max_turnover,
        "transaction_cost_bps": transaction_cost_bps,
        "constraints": {
            "long_only": long_only,
            "max_weight": max_weight,
        },
    }

    if slippage_bps > 0:
        portfolio_cfg["cost_model"] = {
            "bps_per_trade": transaction_cost_bps,
            "slippage_bps": slippage_bps,
        }

    if bottom_k > 0:
        portfolio_cfg["rank"]["bottom_k"] = bottom_k

    if min_tenure_periods > 0:
        portfolio_cfg["min_tenure_n"] = min_tenure_periods
    if max_changes_per_period > 0:
        portfolio_cfg["turnover_budget_max_changes"] = max_changes_per_period
    if max_active_positions > 0:
        portfolio_cfg.setdefault("constraints", {})
        portfolio_cfg["constraints"]["max_active_positions"] = max_active_positions

    return portfolio_cfg


def build_config_from_ui_state(
    *,
    returns: pd.DataFrame,
    model_state: Mapping[str, Any],
    benchmark: str | None,
    frequency: str,
    csv_path: str | None,
) -> Config:
    weights = normalise_metric_weights(model_state.get("metric_weights", {}))
    index = returns.index
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)
    sample_split = build_sample_split(index, model_state)
    vol_target = coerce_positive_float(model_state.get("risk_target"), default=0.1)

    vol_adjust_enabled = bool(model_state.get("vol_adjust_enabled", True))
    vol_floor = coerce_positive_float(model_state.get("vol_floor"), default=0.015)
    warmup_periods = coerce_positive_int(model_state.get("warmup_periods"), default=0, minimum=0)
    vol_window_length = coerce_positive_int(
        model_state.get("vol_window_length"), default=63, minimum=1
    )
    vol_window_decay = str(model_state.get("vol_window_decay", "ewma") or "ewma").lower()
    if vol_window_decay == "constant":
        vol_window_decay = "simple"
    if vol_window_decay not in {"ewma", "simple"}:
        vol_window_decay = "ewma"
    vol_ewma_lambda = coerce_positive_float(model_state.get("vol_ewma_lambda"), default=0.94)
    if not (0.0 < vol_ewma_lambda < 1.0):
        vol_ewma_lambda = 0.94
    rf_override_enabled = bool(model_state.get("rf_override_enabled", False))
    rf_rate_annual = coerce_positive_float(model_state.get("rf_rate_annual"), default=0.0)

    trend_spec = {
        "window": model_state.get("trend_window"),
        "lag": model_state.get("trend_lag"),
        "min_periods": model_state.get("trend_min_periods"),
        "zscore": model_state.get("trend_zscore"),
        "vol_adjust": model_state.get("trend_vol_adjust"),
        "vol_target": model_state.get("trend_vol_target"),
    }
    signals_cfg = build_signals_config(trend_spec)

    portfolio_cfg = build_portfolio_config(model_state, weights)

    mp_max_funds = coerce_positive_int(model_state.get("mp_max_funds"), default=0, minimum=0)
    if mp_max_funds > 0:
        portfolio_cfg.setdefault("constraints", {})
        portfolio_cfg["constraints"]["max_funds"] = mp_max_funds

    mp_min_funds = coerce_positive_int(model_state.get("mp_min_funds"), default=0, minimum=0)
    if mp_min_funds > 0:
        portfolio_cfg.setdefault("constraints", {})
        portfolio_cfg["constraints"]["min_funds"] = mp_min_funds

    min_weight_raw = model_state.get("min_weight")
    if min_weight_raw is not None:
        min_weight = coerce_positive_float(min_weight_raw, default=0.05)
        portfolio_cfg.setdefault("constraints", {})
        portfolio_cfg["constraints"]["min_weight"] = min_weight

    min_weight_strikes = coerce_positive_int(model_state.get("min_weight_strikes"), default=0)
    if min_weight_strikes > 0:
        portfolio_cfg.setdefault("constraints", {})
        portfolio_cfg["constraints"]["min_weight_strikes"] = min_weight_strikes

    cooldown_periods = coerce_positive_int(model_state.get("cooldown_periods"), default=0)
    if cooldown_periods > 0:
        portfolio_cfg["cooldown_periods"] = cooldown_periods

    metrics_registry = [METRIC_REGISTRY.get(name, name) for name in weights]

    benchmark_map: dict[str, str] = {}
    if benchmark:
        benchmark_map[benchmark] = benchmark

    random_seed_raw = model_state.get("random_seed")
    seed = 42
    try:
        if random_seed_raw is not None:
            seed = int(random_seed_raw)
    except (TypeError, ValueError):
        seed = 42

    preset_name = model_state.get("preset") or model_state.get("trend_spec_preset")

    regime_enabled = bool(model_state.get("regime_enabled", False))
    regime_proxy = str(model_state.get("regime_proxy", "SPX") or "SPX")
    regime_cfg = {
        "enabled": regime_enabled,
        "proxy": regime_proxy,
    }

    shrinkage_enabled = bool(model_state.get("shrinkage_enabled", True))
    shrinkage_method = str(model_state.get("shrinkage_method", "ledoit_wolf") or "ledoit_wolf")

    robustness_cfg = {
        "shrinkage": {
            "enabled": shrinkage_enabled,
            "method": shrinkage_method,
        },
    }

    condition_threshold = float(model_state.get("condition_threshold", 1.0e12) or 1.0e12)
    safe_mode = str(model_state.get("safe_mode", "hrp") or "hrp")
    robustness_cfg["condition_check"] = {
        "enabled": True,
        "threshold": condition_threshold,
        "safe_mode": safe_mode,
    }

    z_entry_soft = float(model_state.get("z_entry_soft", 1.0) or 1.0)
    z_exit_soft = float(model_state.get("z_exit_soft", -1.0) or -1.0)
    soft_strikes = int(model_state.get("soft_strikes", 2) or 2)
    entry_soft_strikes = int(model_state.get("entry_soft_strikes", 1) or 1)
    sticky_add_periods = int(model_state.get("sticky_add_periods", 1) or 1)
    sticky_drop_periods = int(model_state.get("sticky_drop_periods", 1) or 1)
    ci_level = float(model_state.get("ci_level", 0.0) or 0.0)

    z_entry_hard_val = model_state.get("z_entry_hard")
    z_exit_hard_val = model_state.get("z_exit_hard")
    z_entry_hard = float(z_entry_hard_val) if z_entry_hard_val is not None else None
    z_exit_hard = float(z_exit_hard_val) if z_exit_hard_val is not None else None

    threshold_hold_cfg: dict[str, Any] = {
        "z_entry_soft": z_entry_soft,
        "z_exit_soft": z_exit_soft,
        "soft_strikes": soft_strikes,
        "entry_soft_strikes": entry_soft_strikes,
    }

    selection_count = coerce_positive_int(model_state.get("selection_count"), default=10)
    threshold_hold_cfg["metric"] = "blended"
    threshold_hold_cfg["blended_weights"] = {
        METRIC_REGISTRY.get(metric, metric): float(weight) for metric, weight in weights.items()
    }
    threshold_hold_cfg["target_n"] = selection_count
    if z_entry_hard is not None:
        threshold_hold_cfg["z_entry_hard"] = z_entry_hard
    if z_exit_hard is not None:
        threshold_hold_cfg["z_exit_hard"] = z_exit_hard
    min_tenure_periods = coerce_positive_int(model_state.get("min_tenure_periods"), default=0)
    if min_tenure_periods > 0:
        threshold_hold_cfg["min_tenure_n"] = min_tenure_periods

    portfolio_cfg["policy"] = "threshold_hold"
    portfolio_cfg["threshold_hold"] = threshold_hold_cfg
    portfolio_cfg["sticky_add_x"] = sticky_add_periods
    portfolio_cfg["sticky_drop_y"] = sticky_drop_periods
    portfolio_cfg["ci_level"] = ci_level

    multi_period_enabled = bool(model_state.get("multi_period_enabled", False))
    multi_period_cfg = None
    if multi_period_enabled:
        multi_period_frequency = str(model_state.get("multi_period_frequency", "A") or "A")
        in_sample_len = coerce_positive_int(
            model_state.get("lookback_periods")
            or model_state.get("in_sample_years")
            or model_state.get("multi_period_in_sample_years"),
            default=3,
            minimum=1,
        )
        out_sample_len = coerce_positive_int(
            model_state.get("evaluation_periods")
            or model_state.get("out_sample_years")
            or model_state.get("multi_period_out_sample_years"),
            default=1,
            minimum=1,
        )

        min_history_len = coerce_positive_int(
            model_state.get("min_history_periods"),
            default=in_sample_len,
            minimum=1,
        )
        min_history_len = min(min_history_len, in_sample_len)

        data_index = returns.index
        data_start = data_index.min()
        data_end = data_index.max()

        user_start = model_state.get("start_date")
        user_end = model_state.get("end_date")

        date_mode = str(model_state.get("date_mode", "relative") or "relative").lower()

        if user_start:
            try:
                sim_start = pd.Timestamp(user_start)
            except (ValueError, TypeError):
                sim_start = data_start
        else:
            sim_start = data_start

        if user_end:
            try:
                sim_end = pd.Timestamp(user_end)
            except (ValueError, TypeError):
                sim_end = data_end
        else:
            sim_end = data_end

        start_me = month_end(sim_start)
        end_me = month_end(sim_end)
        start_str = start_me.strftime("%Y-%m-%d")
        end_str = end_me.strftime("%Y-%m-%d")

        multi_period_cfg = {
            "frequency": multi_period_frequency,
            "in_sample_len": in_sample_len,
            "out_sample_len": out_sample_len,
            "min_history_periods": min_history_len,
            "start": start_str,
            "end": end_str,
            "start_mode": "oos" if date_mode == "explicit" else "in",
        }

    missing_policy = model_state.get("missing_policy")
    missing_limit = model_state.get("missing_limit")
    data_cfg: dict[str, Any] = {
        "allow_risk_free_fallback": True,
        "date_column": "Date",
        "frequency": frequency,
        "missing_policy": missing_policy or "ffill",
    }
    if missing_limit is not None:
        data_cfg["missing_limit"] = missing_limit
    if csv_path:
        data_cfg["csv_path"] = csv_path

    risk_free_column = model_state.get("risk_free_column")
    if isinstance(risk_free_column, str) and risk_free_column.strip():
        data_cfg["risk_free_column"] = risk_free_column.strip()

    preprocessing_cfg: dict[str, Any] = {}

    portfolio_cfg.setdefault("rebalance_calendar", "NYSE")

    return Config(
        version="1",
        data=data_cfg,
        preprocessing=preprocessing_cfg,
        vol_adjust={
            "enabled": vol_adjust_enabled,
            "target_vol": vol_target,
            "floor_vol": vol_floor,
            "warmup_periods": warmup_periods,
            "window": {
                "length": vol_window_length,
                "decay": vol_window_decay,
                "lambda": vol_ewma_lambda,
            },
        },
        sample_split=sample_split,
        portfolio=portfolio_cfg,
        signals=signals_cfg,
        benchmarks=benchmark_map,
        regime=regime_cfg,
        robustness=robustness_cfg,
        metrics={
            "registry": metrics_registry,
            "rf_rate_annual": rf_rate_annual,
            "rf_override_enabled": rf_override_enabled,
        },
        export={},
        run={"trend_preset": preset_name},
        seed=seed,
        multi_period=multi_period_cfg,
    )
