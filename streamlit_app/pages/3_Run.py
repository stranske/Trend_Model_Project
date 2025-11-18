from __future__ import annotations

import traceback
import uuid
from collections.abc import Mapping
from typing import Any, Callable, Dict

import pandas as pd

from streamlit_app.components.disclaimer import show_disclaimer
from streamlit_app.components.guardrails import (
    estimate_resource_usage,
    prepare_dry_run_plan,
)
from trend_analysis.logging import get_default_log_path, init_run_logger, log_step
from trend_analysis.signals import TrendSpec


class StreamlitConfig:
    def __init__(self, **data: object) -> None:
        self.__dict__.update(data)

    def model_dump(self) -> Dict[str, object]:
        return dict(self.__dict__)


Config = StreamlitConfig

RunSimulationFn = Callable[[StreamlitConfig, pd.DataFrame], Any]
run_simulation: RunSimulationFn | None = None


def _resolve_run_simulation() -> RunSimulationFn:
    """Return the configured run_simulation callable, importing lazily if needed."""

    if callable(run_simulation):
        return run_simulation  # type: ignore[return-value]
    from trend_analysis.api import run_simulation as run_simulation_impl

    return run_simulation_impl


def _make_config(config_data: Dict[str, object]) -> StreamlitConfig:
    return Config(**config_data)


def format_error_message(error: Exception) -> str:
    """Return a user-friendly description for common analysis errors."""

    error_type = type(error).__name__
    error_msg = str(error)

    error_mappings = {
        "KeyError": "Missing required data field",
        "ValueError": "Invalid data or configuration value",
        "FileNotFoundError": "Required file not found",
        "PermissionError": "Access denied to file or directory",
        "IsADirectoryError": "Expected a file but found a directory",
        "ImportError": "Missing required dependency",
        "MemoryError": "Insufficient memory for analysis",
        "TimeoutError": "Analysis took too long to complete",
    }

    lowered = error_msg.lower()
    if "date" in lowered:
        return (
            "Data validation error: Your dataset must include a Date column with "
            "properly formatted dates."
        )
    if "sample_split" in lowered:
        return (
            "Configuration error: Invalid date ranges specified. Please check "
            "your in-sample and out-of-sample periods."
        )
    if "returns" in lowered:
        return (
            "Data error: Invalid returns data format. Please ensure your data "
            "contains numeric return values."
        )
    if "config" in lowered:
        return (
            "Configuration error: Invalid configuration settings. Please review "
            "your analysis parameters."
        )

    if error_type in error_mappings:
        return f"{error_mappings[error_type]}: {error_msg}"

    return f"Analysis error ({error_type}): {error_msg}"


def _render_error_details(error: Exception) -> None:
    """Display a collapsible traceback to aid debugging."""

    import streamlit as st  # Local import to honor patched modules in tests

    details = (
        f"Exception Type: {type(error).__name__}\n\n"
        f"Exception Message:\n{error}\n\n"
        f"Full Traceback:\n{traceback.format_exc()}"
    )
    expander = st.expander("🔍 Show Technical Details", expanded=False)
    try:
        expander.code(details)
    except Exception:  # pragma: no cover - Mock fallback
        st.code(details)


def _coerce_positive_int(value: Any, *, default: int, minimum: int = 1) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return max(default, minimum)
    return max(coerced, minimum)


def _infer_date_bounds(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    try:
        raw_index = pd.to_datetime(df.index, errors="coerce")
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(
            "Unable to derive analysis period from dataset index."
        ) from exc
    valid = raw_index[~pd.isna(raw_index)]
    if len(valid) == 0:
        raise ValueError("Unable to derive analysis period from dataset index.")
    ordered = pd.DatetimeIndex(valid).sort_values()
    return pd.Timestamp(ordered.min()), pd.Timestamp(ordered.max())


def _config_from_model_state(
    model_state: Mapping[str, Any], df: pd.DataFrame
) -> dict[str, Any]:
    start_bounds, end_bounds = _infer_date_bounds(df)
    evaluation_months = _coerce_positive_int(
        model_state.get("evaluation_months"), default=12, minimum=1
    )
    lookback_months = _coerce_positive_int(
        model_state.get("lookback_months"), default=36, minimum=1
    )
    out_start = end_bounds - pd.DateOffset(months=evaluation_months - 1)
    if out_start < start_bounds:
        out_start = start_bounds
    portfolio_cfg = {
        "weighting_scheme": model_state.get("weighting_scheme", "equal"),
    }
    trend_spec = dict(model_state.get("trend_spec", {}))
    return {
        "lookback_months": lookback_months,
        "evaluation_months": evaluation_months,
        "start": out_start.normalize(),
        "end": end_bounds.normalize(),
        "risk_target": model_state.get("risk_target", 0.1),
        "portfolio": portfolio_cfg,
        "trend_spec": trend_spec,
        "signals": trend_spec,
        "trend_preset": model_state.get("trend_spec_preset"),
    }


def main() -> None:
    # Re-import streamlit within the function to ensure any test-time monkeypatches
    # (e.g. replacing ``st.button``) are respected even if the module was previously
    # imported elsewhere.
    import streamlit as st  # noqa: E402 - local import for testability

    st.title("Run")
    # Always render disclaimer + button first so tests can capture state
    accepted = show_disclaimer()
    dry_run_clicked = st.button("Dry run (sample)", disabled=not accepted)
    run_clicked = st.button("Run simulation", type="primary", disabled=not accepted)
    # Robust session_state checks (work in bare mode/tests where session_state
    # may be a Mock that isn't iterable)
    try:
        has_returns = (
            "returns_df" in st.session_state
            and st.session_state.get("returns_df") is not None
        )
        legacy_cfg = st.session_state.get("sim_config")
        model_state = st.session_state.get("model_state")
    except TypeError:
        # Fallback when session_state doesn't support membership tests
        has_returns = getattr(st.session_state, "returns_df", None) is not None
        legacy_cfg = getattr(st.session_state, "sim_config", None)
        model_state = getattr(st.session_state, "model_state", None)
    has_model_state = isinstance(model_state, Mapping) and bool(model_state)
    has_config = legacy_cfg is not None or has_model_state

    if not (has_returns and has_config):
        st.error("Upload data and set configuration first.")
        return
    if not accepted and not dry_run_clicked and not run_clicked:
        return

    try:
        df = st.session_state.get("returns_df")  # type: ignore[attr-defined]
        cfg = st.session_state.get("sim_config")  # type: ignore[attr-defined]
        model_state = st.session_state.get("model_state")  # type: ignore[attr-defined]
    except TypeError:
        # In bare-mode tests, session_state may be a Mock
        df = getattr(st.session_state, "returns_df", None)
        cfg = getattr(st.session_state, "sim_config", None)
        model_state = getattr(st.session_state, "model_state", None)
    if df is None or cfg is None:
        if isinstance(model_state, Mapping) and model_state and df is not None:
            try:
                cfg = _config_from_model_state(model_state, df)
            except ValueError as exc:
                st.error(str(exc))
                return
        else:
            st.error("Upload data and set configuration first.")
            return

    try:
        config_state = st.session_state.get("config_state", {})
        estimate = config_state.get("resource_estimate")
    except (TypeError, AttributeError):  # pragma: no cover - defensive fallback
        estimate = None
    if estimate is None:
        estimate = estimate_resource_usage(df.shape[0], df.shape[1])
    st.session_state["resource_estimate"] = estimate
    st.caption(
        f"Full run estimate: ~{estimate.approx_memory_mb:.1f} MB memory, "
        f"~{estimate.estimated_runtime_s/60:.1f} min runtime."
    )
    for warn in getattr(estimate, "warnings", ()):  # type: ignore[arg-type]
        st.warning(warn)

    returns = df.reset_index().rename(columns={df.index.name or "index": "Date"})

    def cfg_get(d, key, default=None):
        try:
            getter = getattr(d, "get", None)
            if callable(getter):
                return d.get(key, default)
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            return d[key]  # type: ignore[index]
        except Exception:  # pragma: no cover - defensive
            return default

    raw_lookback = cfg_get(cfg, "lookback_months", 0)
    try:
        lookback = int(raw_lookback) if raw_lookback is not None else 0
    except Exception:  # pragma: no cover - defensive
        lookback = 0

    start = cfg_get(cfg, "start")
    end = cfg_get(cfg, "end")
    # Coerce to pandas.Timestamp when possible (handles python date, str, etc.)
    try:
        if start is not None and not isinstance(start, pd.Timestamp):
            start = pd.to_datetime(start)
        if end is not None and not isinstance(end, pd.Timestamp):
            end = pd.to_datetime(end)
    except Exception:  # pragma: no cover - defensive
        start = None
        end = None

    # If dates are unavailable due to mocked state, bail out gracefully
    if start is None or end is None:
        st.error("Missing start/end dates in configuration.")
        return

    portfolio_cfg = {
        "weighting_scheme": cfg_get(cfg, "portfolio", {}).get(
            "weighting_scheme", "equal"
        )
    }

    signals_input = cfg_get(cfg, "signals", {})
    if not isinstance(signals_input, Mapping) or not signals_input:
        signals_input = cfg_get(cfg, "trend_spec", {})
    signals_cfg = _build_signals_config(signals_input)

    config_data = {
        "version": "1",
        "data": {},
        "preprocessing": {},
        "vol_adjust": {"target_vol": cfg_get(cfg, "risk_target", 1.0)},
        "sample_split": {
            "in_start": (start - pd.DateOffset(months=lookback)).strftime("%Y-%m"),
            "in_end": (start - pd.DateOffset(months=1)).strftime("%Y-%m"),
            "out_start": start.strftime("%Y-%m"),
            "out_end": end.strftime("%Y-%m"),
        },
        "portfolio": portfolio_cfg,
        "signals": signals_cfg,
        "metrics": {},
        "export": {},
        "benchmarks": {},
        "run": {"trend_preset": cfg_get(cfg, "trend_preset")},
    }

    run_sim = _resolve_run_simulation()

    if dry_run_clicked and not run_clicked:
        try:
            plan = prepare_dry_run_plan(df, lookback or 0)
        except ValueError as exc:
            st.error(format_error_message(exc))
            _render_error_details(exc)
            return
        dry_returns = plan.frame.reset_index().rename(
            columns={plan.frame.index.name or "index": "Date"}
        )
        dry_run_id = f"dry-{uuid.uuid4().hex[:10]}"
        dry_config = _make_config(
            {
                **config_data,
                "sample_split": plan.sample_split(),
                "run_id": dry_run_id,
            }
        )
        with st.spinner("Running dry run on a small sample..."):
            result = run_sim(dry_config, dry_returns)
        st.session_state["dry_run_results"] = result
        st.session_state["dry_run_summary"] = plan.summary()
        st.success(
            f"Dry run completed on {plan.frame.shape[0]} rows × {plan.frame.shape[1]} columns."
        )
        st.json(plan.summary())
        return

    if not run_clicked:
        return

    run_id = f"run-{uuid.uuid4().hex[:10]}"
    log_path = get_default_log_path(run_id)

    config = _make_config({**config_data, "run_id": run_id})
    run_logger = init_run_logger(run_id, log_path)

    run_logger.info("Starting run")
    run_logger.info("Lookback months: %s", raw_lookback)
    run_logger.info("Run ID: %s", run_id)

    progress = st.progress(0)
    try:
        result = run_sim(config, returns)
    except Exception as exc:
        progress.progress(0)
        st.error(format_error_message(exc))
        _render_error_details(exc)
        return
    progress.progress(100)
    log_step(run_id, "ui_end", "Run completed")
    st.session_state["sim_results"] = result
    st.session_state["run_log_path"] = str(log_path)

    fallback_dismissed = False
    try:
        fallback_dismissed = bool(
            getattr(st.session_state, "dismiss_weight_engine_fallback", False)
        )
    except Exception:  # pragma: no cover - defensive
        fallback_dismissed = False

    fallback_info = getattr(result, "fallback_info", None)
    if isinstance(fallback_info, dict) and not fallback_dismissed:
        engine_name = fallback_info.get("engine", "unknown engine")
        message = (
            f"Weight engine '{engine_name}' failed; falling back to equal weights."
        )
        with st.warning(message):
            dismiss_clicked = st.button(
                "Dismiss weight engine warning",
                key="dismiss_weight_engine_warning",
            )
        if dismiss_clicked:
            st.session_state["dismiss_weight_engine_fallback"] = True
            st.rerun()
        else:
            st.session_state["dismiss_weight_engine_fallback"] = False
    elif isinstance(fallback_info, dict):
        st.session_state["dismiss_weight_engine_fallback"] = True

    st.success("Done.")
    metrics = getattr(result, "metrics", None)
    if metrics is not None:
        render_metrics = True
        if hasattr(metrics, "empty"):
            try:
                render_metrics = not metrics.empty  # type: ignore[assignment]
            except Exception:  # pragma: no cover - defensive
                render_metrics = True
        elif isinstance(metrics, (list, tuple, set, dict)):
            render_metrics = bool(metrics)
        if render_metrics:
            st.write("Summary:", metrics)


def _build_signals_config(config: Mapping[str, Any]) -> dict[str, Any]:
    base = TrendSpec()
    payload: dict[str, Any] = {
        "kind": config.get("kind", base.kind),
        "lag": int(config.get("lag", base.lag)),
        "window": int(config.get("window", base.window)),
        "vol_adjust": bool(config.get("vol_adjust", base.vol_adjust)),
        "zscore": bool(config.get("zscore", base.zscore)),
    }
    min_periods = config.get("min_periods")
    if min_periods is not None:
        try:
            payload["min_periods"] = max(0, int(min_periods))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pass
    vol_target = config.get("vol_target")
    if vol_target is not None:
        try:
            payload["vol_target"] = float(vol_target)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pass
    return payload


if __name__ == "__main__":
    main()
