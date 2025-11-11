from __future__ import annotations

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


def main() -> None:
    # Re-import streamlit within the function to ensure any test-time monkeypatches
    # (e.g. replacing ``st.button``) are respected even if the module was previously
    # imported elsewhere.
    import streamlit as st  # noqa: PLC0415 - local import for testability

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
        has_config = (
            "sim_config" in st.session_state
            and st.session_state.get("sim_config") is not None
        )
    except TypeError:
        # Fallback when session_state doesn't support membership tests
        has_returns = getattr(st.session_state, "returns_df", None) is not None
        has_config = getattr(st.session_state, "sim_config", None) is not None

    if not (has_returns and has_config):
        st.error("Upload data and set configuration first.")
        return
    if not accepted and not dry_run_clicked and not run_clicked:
        return

    try:
        df = st.session_state.get("returns_df")  # type: ignore[attr-defined]
        cfg = st.session_state.get("sim_config")  # type: ignore[attr-defined]
    except TypeError:
        # In bare-mode tests, session_state may be a Mock
        df = getattr(st.session_state, "returns_df", None)
        cfg = getattr(st.session_state, "sim_config", None)
    if df is None or cfg is None:
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
            st.error(f"Dry run unavailable: {exc}")
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
    result = run_sim(config, returns)
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
