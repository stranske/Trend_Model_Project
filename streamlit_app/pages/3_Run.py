import uuid
from typing import Dict

import pandas as pd

from streamlit_app.components.disclaimer import show_disclaimer
from streamlit_app.components.guardrails import (
    estimate_resource_usage,
    prepare_dry_run_plan,
)
from trend_analysis.api import run_simulation
from trend_analysis.logging import get_default_log_path, init_run_logger, log_step


class StreamlitConfig:
    def __init__(self, **data: object) -> None:
        self.__dict__.update(data)

    def model_dump(self) -> Dict[str, object]:
        return dict(self.__dict__)


Config = StreamlitConfig


def _make_config(config_data: Dict[str, object]) -> StreamlitConfig:
    return Config(**config_data)


def main():
    # Re-import streamlit within the function to ensure any test-time monkeypatches
    # (e.g. replacing ``st.button``) are respected even if the module was previously
    # imported elsewhere.
    import streamlit as st  # noqa: WPS433 - local import for testability

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
        except Exception:
            pass
        try:
            return d[key]  # type: ignore[index]
        except Exception:
            return default

    raw_lookback = cfg_get(cfg, "lookback_months", 0)
    try:
        lookback = int(raw_lookback) if raw_lookback is not None else 0
    except Exception:
        lookback = 0

    start = cfg_get(cfg, "start")
    end = cfg_get(cfg, "end")
    # Coerce to pandas.Timestamp when possible (handles python date, str, etc.)
    try:
        if start is not None and not isinstance(start, pd.Timestamp):
            start = pd.to_datetime(start)
        if end is not None and not isinstance(end, pd.Timestamp):
            end = pd.to_datetime(end)
    except Exception:
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
        "metrics": {},
        "export": {},
        "benchmarks": {},
        "run": {},
    }

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
            result = run_simulation(dry_config, dry_returns)
        st.session_state["dry_run_results"] = result
        st.session_state["dry_run_summary"] = plan.summary()
        st.success(
            f"Dry run completed on {plan.frame.shape[0]} rows × {plan.frame.shape[1]} columns."
        )
        st.json(plan.summary())
        return

    if not run_clicked:
        return

    # Assign / persist run_id in session
    progress = st.progress(0, "Running simulation...")
    run_id = st.session_state.get("run_id") or uuid.uuid4().hex[:12]
    st.session_state["run_id"] = run_id
    config_data["run_id"] = run_id
    config = _make_config(config_data)
    log_path = get_default_log_path(run_id)
    init_run_logger(run_id, log_path)
    log_step(run_id, "ui_start", "Streamlit run initiated")
    config.run_id = run_id  # type: ignore[attr-defined]
    log_step(run_id, "api_call", "Dispatching run_simulation")
    result = run_simulation(config, returns)
    log_step(
        run_id,
        "api_return",
        "run_simulation returned",
        metrics_rows=len(result.metrics),
    )
    progress.progress(100)
    st.session_state["sim_results"] = result
    st.session_state["run_log_path"] = str(log_path)
    log_step(run_id, "ui_end", "Run complete")
    # Show fallback banner if a weight engine failed
    try:
        fb = getattr(result, "fallback_info", None)
    except Exception:  # pragma: no cover - defensive
        fb = None
    if fb and not st.session_state.get("dismiss_weight_engine_fallback"):
        with st.warning(
            "⚠️ Weight engine '%s' failed (%s). Using equal weights."
            % (fb.get("engine"), fb.get("error_type")),
        ):
            if st.button(
                "Dismiss",
                key="btn_dismiss_weight_engine_fallback",
                help="Hide this warning for the current session.",
            ):
                st.session_state["dismiss_weight_engine_fallback"] = True
                st.rerun()
    st.success("Done.")
    st.write("Summary:", result.metrics)


if __name__ == "__main__":
    main()
