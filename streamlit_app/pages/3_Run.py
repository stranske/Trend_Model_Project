import pandas as pd

from streamlit_app.components.disclaimer import show_disclaimer
from streamlit_app.components.fallback_warning import format_weight_engine_warning
from trend_analysis.api import run_simulation
from trend_analysis.config import Config


def main():
    # Re-import streamlit within the function to ensure any test-time monkeypatches
    # (e.g. replacing ``st.button``) are respected even if the module was previously
    # imported elsewhere.
    import streamlit as st  # noqa: WPS433 - local import for testability

    st.title("Run")
    # Always render disclaimer + button first so tests can capture state
    accepted = show_disclaimer()
    run_clicked = st.button("Run simulation", disabled=not accepted)
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
    # If disclaimer isn't accepted or button not pressed, stop early
    if not accepted or not run_clicked:
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

    returns = df.reset_index().rename(columns={df.index.name or "index": "Date"})

    progress = st.progress(0, "Running simulation...")

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

    config = Config(
        version="1",
        data={},
        preprocessing={},
        vol_adjust={"target_vol": cfg_get(cfg, "risk_target", 1.0)},
        sample_split={
            "in_start": (start - pd.DateOffset(months=lookback)).strftime("%Y-%m"),
            "in_end": (start - pd.DateOffset(months=1)).strftime("%Y-%m"),
            "out_start": start.strftime("%Y-%m"),
            "out_end": end.strftime("%Y-%m"),
        },
        portfolio=portfolio_cfg,
        metrics={},
        export={},
        run={},
    )

    result = run_simulation(config, returns)
    progress.progress(100)
    st.session_state["sim_results"] = result
    # Show fallback banner if a weight engine failed
    try:
        fb = getattr(result, "fallback_info", None)
    except Exception:  # pragma: no cover - defensive
        fb = None
    if fb and not st.session_state.get("dismiss_weight_engine_fallback"):
        message = format_weight_engine_warning(
            fb,
            suffix="Using equal weights.",
        )
        with st.warning(message):
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
