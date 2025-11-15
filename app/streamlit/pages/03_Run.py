"""Run page for the legacy Streamlit Trend Analysis application."""

from __future__ import annotations

import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from typing import Any, Optional, cast

import pandas as pd
import streamlit as st

_SRC_PATH = Path(__file__).resolve().parents[3] / "src"
if str(_SRC_PATH) not in sys.path:  # pragma: no cover - import side effect
    sys.path.append(str(_SRC_PATH))


RunSimulationFn = Callable[[Any, pd.DataFrame], object]


def _api() -> tuple[type[Any], RunSimulationFn]:
    from trend_analysis.api import RunResult, run_simulation

    return RunResult, run_simulation


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamlitLogHandler(logging.Handler):
    """Custom log handler that records log lines for display in Streamlit."""

    def __init__(self) -> None:
        super().__init__()
        self.log_messages: list[dict[str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
        log_message = self.format(record)
        self.log_messages.append(
            {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "level": record.levelname,
                "message": log_message,
            }
        )

    def get_logs(self) -> list[dict[str, str]]:
        return self.log_messages

    def clear_logs(self) -> None:  # pragma: no cover - unused helper
        self.log_messages = []


def format_error_message(error: Exception) -> str:
    """Convert an exception into a human-readable error message."""

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

    if "Date" in error_msg:
        return (
            "Data validation error: Your dataset must include a Date column with "
            "properly formatted dates."
        )
    if "sample_split" in error_msg:
        return (
            "Configuration error: Invalid date ranges specified. Please check "
            "your in-sample and out-of-sample periods."
        )
    if "returns" in error_msg.lower():
        return (
            "Data error: Invalid returns data format. Please ensure your data "
            "contains numeric return values."
        )
    if "config" in error_msg.lower():
        return (
            "Configuration error: Invalid configuration settings. Please review "
            "your analysis parameters."
        )

    if error_type in error_mappings:
        return f"{error_mappings[error_type]}: {error_msg}"

    return f"Analysis error ({error_type}): {error_msg}"


def create_config_from_session_state() -> Optional[object]:
    """Create a Config object from the current Streamlit session state."""

    try:
        sim_config = st.session_state.get("sim_config", {})
        if not sim_config:
            st.error(
                "No configuration found. Please set up your analysis configuration first."
            )
            return None

        lookback = sim_config.get("lookback_months", 0)
        start = sim_config.get("start")
        end = sim_config.get("end")

        if not start or not end:
            st.error("Start and end dates are required in configuration.")
            return None

        from trend_analysis.config import Config as _Config  # local import

        ConfigModel = cast(Any, _Config)
        config = cast(
            object,
            ConfigModel(
                version="1",
                data={},
                preprocessing=sim_config.get("preprocessing", {}),
                vol_adjust={
                    "target_vol": sim_config.get("risk_target", 1.0),
                    "window": sim_config.get("vol_window", {}),
                },
                sample_split={
                    "in_start": (start - pd.DateOffset(months=lookback)).strftime(
                        "%Y-%m"
                    ),
                    "in_end": (start - pd.DateOffset(months=1)).strftime("%Y-%m"),
                    "out_start": start.strftime("%Y-%m"),
                    "out_end": end.strftime("%Y-%m"),
                },
                portfolio=sim_config.get("portfolio", {}),
                benchmarks=sim_config.get("benchmarks", {}),
                metrics=sim_config.get("metrics", {}),
                export=sim_config.get("export", {}),
                run=sim_config.get("run", {}),
            ),
        )
        return config

    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Failed to create configuration: {format_error_message(exc)}")
        return None


def prepare_returns_data() -> Optional[pd.DataFrame]:
    """Prepare the returns data stored in session state."""

    try:
        df = st.session_state.get("returns_df")
        if df is None:
            st.error("No data found. Please upload your returns data first.")
            return None

        if "Date" not in df.columns:
            df = df.reset_index()
            if df.index.name:
                df = df.rename(columns={df.index.name: "Date"})
            elif "index" in df.columns:
                df = df.rename(columns={"index": "Date"})
            else:
                date_cols = [col for col in df.columns if "date" in col.lower()]
                if date_cols:
                    df = df.rename(columns={date_cols[0]: "Date"})
                else:
                    st.error(
                        "Could not find a Date column in your data. Please ensure your dataset includes dates."
                    )
                    return None

        return df

    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Failed to prepare data: {format_error_message(exc)}")
        return None


def run_analysis_with_progress() -> Optional[object]:
    """Run the analysis while streaming progress and log output to Streamlit."""

    log_handler = StreamlitLogHandler()
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(
        logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    )

    trend_logger = logging.getLogger("trend_analysis")
    trend_logger.addHandler(log_handler)
    trend_logger.setLevel(logging.INFO)

    progress_bar = st.progress(0, "Initializing analysis...")
    status_text = st.empty()
    log_expander = st.expander("📋 Analysis Log", expanded=True)
    log_display = log_expander.empty()

    try:
        status_text.text("🔧 Preparing data and configuration...")
        progress_bar.progress(10, "Preparing data and configuration...")

        config = create_config_from_session_state()
        if config is None:
            return None

        returns_df = prepare_returns_data()
        if returns_df is None:
            return None

        progress_bar.progress(25, "Data preparation complete...")
        status_text.text("✅ Validating inputs...")
        progress_bar.progress(40, "Validating inputs...")

        if len(returns_df) == 0:
            raise ValueError("Returns data is empty")
        if "Date" not in returns_df.columns:
            raise ValueError("Date column is missing from returns data")

        progress_bar.progress(50, "Input validation complete...")
        status_text.text("🚀 Running trend analysis...")
        progress_bar.progress(60, "Running trend analysis...")

        RunResult, run_simulation = _api()
        result = run_simulation(config, returns_df)

        progress_bar.progress(90, "Analysis complete, finalizing results...")
        status_text.text("✨ Finalizing results...")
        progress_bar.progress(100, "Analysis completed successfully!")

        logs = log_handler.get_logs()
        if logs:
            log_text = "\n".join(
                f"[{log['timestamp']}] {log['level']}: {log['message']}"
                for log in logs[-10:]
            )
            log_display.code(log_text)

        status_text.text("🎉 Analysis completed successfully!")
        return result

    except Exception as exc:
        progress_bar.progress(0, "Analysis failed")
        status_text.text("❌ Analysis failed")

        logs = log_handler.get_logs()
        if logs:
            log_text = "\n".join(
                f"[{log['timestamp']}] {log['level']}: {log['message']}" for log in logs
            )
            log_display.code(log_text)

        error_msg = format_error_message(exc)
        st.error(f"**Analysis Failed**: {error_msg}")
        details = (
            f"Exception Type: {type(exc).__name__}\n\n"
            f"Exception Message:\n{str(exc)}\n\n"
            f"Full Traceback:\n{traceback.format_exc()}"
        )
        expander = st.expander("🔍 Show Technical Details", expanded=False)
        try:
            expander.code(details)
        except Exception:  # pragma: no cover - fallback for mocks
            st.code(details)
        return None

    finally:
        trend_logger.removeHandler(log_handler)


def main() -> None:
    """Render the legacy Streamlit Run page."""

    st.title("🚀 Run Analysis")

    if "returns_df" not in st.session_state or st.session_state["returns_df"] is None:
        st.warning("⚠️ **Upload Required**: Please upload your returns data first.")
        st.info("👈 Go to the **Upload** page to load your data.")
        return

    if "sim_config" not in st.session_state or not st.session_state["sim_config"]:
        st.warning(
            "⚠️ **Configuration Required**: Please configure your analysis parameters first."
        )
        st.info("👈 Go to the **Configure** page to set up your analysis.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.info("📊 **Data Status**: Ready")
        df_info = st.session_state["returns_df"]
        st.write(f"- **Rows**: {len(df_info):,}")
        st.write(f"- **Columns**: {len(df_info.columns)}")

    with col2:
        st.info("⚙️ **Configuration Status**: Ready")
        config_info = st.session_state["sim_config"]
        st.write(f"- **Start Date**: {config_info.get('start', 'Not set')}")
        st.write(f"- **End Date**: {config_info.get('end', 'Not set')}")

    st.markdown("---")
    run_col, clear_col = st.columns([3, 1])

    with run_col:
        if st.button("🚀 **Run Analysis**", type="primary", use_container_width=True):
            with st.spinner("Running analysis..."):
                result = run_analysis_with_progress()
                if result is not None:
                    st.session_state["sim_results"] = result
                    st.session_state["last_run_timestamp"] = datetime.now()
                    st.success("✅ **Analysis completed successfully!**")

                    with st.expander("📈 Quick Results Summary", expanded=True):
                        if hasattr(result, "metrics") and not result.metrics.empty:
                            st.dataframe(result.metrics.head())
                        else:
                            st.info("No metrics to display.")

                    st.info(
                        "👉 Go to the **Results** page to explore detailed findings."
                    )

    with clear_col:
        if st.button("🗑️ Clear", help="Clear previous results"):
            st.session_state.pop("sim_results", None)
            st.session_state.pop("last_run_timestamp", None)
            st.success("Results cleared!")
            st.rerun()

    if st.session_state.get("sim_results") is not None:
        st.markdown("---")
        st.markdown("### 📋 Previous Results")
        last_run = st.session_state.get("last_run_timestamp")
        if last_run:
            st.caption(f"Last run: {last_run.strftime('%Y-%m-%d %H:%M:%S')}")

        result = st.session_state["sim_results"]
        if hasattr(result, "metrics") and not result.metrics.empty:
            st.dataframe(result.metrics)
        else:
            st.info("No previous results available.")


if __name__ == "__main__":  # pragma: no cover - Streamlit entry-point
    main()
