"""Run page for Streamlit trend analysis app with unified execution and progress."""

import logging
import json
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from trend_analysis.api import RunResult, run_simulation

# Configure logging for the app (streamlit already initialises logging, but ensure level)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_LOG_DIR = Path.cwd() / "run_logs"


def _read_log_entries(path: Path) -> list[dict[str, object]]:
    """Load JSONL log entries from ``path``."""

    if not path.exists():
        return []
    entries: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    entries.append(parsed)  # type: ignore[arg-type]
    except OSError:
        return entries
    return entries


def _format_log_block(entries: list[dict[str, object]], limit: int = 30) -> str:
    if limit and len(entries) > limit:
        entries = entries[-limit:]
    lines: list[str] = []
    for item in entries:
        timestamp = item.get("timestamp", "?")
        level = item.get("level", "INFO")
        step = item.get("step", "-")
        message = item.get("message", "")
        lines.append(f"[{timestamp}] [{level}] ({step}) {message}")
    return "\n".join(lines)


def _summarise_errors(entries: list[dict[str, object]]) -> str | None:
    errors = [item for item in entries if item.get("level") in {"ERROR", "CRITICAL"}]
    if not errors:
        return None
    seen: set[str] = set()
    unique: list[str] = []
    for item in errors:
        message = item.get("message", "")
        if message not in seen:
            seen.add(message)
            unique.append(message)
    if not unique:
        return None
    bullets = "\n".join(f"- {msg}" for msg in unique[-5:])
    return f"### 🚨 Errors detected\n{bullets}"


def format_error_message(error: Exception) -> str:
    """Convert an exception into a human-readable error message."""
    error_type = type(error).__name__
    error_msg = str(error)

    # Common error patterns and user-friendly messages
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

    # Try to provide more specific guidance
    if "Date" in error_msg:
        # Keep wording aligned with tests (no quotes around Date)
        return "Data validation error: Your dataset must include a Date column with properly formatted dates."
    elif "sample_split" in error_msg:
        return "Configuration error: Invalid date ranges specified. Please check your in-sample and out-of-sample periods."
    elif "returns" in error_msg.lower():
        return "Data error: Invalid returns data format. Please ensure your data contains numeric return values."
    elif "config" in error_msg.lower():
        return "Configuration error: Invalid configuration settings. Please review your analysis parameters."

    # Use generic mapping if available
    if error_type in error_mappings:
        return f"{error_mappings[error_type]}: {error_msg}"

    # Fallback to a generic but helpful message
    return f"Analysis error ({error_type}): {error_msg}"


def create_config_from_session_state() -> Optional[object]:
    """Create a Config object from session state data."""
    try:
        # Get configuration from session state
        sim_config = st.session_state.get("sim_config", {})

        if not sim_config:
            st.error(
                "No configuration found. Please set up your analysis configuration first."
            )
            return None

        # Extract required parameters with defaults
        lookback = sim_config.get("lookback_months", 0)
        start = sim_config.get("start")
        end = sim_config.get("end")

        if not start or not end:
            st.error("Start and end dates are required in configuration.")
            return None

        # Import Config at runtime to avoid stale class identity across reloads
        from trend_analysis.config import Config as _Config  # local import

        # Create Config object
        config = _Config(
            version="1",
            data={},
            preprocessing=sim_config.get("preprocessing", {}),
            vol_adjust={
                "target_vol": sim_config.get("risk_target", 1.0),
                "window": sim_config.get("vol_window", {}),
            },
            sample_split={
                "in_start": (start - pd.DateOffset(months=lookback)).strftime("%Y-%m"),
                "in_end": (start - pd.DateOffset(months=1)).strftime("%Y-%m"),
                "out_start": start.strftime("%Y-%m"),
                "out_end": end.strftime("%Y-%m"),
            },
            portfolio=sim_config.get("portfolio", {}),
            benchmarks=sim_config.get("benchmarks", {}),
            metrics=sim_config.get("metrics", {}),
            export=sim_config.get("export", {}),
            run=sim_config.get("run", {}),
        )
        return config

    except Exception as e:
        st.error(f"Failed to create configuration: {format_error_message(e)}")
        return None


def prepare_returns_data() -> Optional[pd.DataFrame]:
    """Prepare returns data from session state."""
    try:
        df = st.session_state.get("returns_df")

        if df is None:
            st.error("No data found. Please upload your returns data first.")
            return None

        # Ensure 'Date' column exists for the pipeline
        if "Date" not in df.columns:
            df = df.reset_index()
            if df.index.name:
                df = df.rename(columns={df.index.name: "Date"})
            elif "index" in df.columns:
                df = df.rename(columns={"index": "Date"})
            else:
                # Try to find a date-like column
                date_cols = [col for col in df.columns if "date" in col.lower()]
                if date_cols:
                    df = df.rename(columns={date_cols[0]: "Date"})
                else:
                    st.error(
                        "Could not find a Date column in your data. Please ensure your dataset includes dates."
                    )
                    return None

        return df

    except Exception as e:
        st.error(f"Failed to prepare data: {format_error_message(e)}")
        return None


def run_analysis_with_progress() -> Optional[RunResult]:
    """Run the analysis with progress reporting, structured logs and error summary."""

    progress_bar = st.progress(0, "Initializing analysis...")
    status_text = st.empty()
    error_summary = st.empty()
    log_expander = st.expander("📋 Run Log", expanded=True)
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

        progress_bar.progress(55, "Input validation complete...")

        run_id = uuid.uuid4().hex
        log_path = _LOG_DIR / f"{run_id}.jsonl"
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        try:
            if log_path.exists():
                log_path.unlink()
        except OSError:
            pass

        status_text.text("🚀 Running trend analysis...")
        progress_bar.progress(65, "Running trend analysis...")

        state: dict[str, object | None] = {"result": None, "error": None, "traceback": None}

        def _worker() -> None:
            try:
                state["result"] = run_simulation(
                    config,
                    returns_df,
                    run_id=run_id,
                    log_dir=_LOG_DIR,
                )
            except Exception as exc:  # pragma: no cover - surfaced in UI
                state["error"] = exc
                state["traceback"] = traceback.format_exc()

        worker = threading.Thread(target=_worker, name=f"run-sim-{run_id[:8]}", daemon=True)
        worker.start()

        while worker.is_alive():
            entries = _read_log_entries(log_path)
            summary = _summarise_errors(entries)
            if summary:
                error_summary.markdown(summary)
            else:
                error_summary.markdown("✅ No errors logged so far.")
            log_block = _format_log_block(entries)
            log_display.code(log_block or "(log pending...)")
            progress_bar.progress(75, "Running trend analysis...")
            time.sleep(0.25)

        worker.join()

        entries = _read_log_entries(log_path)
        summary = _summarise_errors(entries)
        if summary:
            error_summary.markdown(summary)
        else:
            error_summary.markdown("✅ No errors logged.")
        log_display.code(_format_log_block(entries) or "(no log entries)")

        if state["error"] is not None:
            progress_bar.progress(0, "Analysis failed")
            status_text.text("❌ Analysis failed")
            error_msg = format_error_message(state["error"])  # type: ignore[arg-type]
            st.error(f"**Analysis Failed**: {error_msg}")
            details = (
                f"Exception Type: {type(state['error']).__name__}\n\n"  # type: ignore[index]
                f"Exception Message:\n{state['error']}\n\n"  # type: ignore[index]
                f"Full Traceback:\n{state['traceback'] or ''}"
            )
            expander = st.expander("🔍 Show Technical Details", expanded=False)
            try:
                expander.code(details)
            except Exception:
                st.code(details)
            return None

        result = state.get("result")
        if not isinstance(result, RunResult):
            st.error("Unexpected result type from run_simulation.")
            return None

        progress_bar.progress(100, "Analysis completed successfully!")
        status_text.text("🎉 Analysis completed successfully!")

        return result

    except Exception as exc:
        progress_bar.progress(0, "Analysis failed")
        status_text.text("❌ Analysis failed")
        st.error(f"**Analysis Failed**: {format_error_message(exc)}")
        details = (
            f"Exception Type: {type(exc).__name__}\n\n"
            f"Exception Message:\n{exc}\n\n"
            f"Full Traceback:\n{traceback.format_exc()}"
        )
        expander = st.expander("🔍 Show Technical Details", expanded=False)
        try:
            expander.code(details)
        except Exception:
            st.code(details)
        return None


def main():
    """Main function for the Run page."""
    st.title("🚀 Run Analysis")

    # Check prerequisites
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

    # Display current setup
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

    # Main run button
    st.markdown("---")

    run_col, clear_col = st.columns([3, 1])

    with run_col:
        if st.button("🚀 **Run Analysis**", type="primary", use_container_width=True):
            with st.spinner("Running analysis..."):
                result = run_analysis_with_progress()

                if result is not None:
                    # Store results in session state
                    st.session_state["sim_results"] = result
                    st.session_state["last_run_timestamp"] = datetime.now()

                    # Display summary
                    st.success("✅ **Analysis completed successfully!**")

                    log_meta = getattr(result, "log", None)
                    if log_meta and getattr(log_meta, "path", None):
                        try:
                            log_bytes = Path(log_meta.path).read_bytes()
                        except Exception:
                            st.warning("Run log is unavailable for download.")
                        else:
                            st.download_button(
                                "📥 Download run log (JSONL)",
                                data=log_bytes,
                                file_name=f"{getattr(log_meta, 'run_id', 'run')}.jsonl",
                                mime="application/json",
                                use_container_width=True,
                            )

                    with st.expander("📈 Quick Results Summary", expanded=True):
                        if not result.metrics.empty:
                            st.dataframe(result.metrics.head())
                        else:
                            st.info("No metrics to display.")

                    st.info(
                        "👉 Go to the **Results** page to explore detailed findings."
                    )

    with clear_col:
        if st.button("🗑️ Clear", help="Clear previous results"):
            if "sim_results" in st.session_state:
                del st.session_state["sim_results"]
            if "last_run_timestamp" in st.session_state:
                del st.session_state["last_run_timestamp"]
            st.success("Results cleared!")
            st.rerun()

    # Show previous results if available
    if (
        "sim_results" in st.session_state
        and st.session_state["sim_results"] is not None
    ):
        st.markdown("---")
        st.markdown("### 📋 Previous Results")

        last_run = st.session_state.get("last_run_timestamp")
        if last_run:
            st.caption(f"Last run: {last_run.strftime('%Y-%m-%d %H:%M:%S')}")

        result = st.session_state["sim_results"]

        if not result.metrics.empty:
            st.dataframe(result.metrics)
        else:
            st.info("No previous results available.")


if __name__ == "__main__":
    main()
