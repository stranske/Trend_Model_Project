"""Main Streamlit app for trend analysis."""

import sys
import textwrap
from pathlib import Path

import streamlit as st

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

st.set_page_config(
    page_title="Trend Analysis App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.title("📈 Trend Analysis Application")

    st.markdown(
        textwrap.dedent(
            """
            Welcome to the enhanced Trend Analysis application with unified execution and progress tracking.

            ### Getting Started:
            1. **📥 Data**: Load your returns data (CSV/Excel or sample)
            2. **⚙️ Model**: Choose presets and portfolio settings
            3. **📊 Results**: View performance charts and diagnostics
            4. **💾 Export**: Save your findings

            Use the sidebar navigation to move between sections.
            """
        )
    )

    # Status indicators
    col1, col2, col3 = st.columns(3)

    with col1:
        if (
            "returns_df" in st.session_state
            and st.session_state["returns_df"] is not None
        ):
            st.success("✅ Data Uploaded")
        else:
            st.info("📤 Upload Data")

    with col2:
        if "model_state" in st.session_state and st.session_state["model_state"]:
            st.success("⚙️ Model Configured")
        else:
            st.info("⚙️ Configure Model")

    with col3:
        if (
            "analysis_result" in st.session_state
            and st.session_state["analysis_result"] is not None
        ):
            st.success("📊 Analysis Complete")
        else:
            st.info("🚀 Run Analysis")

    # Quick actions
    st.markdown("---")
    st.markdown("### Quick Actions")

    action_col1, action_col2, action_col3, action_col4 = st.columns(4)

    with action_col1:
        if st.button("📥 Go to Data", use_container_width=True):
            st.switch_page("pages/1_Data.py")

    with action_col2:
        if st.button("⚙️ Go to Model", use_container_width=True):
            st.switch_page("pages/2_Model.py")

    with action_col3:
        if st.button("📊 Go to Results", use_container_width=True):
            st.switch_page("pages/4_Results.py")

    with action_col4:
        if st.button("💾 Go to Export", use_container_width=True):
            st.switch_page("pages/5_Export.py")

    # App info
    st.markdown("---")
    st.markdown("### About This App")

    with st.expander("ℹ️ Application Features", expanded=False):
        st.markdown(
            textwrap.dedent(
                """
                **Enhanced Run Page Features:**
                - 📊 **Progress Tracking**: Real-time progress bar with 5-phase analysis
                - 📋 **Live Logging**: View analysis logs as they happen
                - ❌ **Smart Error Handling**: Human-readable error messages with technical details on demand
                - ✅ **Validation**: Comprehensive input validation before execution
                - 🔧 **Session Management**: Persistent configuration and results

                **Key Improvements:**
                - No more raw Python tracebacks for users
                - Clear progress indication throughout analysis
                - Expandable technical details for debugging
                - Robust error recovery and user guidance
                """
            )
        )


if __name__ == "__main__":
    main()
