"""Main Streamlit app for trend analysis."""

import streamlit as st
import sys
from pathlib import Path

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
        """
    Welcome to the enhanced Trend Analysis application with unified execution and progress tracking.
    
    ### Getting Started:
    1. **📤 Upload**: Load your returns data (CSV/Excel)
    2. **⚙️ Configure**: Set analysis parameters and date ranges  
    3. **🚀 Run**: Execute analysis with progress tracking
    4. **📊 Results**: View detailed analysis results
    5. **💾 Export**: Save your findings
    
    Use the sidebar navigation to move between sections.
    """
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
        if "sim_config" in st.session_state and st.session_state["sim_config"]:
            st.success("⚙️ Configuration Set")
        else:
            st.info("⚙️ Configure Analysis")

    with col3:
        if (
            "sim_results" in st.session_state
            and st.session_state["sim_results"] is not None
        ):
            st.success("📊 Analysis Complete")
        else:
            st.info("🚀 Run Analysis")

    # Quick actions
    st.markdown("---")
    st.markdown("### Quick Actions")

    action_col1, action_col2, action_col3, action_col4 = st.columns(4)

    with action_col1:
        if st.button("📤 Go to Upload", use_container_width=True):
            st.switch_page("pages/01_Upload.py")

    with action_col2:
        if st.button("⚙️ Go to Configure", use_container_width=True):
            st.switch_page("pages/02_Configure.py")

    with action_col3:
        if st.button("🚀 Go to Run", use_container_width=True):
            st.switch_page("pages/03_Run.py")

    with action_col4:
        if st.button("📊 Go to Results", use_container_width=True):
            st.switch_page("pages/04_Results.py")

    # App info
    st.markdown("---")
    st.markdown("### About This App")

    with st.expander("ℹ️ Application Features", expanded=False):
        st.markdown(
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


if __name__ == "__main__":
    main()
