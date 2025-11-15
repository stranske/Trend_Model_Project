"""Export bundle download page for the Streamlit app."""

from __future__ import annotations

import hashlib
import tempfile
import textwrap
from pathlib import Path

import streamlit as st

from trend_analysis.export.bundle import export_bundle

st.title("Export")

if "sim_results" not in st.session_state or "sim_config" not in st.session_state:
    st.error("Run a simulation first.")
    st.stop()

run = st.session_state["sim_results"]
config = st.session_state.get("sim_config", {})

# Attach config and seed to run object if available
setattr(run, "config", config)
setattr(run, "seed", st.session_state.get("seed", None))


def _generate_cache_key(run_obj, config_dict) -> str:
    """Generate a cache key for the results and config."""
    # Create a hash from the portfolio data and config to detect changes
    portfolio_hash = hashlib.sha256(
        run_obj.portfolio.to_csv().encode("utf-8")
    ).hexdigest()[:16]

    config_hash = hashlib.sha256(
        str(sorted(config_dict.items())).encode("utf-8")
    ).hexdigest()[:16]

    return f"export_bundle_{portfolio_hash}_{config_hash}"


@st.cache_data(ttl=300)  # Cache for 5 minutes
def _cached_export_bundle(cache_key: str, config_dict, _run):
    """
    Cached version of export_bundle to avoid regenerating identical bundles.

    Args:
        cache_key: A cache key for the results data
        config_dict: Configuration dictionary (must be hashable)
        _run: The actual results object (prefixed with _ to exclude from hashing)

    Returns:
        Tuple of (bundle_data_bytes, filename)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "analysis_bundle.zip"
        export_bundle(_run, zip_path)

        # Read the bundle data into memory
        with open(zip_path, "rb") as f:
            bundle_data = f.read()

        return bundle_data, zip_path.name


st.markdown(
    textwrap.dedent(
        """
        ### Export Analysis Bundle

        Create a comprehensive ZIP bundle containing all analysis results, charts, and metadata.

        **Bundle Contents:**
        - 📊 **Results**: Portfolio returns, benchmark data, and weights (CSV format)
        - 📈 **Charts**: Equity curve and drawdown visualizations (PNG format)
        - 📋 **Summary**: Analysis metrics and performance data (Excel format)
        - 🔍 **Metadata**: Configuration, versions, git hash, and reproducibility receipt
        - 📄 **README**: Package description and contents guide
        """
    )
)

# Generate cache key for current data
cache_key = _generate_cache_key(run, config)

# Create the bundle (cached if identical data exists)
try:
    with st.spinner("Creating export bundle..."):
        bundle_data, filename = _cached_export_bundle(cache_key, config, run)

    st.success(f"✅ Bundle created successfully: {filename}")

    # Show bundle information
    bundle_size_mb = len(bundle_data) / (1024 * 1024)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("Bundle Size", f"{bundle_size_mb:.2f} MB")
        st.metric("Files Included", "Multiple files")

    # Size warning if approaching limit
    if bundle_size_mb > 40:
        st.warning(
            f"⚠️ Bundle size ({bundle_size_mb:.1f} MB) is approaching the 50 MB limit."
        )
    elif bundle_size_mb > 50:
        st.error(f"❌ Bundle size ({bundle_size_mb:.1f} MB) exceeds the 50 MB limit.")

    # Provide download button
    st.download_button(
        label="📥 Download Bundle",
        data=bundle_data,
        file_name=filename,
        mime="application/zip",
        help="Click to download the complete analysis results bundle",
        type="primary",
    )

    # Bundle details
    with st.expander("📋 Bundle Details", expanded=False):
        st.markdown(
            textwrap.dedent(
                f"""
                **Bundle Information:**
                - **Size**: {bundle_size_mb:.2f} MB
                - **Format**: ZIP archive
                - **Created**: Just now

                **Contents Preview:**
                ```
                analysis_bundle.zip
                ├── results/
                │   ├── portfolio.csv      # Portfolio returns
                │   ├── benchmark.csv      # Benchmark returns (if available)
                │   └── weights.csv        # Portfolio weights (if available)
                ├── charts/
                │   ├── equity_curve.png   # Cumulative returns chart
                │   └── drawdown.png       # Drawdown analysis chart
                ├── summary.xlsx           # Summary metrics workbook
                ├── run_meta.json          # Configuration and metadata
                └── README.txt             # Bundle description
                ```

                **Reproducibility:**
                - Configuration snapshot included
                - Git commit hash recorded
                - Environment versions captured
                - Input file SHA256 hash (if available)
                """
            )
        )

except Exception as e:
    st.error(f"❌ Failed to create export bundle: {str(e)}")
    st.markdown(
        """
    **Troubleshooting:**
    - Ensure you have completed a simulation run
    - Check that results are available in session state
    - Try running the analysis again if the error persists
    """
    )
