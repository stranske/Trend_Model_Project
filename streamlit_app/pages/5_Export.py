import os
import hashlib
import streamlit as st
from trend_portfolio_app.io_utils import export_bundle

st.title("Export")

if "sim_results" not in st.session_state or "sim_config" not in st.session_state:
    st.error("Run a simulation first.")
    st.stop()

res = st.session_state["sim_results"]
cfg = st.session_state["sim_config"]


def _generate_cache_key(results, config_dict) -> str:
    """Generate a cache key for the results and config."""
    # Create a hash from the portfolio data and config to detect changes
    portfolio_hash = hashlib.sha256(
        results.portfolio.to_csv().encode("utf-8")
    ).hexdigest()[:16]

    config_hash = hashlib.sha256(
        str(sorted(config_dict.items())).encode("utf-8")
    ).hexdigest()[:16]

    return f"export_bundle_{portfolio_hash}_{config_hash}"


@st.cache_data(ttl=300)  # Cache for 5 minutes
def _cached_export_bundle(results_cache_key: str, config_dict, _results):
    """
    Cached version of export_bundle to avoid regenerating identical bundles.

    Args:
        results_cache_key: A cache key for the results data
        config_dict: Configuration dictionary (must be hashable)
        _results: The actual results object (prefixed with _ to exclude from hashing)

    Returns:
        Tuple of (bundle_data_bytes, filename)
    """
    path = export_bundle(_results, config_dict)

    # Read the bundle data into memory
    with open(path, "rb") as f:
        bundle_data = f.read()

    filename = os.path.basename(path)

    return bundle_data, filename


# Generate cache key for current data
cache_key = _generate_cache_key(res, cfg)

# Create the bundle (cached if identical data exists)
try:
    bundle_data, filename = _cached_export_bundle(
        cache_key,
        cfg,  # Config dict should be serializable/hashable
        res,  # Results object excluded from cache key with _results
    )

    st.success(f"Bundle created: {filename}")

    # Provide download button with the cached data
    st.download_button(
        label="Download bundle",
        data=bundle_data,
        file_name=filename,
        mime="application/zip",
        help="Click to download the analysis results bundle",
    )

    # Show bundle info
    bundle_size_mb = len(bundle_data) / (1024 * 1024)
    st.info(f"Bundle size: {bundle_size_mb:.2f} MB")

except Exception as e:
    st.error(f"Failed to create export bundle: {e}")

    # The temporary files will be cleaned up automatically on process exit
