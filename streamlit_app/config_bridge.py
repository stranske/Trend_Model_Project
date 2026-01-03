"""Bridge helpers aligning the Streamlit app with CLI configuration checks."""

from __future__ import annotations

import warnings

from trend_analysis.config.bridge import build_config_payload, validate_payload

warnings.warn(
    "streamlit_app.config_bridge is deprecated; import from "
    "trend_analysis.config.bridge instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["build_config_payload", "validate_payload"]
