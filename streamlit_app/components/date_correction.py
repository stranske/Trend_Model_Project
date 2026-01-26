"""Streamlit wrapper for shared date-correction helpers."""

from __future__ import annotations

import warnings

from trend_analysis.io.date_correction import (  # noqa: F401
    DateCorrection,
    DateCorrectionResult,
    analyze_date_column,
    apply_date_corrections,
    format_corrections_for_display,
)

warnings.warn(
    "streamlit_app.components.date_correction is deprecated; "
    "import from trend_analysis.io.date_correction instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "DateCorrection",
    "DateCorrectionResult",
    "analyze_date_column",
    "apply_date_corrections",
    "format_corrections_for_display",
]
