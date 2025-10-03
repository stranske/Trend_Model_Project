"""Backwards-compatible reporting access for legacy imports."""

from __future__ import annotations

try:
    from trend.reporting import ReportArtifacts, generate_unified_report
except ImportError:
    class ReportArtifacts:
        def __init__(self, *args, **kwargs):
            raise ImportError("trend.reporting.ReportArtifacts not found. Please ensure trend.reporting is available.")
    def generate_unified_report(*args, **kwargs):
        raise ImportError("trend.reporting.generate_unified_report not found. Please ensure trend.reporting is available.")

__all__ = ["ReportArtifacts", "generate_unified_report"]
