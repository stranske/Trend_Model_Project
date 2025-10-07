"""Backwards-compatible reporting access for legacy imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:  # pragma: no cover - typing only
    from trend.reporting import ReportArtifacts, generate_unified_report
else:
    try:
        from trend.reporting import ReportArtifacts, generate_unified_report
    except ImportError:

        @dataclass(slots=True)
        class ReportArtifacts:
            html: str
            pdf_bytes: bytes | None
            context: Mapping[str, Any]

        def generate_unified_report(
            result: Any,
            config: Any,
            *,
            run_id: str | None = None,
            include_pdf: bool = False,
            spec: Any | None = None,
        ) -> ReportArtifacts:
            raise ImportError(
                "trend.reporting.generate_unified_report not found. Please ensure trend.reporting is available."
            )


__all__ = ["ReportArtifacts", "generate_unified_report"]
