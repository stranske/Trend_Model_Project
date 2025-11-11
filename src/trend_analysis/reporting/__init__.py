"""Backwards-compatible reporting access for legacy imports."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import TYPE_CHECKING, Any, Mapping

__all__ = ["ReportArtifacts", "generate_unified_report"]


_MODULE_SELF: ModuleType = sys.modules[__name__]


def _ensure_registered() -> None:
    if sys.modules.get(__name__) is not _MODULE_SELF:
        sys.modules[__name__] = _MODULE_SELF


_ORIGINAL_SPEC = globals().get("__spec__")


class _SpecProxy:
    __slots__ = ("_spec",)

    def __init__(self, spec: ModuleSpec) -> None:
        self._spec = spec

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._spec, attr)

    @property
    def name(self) -> str:
        _ensure_registered()
        return self._spec.name


if _ORIGINAL_SPEC is not None:
    globals()["__spec__"] = _SpecProxy(_ORIGINAL_SPEC)

_ensure_registered()


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
