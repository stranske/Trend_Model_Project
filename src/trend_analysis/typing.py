"""Shared typing aliases for trend_analysis multi-period outputs."""

import sys
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Mapping, MutableMapping, MutableSequence, TypeAlias, TypedDict

CovarianceDiagonal: TypeAlias = list[float]
StatsMapping: TypeAlias = Mapping[str, float] | MutableMapping[str, float]

__all__ = ["CovarianceDiagonal", "StatsMapping", "MultiPeriodPeriodResult"]


_MODULE_SELF: ModuleType = sys.modules[__name__]


def _ensure_registered() -> None:
    if sys.modules.get(__name__) is not _MODULE_SELF:
        sys.modules[__name__] = _MODULE_SELF


_ORIGINAL_SPEC = globals().get("__spec__")


class _SpecProxy:
    __slots__ = ("_spec",)

    def __init__(self, spec: ModuleSpec) -> None:
        self._spec = spec

    def __getattr__(self, attr: str) -> object:
        return getattr(self._spec, attr)

    @property
    def name(self) -> str:
        _ensure_registered()
        return self._spec.name


if _ORIGINAL_SPEC is not None:
    globals()["__spec__"] = _SpecProxy(_ORIGINAL_SPEC)

_ensure_registered()


class MultiPeriodPeriodResult(TypedDict, total=False):
    """Typed contract for multi-period analysis results."""

    period: tuple[str, str, str, str]
    out_ew_stats: StatsMapping
    out_user_stats: StatsMapping
    manager_changes: MutableSequence[dict[str, object]]
    turnover: float
    transaction_cost: float
    cov_diag: CovarianceDiagonal
    cache_stats: StatsMapping
