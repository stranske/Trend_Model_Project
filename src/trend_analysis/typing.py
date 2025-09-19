"""Shared typing aliases for trend_analysis multi-period outputs."""

from __future__ import annotations

from typing import Mapping, MutableMapping, MutableSequence, TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from typing import TypeAlias
else:  # pragma: no cover - runtime fallback for older interpreters.
    try:
        from typing import TypeAlias
    except ImportError:
        from typing_extensions import TypeAlias


CovarianceDiagonal: TypeAlias = list[float]
StatsMapping: TypeAlias = Mapping[str, float] | MutableMapping[str, float]

__all__ = [
    "CovarianceDiagonal",
    "StatsMapping",
    "MultiPeriodPeriodResult",
]


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
