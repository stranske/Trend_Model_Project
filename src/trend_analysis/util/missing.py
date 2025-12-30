"""Utilities for enforcing missing-data policies on return frames."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

__all__ = ["MissingPolicyResult", "apply_missing_policy"]


@dataclass(frozen=True, slots=True)
class MissingPolicyResult(Mapping[str, Any]):
    """Structured summary returned by :func:`apply_missing_policy`."""

    policy: dict[str, str]
    default_policy: str
    limit: dict[str, int | None]
    default_limit: int | None
    filled: dict[str, int]
    dropped_assets: tuple[str, ...]
    summary: str
    _mapping: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        policy = dict(self.policy)
        limit = dict(self.limit)
        filled = {k: int(v) for k, v in self.filled.items()}
        dropped = tuple(self.dropped_assets)
        total_filled = sum(filled.values())
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "limit", limit)
        object.__setattr__(self, "filled", filled)
        object.__setattr__(self, "dropped_assets", dropped)
        mapping: dict[str, Any] = {
            "policy": policy,
            "policy_map": policy,
            "default_policy": self.default_policy,
            "limit": limit,
            "limit_map": limit,
            "default_limit": self.default_limit,
            "filled": filled,
            "dropped": list(dropped),
            "dropped_assets": list(dropped),
            "summary": self.summary,
            "total_filled": total_filled,
        }
        object.__setattr__(self, "_mapping", mapping)

    def __getitem__(self, key: str) -> Any:  # pragma: no cover - delegate
        return self._mapping[key]

    def __iter__(self) -> Iterator[str]:  # pragma: no cover - delegate
        return iter(self._mapping)

    def __len__(self) -> int:  # pragma: no cover - delegate
        return len(self._mapping)

    def get(self, key: str, default: Any = None) -> Any:
        return self._mapping.get(key, default)

    @property
    def filled_cells(self) -> tuple[tuple[str, int], ...]:
        return tuple((asset, count) for asset, count in sorted(self.filled.items()) if count > 0)

    @property
    def total_filled(self) -> int:
        return sum(self.filled.values())


def _coerce_policy(policy: str | None) -> str:
    value = (policy or "drop").strip().lower()
    if value in {"both", "bfill", "backfill"}:
        value = "ffill"
    if value in {"zeros", "zero_fill", "fillzero"}:
        value = "zero"
    if value not in {"drop", "ffill", "zero"}:
        raise ValueError(f"Unsupported missing-data policy: {policy!r}")
    return value


def _coerce_limit(limit: Any) -> int | None:
    if limit is None:
        return None
    try:
        value = int(limit)
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        raise ValueError(f"Invalid forward-fill limit: {limit!r}") from None
    if value < 0:
        raise ValueError("Forward-fill limit must be non-negative")
    return value


def _resolve_mapping(
    spec: str | Mapping[str, str] | None, default: str
) -> tuple[str, dict[str, str]]:
    if spec is None or isinstance(spec, str):
        return _coerce_policy(spec or default), {}
    overrides = {k: _coerce_policy(v) for k, v in spec.items() if k != "default"}
    default_policy = _coerce_policy(spec.get("default", default))
    return default_policy, overrides


def _resolve_limits(
    limit: int | Mapping[str, int | None] | None,
) -> tuple[int | None, dict[str, int | None]]:
    if isinstance(limit, Mapping):
        resolved = {k: _coerce_limit(v) for k, v in limit.items() if k != "default"}
        default_limit = _coerce_limit(limit.get("default"))
        return default_limit, resolved
    return _coerce_limit(limit), {}


def _policy_display(
    default_policy: str,
    overrides: Mapping[str, str],
    default_limit: int | None,
    limit_overrides: Mapping[str, int | None],
) -> str:
    limit_part = f"limit={default_limit}" if default_limit is not None else "unbounded"
    text = [f"default={default_policy}({limit_part})"]
    if overrides:
        over = ", ".join(
            f"{asset}={policy}"
            + (
                f"(limit={limit_overrides.get(asset)})"
                if policy == "ffill" and asset in limit_overrides
                else ""
            )
            for asset, policy in sorted(overrides.items())
        )
        text.append(f"overrides: {over}")
    return "; ".join(text)


def apply_missing_policy(
    df: pd.DataFrame,
    policy: str | Mapping[str, str] | None,
    limit: int | Mapping[str, int | None] | None = None,
    *,
    columns: Iterable[str] | None = None,
    enforce_completeness: bool = True,
) -> tuple[pd.DataFrame, MissingPolicyResult]:
    """Apply a missing-data policy column by column.

    Parameters
    ----------
    df:
        DataFrame of asset returns indexed by date.
    policy:
        Either a single policy name (``"drop"``, ``"ffill"``, ``"zero"``), or a
        mapping of column names to policies.  The special key ``"default"`` sets
        the default policy when a mapping is provided.
    limit:
        Global forward-fill limit, or a mapping of column names to limits.
    columns:
        Optional iterable restricting which columns participate.  Columns not in
        ``columns`` are returned unchanged.

    Returns
    -------
    tuple
        ``(frame, result)`` where ``frame`` is the transformed DataFrame and
        ``result`` is a :class:`MissingPolicyResult` capturing the effective
        policy, limits, fills and any dropped columns.
    """

    cols = list(columns) if columns is not None else list(df.columns)
    work = df.copy()

    default_policy, per_column_policy = _resolve_mapping(policy, "drop")
    default_limit, per_column_limit = _resolve_limits(limit)

    filled_counts: dict[str, int] = {}
    dropped: list[str] = []
    applied_policy: dict[str, str] = {}
    limit_used: dict[str, int | None] = {}

    result_columns: dict[str, pd.Series] = {col: work[col] for col in df.columns if col not in cols}
    for col in cols:
        series = work[col]
        col_policy = per_column_policy.get(col, default_policy)
        applied_policy[col] = col_policy
        col_limit = per_column_limit.get(col, default_limit)
        if col_policy == "drop":
            if series.isna().any() and enforce_completeness:
                dropped.append(col)
                continue
            result_columns[col] = series
            limit_used[col] = None
            continue
        if col_policy == "ffill":
            filled_before = int(series.isna().sum())
            series = series.ffill(limit=col_limit)
            filled_after = int(series.isna().sum())
            filled_counts[col] = filled_before - filled_after
            limit_used[col] = col_limit
            if series.isna().any():
                has_alternative = len(cols) > 1 or result_columns
                if enforce_completeness and has_alternative:
                    dropped.append(col)
                    continue
            result_columns[col] = series
            continue
        if col_policy == "zero":
            filled_before = int(series.isna().sum())
            if filled_before:
                series = series.fillna(0.0)
            filled_counts[col] = filled_before
            limit_used[col] = None
            result_columns[col] = series
            continue
        raise AssertionError(f"Unhandled policy: {col_policy}")

    out = pd.DataFrame(result_columns, index=work.index)

    summary = _policy_display(
        default_policy,
        per_column_policy,
        default_limit,
        per_column_limit,
    )

    result = MissingPolicyResult(
        policy=applied_policy,
        default_policy=default_policy,
        limit=limit_used,
        default_limit=default_limit,
        filled=filled_counts,
        dropped_assets=tuple(dropped),
        summary=summary,
    )

    return out, result
