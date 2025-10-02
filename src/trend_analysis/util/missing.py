"""Utilities for enforcing missing-data policies on return frames."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable

import pandas as pd

__all__ = ["apply_missing_policy"]


def _coerce_policy(policy: str | None) -> str:
    value = (policy or "drop").strip().lower()
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
    limit: int | Mapping[str, int | None] | None
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
    limit_part = (
        f"limit={default_limit}" if default_limit is not None else "unbounded"
    )
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
    limit: int | Mapping[str, int | None] | None,
    *,
    columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
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
        ``(frame, metadata)`` where ``frame`` is the transformed DataFrame and
        ``metadata`` captures the effective policy plus bookkeeping such as the
        number of fills performed and which columns were dropped.
    """

    cols = list(columns) if columns is not None else list(df.columns)
    work = df.copy()

    default_policy, per_column_policy = _resolve_mapping(policy, "drop")
    default_limit, per_column_limit = _resolve_limits(limit)

    filled_counts: dict[str, int] = {}
    dropped: list[str] = []
    applied_policy: dict[str, str] = {}
    limit_used: dict[str, int | None] = {}

    result_columns: dict[str, pd.Series] = {
        col: work[col] for col in df.columns if col not in cols
    }
    for col in cols:
        series = work[col]
        col_policy = per_column_policy.get(col, default_policy)
        applied_policy[col] = col_policy
        col_limit = per_column_limit.get(col, default_limit)
        if col_policy == "drop":
            if series.isna().any():
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

    metadata = {
        "policy": applied_policy,
        "default_policy": default_policy,
        "limit": limit_used,
        "filled": filled_counts,
        "dropped": dropped,
    }
    metadata["summary"] = _policy_display(
        default_policy,
        per_column_policy,
        default_limit,
        per_column_limit,
    )
    return out, metadata
