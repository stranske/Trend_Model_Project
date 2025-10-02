from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = ["MissingPolicyResult", "apply_missing_policy"]


@dataclass(frozen=True)
class MissingPolicyResult:
    """Result metadata produced when applying a missing-data policy."""

    policy: str
    limit: int | None
    dropped_assets: tuple[str, ...]
    filled_cells: tuple[tuple[str, int], ...]

    @property
    def total_filled(self) -> int:
        return sum(count for _, count in self.filled_cells)


def apply_missing_policy(
    df: pd.DataFrame, policy: str, limit: int | None = None
) -> tuple[pd.DataFrame, MissingPolicyResult]:
    """Apply the missing-data policy to ``df`` column by column.

    Parameters
    ----------
    df:
        DataFrame indexed by ``Date`` with asset returns in the remaining columns.
    policy:
        One of ``"drop"``, ``"ffill"`` or ``"zero"``. Comparison is case-insensitive.
    limit:
        Maximum consecutive observations to fill when ``policy='ffill'``.

    Returns
    -------
    tuple[pd.DataFrame, MissingPolicyResult]
        The cleaned DataFrame and structured metadata describing the outcome.

    Raises
    ------
    ValueError
        If ``policy`` is unknown or ``limit`` is negative.
    """

    if limit is not None:
        try:
            limit_value = int(limit)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
            raise ValueError("limit must be an integer when provided") from exc
        if limit_value < 0:
            raise ValueError("limit cannot be negative")
        limit = limit_value

    policy_normalised = policy.lower().strip()
    if policy_normalised not in {"drop", "ffill", "zero"}:
        raise ValueError("policy must be one of {'drop', 'ffill', 'zero'}")

    cleaned = pd.DataFrame(index=df.index)
    dropped: list[str] = []
    filled: list[tuple[str, int]] = []

    for column in df.columns:
        series = df[column]
        if policy_normalised == "drop":
            if series.isna().any():
                dropped.append(column)
                continue
            cleaned[column] = series
            continue

        if policy_normalised == "ffill":
            kwargs: dict[str, int] = {}
            if limit is not None:
                kwargs["limit"] = limit
            filled_series = series.ffill(**kwargs)
            filled_count = int(series.isna().sum() - filled_series.isna().sum())
            if filled_count:
                filled.append((column, filled_count))
            if filled_series.isna().all():
                dropped.append(column)
                continue
            cleaned[column] = filled_series
            continue

        # zero policy
        filled_series = series.fillna(0.0)
        filled_count = int(series.isna().sum())
        if filled_count:
            filled.append((column, filled_count))
        if filled_series.isna().all():
            dropped.append(column)
            continue
        cleaned[column] = filled_series

    result = MissingPolicyResult(
        policy=policy_normalised,
        limit=limit,
        dropped_assets=tuple(dropped),
        filled_cells=tuple((col, count) for col, count in filled),
    )
    return cleaned, result
