"""Cash policy definitions for rebalancing outputs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CashPolicy:
    """Configuration for handling implicit cash in rebalancing outputs."""

    explicit_cash: bool = False
    cash_return_source: str = "risk_free"
    normalize_weights: bool = False


__all__ = ["CashPolicy"]
