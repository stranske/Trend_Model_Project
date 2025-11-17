"""Transaction cost helpers shared across backtests and reports."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CostModel"]


@dataclass(frozen=True, slots=True)
class CostModel:
    """Linear cost model applied to turnover events."""

    bps_per_trade: float = 0.0
    slippage_bps: float = 0.0

    def __post_init__(self) -> None:
        _ensure_non_negative(self.bps_per_trade, "bps_per_trade")
        _ensure_non_negative(self.slippage_bps, "slippage_bps")

    @property
    def total_bps(self) -> float:
        """Combined basis-point impact from explicit and slippage costs."""

        return float(self.bps_per_trade) + float(self.slippage_bps)

    @property
    def multiplier(self) -> float:
        """Return the decimal multiplier applied to turnover."""

        return self.total_bps / 10000.0

    def turnover_cost(self, turnover: float) -> float:
        """Return the cost deduction for ``turnover`` units of trading."""

        if turnover <= 0:
            return 0.0
        return float(turnover) * self.multiplier

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serialisable view of the model parameters."""

        return {
            "bps_per_trade": float(self.bps_per_trade),
            "slippage_bps": float(self.slippage_bps),
            "total_bps": float(self.total_bps),
        }

    @classmethod
    def from_legacy(cls, transaction_cost_bps: float) -> "CostModel":
        """Build a model using the historical single-parameter contract."""

        return cls(bps_per_trade=float(transaction_cost_bps), slippage_bps=0.0)


def _ensure_non_negative(value: float, label: str) -> None:
    if float(value) < 0:
        raise ValueError(f"{label} must be non-negative")
