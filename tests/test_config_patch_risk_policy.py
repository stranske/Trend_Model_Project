"""Boundary contracts for the NL volatility-target confirmation policy."""

from __future__ import annotations

import pytest

from trend_analysis.config.patch import ConfigPatch, PatchOperation, RiskFlag


@pytest.mark.parametrize(
    ("path", "value", "expects_leverage_flag"),
    [
        ("vol_adjust.target_vol", 0.149, False),
        ("vol_adjust.target_vol", 0.15, False),
        ("vol_adjust.target_vol", 0.151, True),
        ("vol_adjust", {"target_vol": 0.149}, False),
        ("vol_adjust", {"target_vol": 0.15}, False),
        ("vol_adjust", {"target_vol": 0.151}, True),
    ],
)
def test_decimal_annualized_volatility_confirmation_boundary(
    path: str,
    value: object,
    expects_leverage_flag: bool,
) -> None:
    operation = PatchOperation(
        op="merge" if path == "vol_adjust" else "set", path=path, value=value
    )

    patch = ConfigPatch(operations=[operation], summary="Set decimal annualized volatility target.")

    assert (RiskFlag.INCREASES_LEVERAGE in patch.risk_flags) is expects_leverage_flag


def test_threshold_boundary_uses_strictly_greater_than() -> None:
    patch = ConfigPatch(
        operations=[PatchOperation(op="set", path="vol_adjust.target_vol", value=0.15)],
        summary="Keep the 15% decimal annualized volatility target.",
    )

    assert RiskFlag.INCREASES_LEVERAGE not in patch.risk_flags
