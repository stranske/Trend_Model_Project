"""Boundary contracts for the NL volatility-target confirmation policy."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trend_analysis.config.model import RiskSettings
from trend_analysis.config.patch import (
    VOLATILITY_TARGET_RISK_POLICY,
    ConfigPatch,
    PatchOperation,
    RiskFlag,
    risky_patch_flags,
)


@pytest.mark.parametrize(
    ("path", "value", "expects_leverage_flag", "expected_confirmation_flags"),
    [
        ("vol_adjust.target_vol", 0.149, False, []),
        ("vol_adjust.target_vol", 0.15, False, []),
        ("vol_adjust.target_vol", 0.151, True, [RiskFlag.INCREASES_LEVERAGE.value]),
        ("vol_adjust", {"target_vol": 0.149}, False, []),
        ("vol_adjust", {"target_vol": 0.15}, False, []),
        ("vol_adjust", {"target_vol": 0.151}, True, [RiskFlag.INCREASES_LEVERAGE.value]),
    ],
)
def test_decimal_annualized_volatility_confirmation_boundary(
    path: str,
    value: object,
    expects_leverage_flag: bool,
    expected_confirmation_flags: list[str],
) -> None:
    operation = PatchOperation(
        op="merge" if path == "vol_adjust" else "set", path=path, value=value
    )

    patch = ConfigPatch(operations=[operation], summary="Set decimal annualized volatility target.")

    assert (RiskFlag.INCREASES_LEVERAGE in patch.risk_flags) is expects_leverage_flag
    assert risky_patch_flags(patch) == expected_confirmation_flags


def test_threshold_boundary_uses_strictly_greater_than() -> None:
    patch = ConfigPatch(
        operations=[PatchOperation(op="set", path="vol_adjust.target_vol", value=0.15)],
        summary="Keep the 15% decimal annualized volatility target.",
    )

    assert RiskFlag.INCREASES_LEVERAGE not in patch.risk_flags


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_risk_settings_reject_non_finite_target_volatility(value: float) -> None:
    with pytest.raises(ValidationError, match="vol_adjust.target_vol must be finite"):
        RiskSettings.model_validate({"target_vol": value})

    assert not VOLATILITY_TARGET_RISK_POLICY.requires_confirmation(value)
