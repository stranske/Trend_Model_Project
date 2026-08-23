from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import yaml

from trend.reporting import unified
from trend.spec import load_run_spec_from_mapping


def _default_payload() -> dict[str, object]:
    payload = yaml.safe_load(Path("config/defaults.yml").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_param_summary_reports_portfolio_vol_targets_when_scaling_is_raw() -> None:
    spec = load_run_spec_from_mapping(_default_payload(), base_path=Path("config"))

    params = dict(unified._build_param_summary(spec.config, spec=spec))

    assert params["Signal scaling"] == "Raw"
    assert params["Target volatility"] == "10.0%"
    assert params["Floor volatility"] == "4.0%"
    assert params["Warm-up periods"] == "0"


def test_param_summary_reports_vol_targets_when_adjusted() -> None:
    payload = deepcopy(_default_payload())
    payload["signals"] = {"window": 63, "lag": 1}
    payload["vol_adjust"] = {"enabled": True, "target_vol": 0.15}
    spec = load_run_spec_from_mapping(payload, base_path=Path("config"))

    params = dict(unified._build_param_summary(spec.config, spec=spec))

    assert params["Signal scaling"] == "Vol-adjusted"
    assert params["Target volatility"] == "15.0%"


def test_param_summary_omits_portfolio_vol_targets_when_disabled() -> None:
    payload = _default_payload()
    payload["vol_adjust"]["enabled"] = False
    spec = load_run_spec_from_mapping(payload, base_path=Path("config"))

    params = dict(unified._build_param_summary(spec.config, spec=spec))

    assert params["Signal scaling"] == "Raw"
    assert "Target volatility" not in params
    assert "Floor volatility" not in params
    assert "Warm-up periods" not in params


def test_param_summary_omits_non_finite_warmup_periods() -> None:
    config = SimpleNamespace(vol_adjust={"enabled": True, "warmup_periods": float("nan")})

    params = dict(unified._build_param_summary(config))

    assert "Warm-up periods" not in params
