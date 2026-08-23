from types import SimpleNamespace

import pytest

from trend.reporting import unified
from trend.spec import _build_trend_spec as build_compatibility_trend_spec
from trend_analysis.pipeline_helpers import (
    _build_trend_spec as build_runtime_trend_spec,
)
from trend_analysis.signals import TrendSpec, trend_spec_from_mapping


def test_report_uses_runtime_numeric_zscore_spec() -> None:
    config = SimpleNamespace(
        sample_split={},
        vol_adjust={},
        portfolio={},
        run={},
        benchmarks={},
        trend_spec=TrendSpec(zscore=True),
    )
    result = SimpleNamespace(details={"signal_spec": TrendSpec(zscore=2.0)})

    params = dict(
        unified._build_param_summary(config, effective_trend_spec=result.details["signal_spec"])
    )

    assert params["Signal z-score"] == "Scale 2"


def test_param_summary_resolves_effective_then_run_then_config_trend_spec() -> None:
    config = SimpleNamespace(
        sample_split={},
        vol_adjust={"enabled": True, "target_vol": 0.15, "floor_vol": 0.05, "warmup_periods": 3},
        portfolio={},
        run={},
        benchmarks={},
        trend_spec=TrendSpec(vol_adjust=False),
    )
    run_spec = SimpleNamespace(trend=TrendSpec(vol_adjust=True))

    effective = dict(
        unified._build_param_summary(
            config,
            spec=run_spec,
            effective_trend_spec=TrendSpec(vol_adjust=False),
        )
    )
    from_run = dict(unified._build_param_summary(config, spec=run_spec))
    from_config = dict(unified._build_param_summary(config))

    assert effective["Signal scaling"] == "Raw"
    assert from_run["Signal scaling"] == "Vol-adjusted"
    assert from_config["Signal scaling"] == "Raw"
    for params in (effective, from_run, from_config):
        assert params["Target volatility"] == "15.0%"
        assert params["Floor volatility"] == "5.0%"
        assert params["Warm-up periods"] == "3"


def test_shared_parser_agrees_across_entrypoints() -> None:
    payload = {
        "signals": {"window": 42, "lag": 2, "zscore": 2.0},
        "vol_adjust": {"enabled": True, "target_vol": 0.15},
    }

    runtime = build_runtime_trend_spec(payload, payload["vol_adjust"])
    compatibility = build_compatibility_trend_spec(payload)

    assert runtime == compatibility
    assert runtime.vol_target == 0.15
    assert runtime.zscore == 2.0


def test_runtime_no_signals_policy_is_preserved() -> None:
    assert build_runtime_trend_spec({"risk_window": 126}, {"enabled": True}) is None


def test_compatibility_no_signals_spec_uses_runtime_fallback() -> None:
    spec = build_compatibility_trend_spec({"vol_adjust": {"enabled": True}})

    assert spec.vol_adjust is False


def test_report_disables_invalid_numeric_zscore_scales() -> None:
    for invalid_scale in (0.0, -1.0, float("nan"), float("inf")):
        params = dict(unified._trend_spec_summary(TrendSpec(zscore=invalid_scale)))
        assert params["Signal z-score"] == "Disabled"


@pytest.mark.parametrize(
    ("signals", "expected_window", "expected_lag", "expected_vol_target"),
    [
        ({"window": float("inf")}, 63, 1, None),
        ({"window": float("nan")}, 63, 1, None),
        ({"lag": float("inf")}, 63, 1, None),
        ({"vol_target": float("nan")}, 63, 1, None),
        ({"vol_target": float("inf")}, 63, 1, None),
    ],
)
def test_trend_spec_from_mapping_rejects_non_finite_numeric_config(
    signals: dict[str, float],
    expected_window: int,
    expected_lag: int,
    expected_vol_target: float | None,
) -> None:
    spec = trend_spec_from_mapping(signals)

    assert spec.window == expected_window
    assert spec.lag == expected_lag
    assert spec.vol_target is expected_vol_target
