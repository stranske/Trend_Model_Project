from types import SimpleNamespace

from trend.reporting import unified
from trend_analysis.pipeline_helpers import (
    _build_trend_spec as build_runtime_trend_spec,
)
from trend_analysis.signals import TrendSpec
from trend_model.spec import _build_trend_spec as build_compatibility_trend_spec


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


def test_shared_parser_agrees_across_entrypoints() -> None:
    payload = {
        "signals": {"trend_window": 42, "trend_lag": 2, "trend_zscore": 2.0},
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
