from __future__ import annotations

from collections import UserDict
from types import SimpleNamespace

import pandas as pd
import pytest

from trend_analysis import api


class _DummyConfig:
    def __init__(self, *, metrics: dict[str, object] | None = None) -> None:
        self.seed = 17
        self.sample_split = {
            "in_start": "2020-01",
            "in_end": "2020-03",
            "out_start": "2020-04",
            "out_end": "2020-06",
        }
        self.metrics = metrics or {}
        self.vol_adjust = {"target_vol": 1.0}
        self.run = {"monthly_cost": 0.0}
        self.portfolio = {
            "selection_mode": "all",
            "random_n": 4,
            "custom_weights": None,
            "rank": {},
            "manual_list": None,
            "indices_list": None,
        }
        self.benchmarks: dict[str, object] = {}


def _make_returns() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    data = {
        "Date": dates,
        "FundA": [0.01, 0.02, 0.03, 0.04],
        "FundB": [0.00, 0.01, -0.01, 0.02],
    }
    return pd.DataFrame(data)


def test_run_simulation_handles_none_result(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _DummyConfig()
    returns = _make_returns()

    monkeypatch.setattr(api, "_run_analysis", lambda *a, **k: None)

    result = api.run_simulation(cfg, returns)

    assert result.metrics.empty
    assert result.details == {}
    assert result.fallback_info is None
    assert result.environment["python"]


def test_run_simulation_sanitizes_details_and_combines_portfolio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _DummyConfig(metrics={"registry": ["Sharpe"]})
    returns = _make_returns()

    in_scaled = returns.set_index("Date").iloc[:2]
    out_scaled = returns.set_index("Date").iloc[2:]

    stats_obj = SimpleNamespace(alpha=1.0, beta=2.0)
    bench_ir = {"bench": {"FundA": 0.5, "FundB": 0.2, "equal_weight": 0.1}}
    regime_table = pd.DataFrame(
        {
            ("User", "Risk-On"): [0.12, 1.1, 5],
            ("User", "Risk-Off"): [0.04, 0.8, 2],
        },
        index=["CAGR", "Sharpe", "Observations"],
    )

    payload = UserDict(
        {
            "out_sample_stats": {"FundA": stats_obj, "FundB": stats_obj},
            "benchmark_ir": bench_ir,
            "selected_funds": ["FundA", "FundB"],
            "weights_user_weight": {"FundA": 0.6, "FundB": 0.4},
            "in_sample_scaled": in_scaled,
            "out_sample_scaled": out_scaled,
            "ew_weights": {"FundA": 0.5, "FundB": 0.5},
            "weight_engine_fallback": {"engine": "test"},
            "weird_keys": {pd.Timestamp("2020-01-31"): {"value": 1}},
            "performance_by_regime": regime_table,
        }
    )

    monkeypatch.setattr(api, "_run_analysis", lambda *a, **k: payload)

    result = api.run_simulation(cfg, returns)

    assert set(result.metrics.columns) >= {"alpha", "beta", "ir_bench"}
    assert "portfolio_equal_weight_combined" in result.details
    assert result.fallback_info == {"engine": "test"}
    assert isinstance(result.details_sanitized, dict)
    sanitized_keys = result.details_sanitized["weird_keys"].keys()
    assert all(isinstance(k, str) for k in sanitized_keys)
    regime_sanitized = result.details_sanitized["performance_by_regime"]
    assert all(isinstance(col, str) for col in regime_sanitized.keys())
    assert "User / Risk-On" in regime_sanitized
    assert all(
        isinstance(metric, str) for metric in regime_sanitized["User / Risk-On"].keys()
    )


def test_run_simulation_handles_unexpected_result_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _DummyConfig()
    returns = _make_returns()

    monkeypatch.setattr(api, "_run_analysis", lambda *a, **k: object())

    result = api.run_simulation(cfg, returns)

    assert result.metrics.empty
    assert result.details == {}


class _StatsProxy:
    def __init__(self) -> None:
        self._store = {"FundA": SimpleNamespace(alpha=3.0, beta=4.0)}

    def items(self):
        return self._store.items()


def test_run_simulation_handles_mapping_payload_and_logging_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _DummyConfig()
    returns = _make_returns()

    payload = {
        "out_sample_stats": _StatsProxy(),
        "selected_funds": ["FundA"],
        "weights_equal_weight": {"FundA": 1.0},
    }

    events: list[str] = []

    def fake_log_step(run_id: str, event: str, message: str, **kwargs: object) -> None:
        events.append(event)
        if event == "selection":
            raise RuntimeError("boom")

    monkeypatch.setattr(api, "_run_analysis", lambda *a, **k: payload)
    monkeypatch.setattr(api, "_log_step", fake_log_step)

    result = api.run_simulation(cfg, returns)

    assert "alpha" in result.metrics.columns
    assert events[:2] == ["api_start", "analysis_start"]
    assert "portfolio_equal_weight_combined" not in result.details
