from __future__ import annotations

from collections import UserDict
from types import SimpleNamespace

import numpy as np
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


def test_run_simulation_passes_regime_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _DummyConfig()
    cfg.regime = {"enabled": True, "proxy": "SPX"}
    returns = _make_returns()
    captured: dict[str, object] = {}

    def _capture_run_analysis(*_args, **kwargs):
        captured["regime_cfg"] = kwargs.get("regime_cfg")
        return None

    monkeypatch.setattr(api, "_run_analysis", _capture_run_analysis)

    api.run_simulation(cfg, returns)

    assert captured["regime_cfg"] == {"enabled": True, "proxy": "SPX"}


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
    assert all(isinstance(metric, str) for metric in regime_sanitized["User / Risk-On"].keys())


def test_run_simulation_builds_user_weight_combined_and_dedupes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _DummyConfig(metrics={"registry": ["Sharpe"]})
    returns = _make_returns()

    in_index = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
    out_index = pd.to_datetime(["2020-03-31", "2020-04-30"])
    in_scaled = pd.DataFrame(
        {"FundA": [0.01, 0.02, 0.03], "FundB": [0.0, 0.01, -0.02]},
        index=in_index,
    )
    out_scaled = pd.DataFrame(
        {"FundA": [0.04, 0.05], "FundB": [0.02, -0.01]},
        index=out_index,
    )

    payload = UserDict(
        {
            "out_sample_stats": {"FundA": SimpleNamespace(alpha=1.0, beta=2.0)},
            "in_sample_scaled": in_scaled,
            "out_sample_scaled": out_scaled,
            "ew_weights": {"FundA": 0.5, "FundB": 0.5},
            "fund_weights": {"FundA": 0.8, "FundB": 0.2},
        }
    )

    monkeypatch.setattr(api, "_run_analysis", lambda *a, **k: payload)

    result = api.run_simulation(cfg, returns)

    from trend_analysis.pipeline import calc_portfolio_returns

    user_weights = np.array([0.8, 0.2])
    ew_weights = np.array([0.5, 0.5])
    expected_user = pd.concat(
        [
            calc_portfolio_returns(user_weights, in_scaled),
            calc_portfolio_returns(user_weights, out_scaled),
        ]
    )
    expected_user = expected_user[~expected_user.index.duplicated(keep="last")].sort_index()
    expected_equal = pd.concat(
        [
            calc_portfolio_returns(ew_weights, in_scaled),
            calc_portfolio_returns(ew_weights, out_scaled),
        ]
    )
    expected_equal = expected_equal[~expected_equal.index.duplicated(keep="last")].sort_index()

    user_series = result.details.get("portfolio_user_weight_combined")
    equal_series = result.details.get("portfolio_equal_weight_combined")

    assert user_series is not None
    assert equal_series is not None
    assert user_series.index.is_unique
    assert user_series.index.is_monotonic_increasing
    assert equal_series.index.is_unique
    assert equal_series.index.is_monotonic_increasing
    pd.testing.assert_series_equal(user_series, expected_user)
    pd.testing.assert_series_equal(equal_series, expected_equal)
    assert (
        user_series.loc[pd.Timestamp("2020-03-31")] == expected_user.loc[pd.Timestamp("2020-03-31")]
    )


def test_run_simulation_combined_keeps_last_duplicate_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _DummyConfig(metrics={"registry": ["Sharpe"]})
    returns = _make_returns()

    in_index = pd.to_datetime(["2020-02-29", "2020-03-31"])
    out_index = pd.to_datetime(["2020-03-31", "2020-04-30"])
    in_scaled = pd.DataFrame(
        {"FundA": [0.01, 0.02], "FundB": [0.03, 0.04]},
        index=in_index,
    )
    out_scaled = pd.DataFrame(
        {"FundA": [0.12, 0.01], "FundB": [-0.08, 0.02]},
        index=out_index,
    )

    payload = UserDict(
        {
            "out_sample_stats": {"FundA": SimpleNamespace(alpha=1.0, beta=2.0)},
            "in_sample_scaled": in_scaled,
            "out_sample_scaled": out_scaled,
            "ew_weights": {"FundA": 0.5, "FundB": 0.5},
            "fund_weights": {"FundA": 0.7, "FundB": 0.3},
        }
    )

    monkeypatch.setattr(api, "_run_analysis", lambda *a, **k: payload)

    result = api.run_simulation(cfg, returns)

    from trend_analysis.pipeline import calc_portfolio_returns

    user_weights = np.array([0.7, 0.3])
    ew_weights = np.array([0.5, 0.5])
    user_out = calc_portfolio_returns(user_weights, out_scaled).loc[pd.Timestamp("2020-03-31")]
    user_in = calc_portfolio_returns(user_weights, in_scaled).loc[pd.Timestamp("2020-03-31")]
    equal_out = calc_portfolio_returns(ew_weights, out_scaled).loc[pd.Timestamp("2020-03-31")]
    equal_in = calc_portfolio_returns(ew_weights, in_scaled).loc[pd.Timestamp("2020-03-31")]

    user_series = result.details.get("portfolio_user_weight_combined")
    equal_series = result.details.get("portfolio_equal_weight_combined")

    assert user_series is not None
    assert equal_series is not None
    assert user_series.loc[pd.Timestamp("2020-03-31")] == user_out
    assert user_series.loc[pd.Timestamp("2020-03-31")] != user_in
    assert equal_series.loc[pd.Timestamp("2020-03-31")] == equal_out
    assert equal_series.loc[pd.Timestamp("2020-03-31")] != equal_in


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
