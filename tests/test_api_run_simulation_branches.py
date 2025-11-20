"""Additional coverage-focused tests for :mod:`trend_analysis.api`."""

from __future__ import annotations

import sys
from collections import UserDict
from types import ModuleType, SimpleNamespace
from typing import cast

import pandas as pd
import pytest

from trend_analysis import api


def _make_returns() -> pd.DataFrame:
    dates = pd.date_range("2022-01-31", periods=3, freq="ME")
    return pd.DataFrame({"Date": dates, "FundA": [0.01, 0.02, 0.03]})


def _make_config(**overrides: object) -> SimpleNamespace:
    base: dict[str, object] = {
        "sample_split": {
            "in_start": "2021-01",
            "in_end": "2021-12",
            "out_start": "2022-01",
            "out_end": "2022-03",
        },
        "metrics": {},
        "vol_adjust": {},
        "portfolio": {},
        "run": {},
        "benchmarks": {},
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_run_simulation_handles_missing_result(monkeypatch):
    """When the pipeline returns ``None`` an empty ``RunResult`` is
    produced."""

    config = _make_config()
    returns = _make_returns()

    monkeypatch.setattr(api, "_run_analysis", lambda *_, **__: None)

    result = api.run_simulation(config, returns)

    assert result.metrics.empty
    assert result.details == {}
    assert result.fallback_info is None
    # The helper falls back to a deterministic default seed when absent.
    assert result.seed == 42
    assert set(result.environment) == {"python", "numpy", "pandas"}


def test_run_simulation_populates_metrics_and_fallback(monkeypatch):
    """Exercise the branches that construct metrics and fallback metadata."""

    metrics_list = ["Sharpe", "Sortino"]
    config = _make_config(
        seed=123,
        metrics={"registry": metrics_list},
        vol_adjust={"target_vol": 1.5},
        portfolio={
            "selection_mode": "rank",
            "random_n": 4,
            "rank": {"inclusion": "top_n"},
        },
        run={"monthly_cost": 0.05},
    )
    returns = _make_returns()

    captured: dict[str, object] = {}

    class DummyRiskStatsConfig:
        def __init__(self, **kwargs: object) -> None:
            calls = cast(
                list[dict[str, object]],
                captured.setdefault("risk_stats_calls", []),
            )
            calls.append(kwargs)

    def fake_canonical_metric_list(values: list[str]) -> list[str]:
        captured["canonical_metrics"] = tuple(values)
        return [value.upper() for value in values]

    rank_module = ModuleType("trend_analysis.core.rank_selection")
    rank_module.RiskStatsConfig = DummyRiskStatsConfig
    rank_module.canonical_metric_list = fake_canonical_metric_list
    monkeypatch.setitem(sys.modules, "trend_analysis.core.rank_selection", rank_module)

    def fake_run_analysis(*args: object, **kwargs: object) -> dict[str, object]:
        captured["run_analysis_args"] = (args, kwargs)
        stats_obj = SimpleNamespace(
            items=lambda: [
                ("FundA", SimpleNamespace(alpha=1.2, beta=0.8)),
                ("FundB", SimpleNamespace(alpha=0.5, beta=0.4)),
            ]
        )
        return {
            "out_sample_stats": stats_obj,
            "benchmark_ir": {
                "bench": {
                    "FundA": 0.3,
                    "equal_weight": 0.0,
                    "user_weight": 0.1,
                }
            },
            "weight_engine_fallback": {"engine": "TestEngine", "error": "boom"},
        }

    monkeypatch.setattr(api, "_run_analysis", fake_run_analysis)

    result = api.run_simulation(config, returns)

    # ``canonical_metric_list`` should receive the raw registry values.
    assert captured["canonical_metrics"] == tuple(metrics_list)

    args, kwargs = captured["run_analysis_args"]
    assert kwargs["seed"] == config.seed
    stats_cfg = kwargs["stats_cfg"]
    assert isinstance(stats_cfg, DummyRiskStatsConfig)
    stats_kwargs = captured["risk_stats_calls"][0]
    assert stats_kwargs["metrics_to_run"] == [value.upper() for value in metrics_list]
    assert stats_kwargs["risk_free"] == 0.0

    # Metrics are built from the stats mapping and benchmark IR data.
    assert set(result.metrics.index) == {"FundA", "FundB"}
    assert result.metrics.loc["FundA", "alpha"] == 1.2
    assert result.metrics.loc["FundA", "ir_bench"] == 0.3

    # The fallback payload is surfaced directly on the RunResult.
    assert result.fallback_info == {"engine": "TestEngine", "error": "boom"}

    # The details object is exactly the payload returned by ``_run_analysis``.
    assert result.details["benchmark_ir"]["bench"]["FundA"] == 0.3


def test_run_simulation_accepts_mapping_payload_and_swallows_logging_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The API should coerce Mapping results and tolerate logging failures."""

    config = _make_config()
    returns = _make_returns()

    stats_mapping = {"FundA": SimpleNamespace(alpha=0.9, beta=0.4)}
    bench_ir = {"bench": {"FundA": 0.2}}
    payload = UserDict(
        {
            "out_sample_stats": stats_mapping,
            "selected_funds": ["FundA"],
            "weights_user_weight": None,
            "weights_equal_weight": {"FundA": 1.0},
            "benchmark_ir": bench_ir,
        }
    )

    monkeypatch.setattr(api, "_run_analysis", lambda *_, **__: payload)

    events: list[tuple[str, str]] = []

    def noisy_log_step(run_id: str, step: str, message: str, **extra: object) -> None:
        events.append((step, message))
        if step == "selection":
            raise RuntimeError("log boom")

    monkeypatch.setattr(api, "_log_step", noisy_log_step)

    result = api.run_simulation(config, returns)

    # Mapping payload should be materialised into a dict with metrics captured.
    assert isinstance(result.metrics, pd.DataFrame)
    assert not result.metrics.empty
    assert result.metrics.loc["FundA", "alpha"] == 0.9

    # The logging error should have been swallowed allowing execution to finish.
    assert ("selection", "Funds selected") in events
    assert ("api_end", "run_simulation complete") in events

    # Benchmark IR column is created even when the payload is a Mapping.
    assert result.metrics.loc["FundA", "ir_bench"] == 0.2


def test_run_simulation_populates_structured_results(monkeypatch) -> None:
    config = _make_config()
    returns = _make_returns()

    portfolio = pd.Series(
        [0.01, 0.02], index=pd.date_range("2023-01-31", periods=2, freq="ME")
    )
    metadata = {"costs": {"monthly_cost": 0.001}, "fingerprint": "abc123def456"}

    payload = {
        "out_sample_stats": {
            "FundA": SimpleNamespace(alpha=1.0, beta=0.5),
        },
        "benchmark_ir": {},
        "portfolio_equal_weight_combined": portfolio,
        "fund_weights": {"FundA": 0.6},
        "risk_diagnostics": {
            "final_weights": pd.Series({"FundA": 0.6}),
            "turnover": pd.Series([0.1, 0.2]),
        },
        "metadata": metadata,
    }

    monkeypatch.setattr(api, "_run_analysis", lambda *_, **__: payload)

    result = api.run_simulation(config, returns)

    assert result.analysis is not None
    pd.testing.assert_series_equal(result.portfolio, portfolio.astype(float))
    assert result.analysis.metadata["fingerprint"] == "abc123def456"
    assert hasattr(result, "weights")
    pd.testing.assert_series_equal(
        result.weights.astype(float), pd.Series({"FundA": 0.6})
    )
