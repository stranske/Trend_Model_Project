"""Tests for missing-policy observability in multi-period engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd
import pytest

from trend_analysis.multi_period import engine as mp_engine


@dataclass
class DummyConfig:
    """Minimal config object that satisfies ``mp_engine.run`` dependencies."""

    multi_period: Dict[str, Any] = field(
        default_factory=lambda: {
            "frequency": "M",
            "in_sample_len": 1,
            "out_sample_len": 1,
            "start": "2020-01",
            "end": "2020-03",
        }
    )
    data: Dict[str, Any] = field(default_factory=lambda: {"csv_path": "unused.csv"})
    portfolio: Dict[str, Any] = field(
        default_factory=lambda: {
            "policy": "standard",
            "selection_mode": "all",
            "random_n": 2,
            "custom_weights": None,
            "rank": {},
            "manual_list": None,
            "indices_list": None,
        }
    )
    vol_adjust: Dict[str, Any] = field(default_factory=lambda: {"target_vol": 1.0})
    performance: Dict[str, Any] = field(default_factory=lambda: {"enable_cache": False})
    benchmarks: List[Any] = field(default_factory=list)
    run: Dict[str, Any] = field(default_factory=lambda: {"monthly_cost": 0.0})
    seed: int = 123

    def model_dump(self) -> Dict[str, Any]:
        return {
            "multi_period": self.multi_period,
            "portfolio": self.portfolio,
            "vol_adjust": self.vol_adjust,
        }


def test_missing_policy_skip_flag_and_log(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = DummyConfig()
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "FundA": [0.1, None, 0.2],
        }
    )

    def fake_run_analysis(
        frame: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        assert frame["Date"].tolist() == [
            pd.Timestamp("2020-01-31"),
            pd.Timestamp("2020-03-31"),
        ]
        return {"analysis": "ok"}

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)
    caplog.set_level(logging.INFO)

    results = mp_engine.run(cfg, df=df)

    assert results
    assert all(res.get("missing_policy_applied") is False for res in results)
    assert all(
        res.get("missing_policy_reason") == "skipped_user_supplied_input"
        for res in results
    )
    assert any(
        "Missing-data policy skipped" in rec.getMessage() for rec in caplog.records
    )


def test_missing_policy_applied_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = DummyConfig()
    cfg.data["missing_policy"] = "ffill"
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "FundA": [0.1, None, 0.2],
            "FundB": [None, 0.05, 0.07],
        }
    )

    def fake_run_analysis(
        frame: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        assert not frame.drop(columns=["Date"]).isna().any().any()
        assert len(frame) == 3
        return {"analysis": "ok"}

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)

    results = mp_engine.run(cfg, df=df)

    assert results
    assert all(res.get("missing_policy_applied") is True for res in results)
    assert all(res.get("missing_policy_reason") == "applied" for res in results)
    assert all(res.get("missing_policy_spec") == "ffill" for res in results)
