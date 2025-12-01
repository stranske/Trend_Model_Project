from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import pandas as pd
import pytest

import trend_analysis.multi_period.engine as mp_engine


@dataclass
class DummyCfg:
    multi_period: Dict[str, Any] = field(
        default_factory=lambda: {
            "frequency": "M",
            "in_sample_len": 1,
            "out_sample_len": 1,
            "start": "2020-01",
            "end": "2020-02",
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
    benchmarks: Dict[str, Any] = field(default_factory=dict)
    run: Dict[str, Any] = field(default_factory=lambda: {"monthly_cost": 0.0})
    performance: Dict[str, Any] = field(default_factory=lambda: {"enable_cache": False})
    seed: int = 7

    def model_dump(self) -> Dict[str, Any]:
        return {
            "multi_period": self.multi_period,
            "portfolio": self.portfolio,
            "vol_adjust": self.vol_adjust,
        }


def test_run_sets_flag_when_missing_policy_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = DummyCfg()

    raw_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "FundA": [0.1, None, 0.3],
        }
    )

    applied: dict[str, Any] = {}

    def fake_loader(path: str, **kwargs: object) -> pd.DataFrame:
        applied["loader_kwargs"] = kwargs
        return raw_df

    def fake_apply_policy(frame: pd.DataFrame, *, policy: object, limit: object):
        applied["policy"] = policy
        applied["limit"] = limit
        return frame.fillna(0.0), {"policy": policy}

    def fake_call_pipeline(*args: Any, **kwargs: Any):
        return mp_engine.DiagnosticResult(value={"result": "ok"}, diagnostic=None)

    monkeypatch.setattr(mp_engine, "load_csv", fake_loader)
    monkeypatch.setattr(mp_engine, "apply_missing_policy", fake_apply_policy)
    monkeypatch.setattr(mp_engine, "_call_pipeline_with_diag", fake_call_pipeline)

    results = mp_engine.run(cfg, df=None)

    assert applied["policy"] == "ffill"
    assert applied["limit"] is None
    assert all(result["missing_policy_applied"] is True for result in results)


def test_run_logs_and_flags_when_missing_policy_skipped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = DummyCfg()

    user_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "FundA": [0.1, 0.2],
        }
    )

    def forbid_apply(*args: object, **kwargs: object):  # pragma: no cover - sanity guard
        raise AssertionError("apply_missing_policy should not be invoked when skipping")

    def fake_call_pipeline(*args: Any, **kwargs: Any):
        return mp_engine.DiagnosticResult(value={"result": "ok"}, diagnostic=None)

    monkeypatch.setattr(mp_engine, "apply_missing_policy", forbid_apply)
    monkeypatch.setattr(mp_engine, "_call_pipeline_with_diag", fake_call_pipeline)

    with caplog.at_level("INFO"):
        results = mp_engine.run(cfg, df=user_df)

    assert any("Skipping missing-data policy" in rec.message for rec in caplog.records)
    assert all(result["missing_policy_applied"] is False for result in results)
