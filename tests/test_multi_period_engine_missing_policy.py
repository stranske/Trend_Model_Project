import pandas as pd
import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List

from trend_analysis.multi_period import engine as mp_engine


@dataclass
class DummyConfig:
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
    benchmarks: List[Any] = field(default_factory=list)
    run: Dict[str, Any] = field(default_factory=lambda: {"monthly_cost": 0.0})
    seed: int = 123

    def model_dump(self) -> Dict[str, Any]:
        return {
            "multi_period": self.multi_period,
            "portfolio": self.portfolio,
            "vol_adjust": self.vol_adjust,
        }


def _fake_pipeline_result() -> mp_engine.DiagnosticResult[dict[str, Any]]:
    return mp_engine.DiagnosticResult(value={"selected_funds": []}, diagnostic=None)


def test_missing_policy_diagnostic_skipped_for_user_supplied_frames(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = DummyConfig()
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "FundA": [0.1, None],
        }
    )

    calls: list[tuple[Any, Any]] = []

    def fake_apply_missing_policy(frame: pd.DataFrame, *, policy: Any, limit: Any):
        calls.append((policy, limit))
        return frame, {"policy": policy, "limit": limit}

    monkeypatch.setattr(mp_engine, "apply_missing_policy", fake_apply_missing_policy)
    monkeypatch.setattr(mp_engine, "_call_pipeline_with_diag", lambda *_, **__: _fake_pipeline_result())

    caplog.set_level("INFO")

    results = mp_engine.run(cfg, df=df)

    assert not calls  # missing policy must be bypassed entirely
    assert results
    diag = results[0]["missing_policy_diagnostic"]
    assert diag["applied"] is False
    assert diag["reason"] == "user_supplied_data_without_policy"
    assert "Missing-data policy skipped" in caplog.text


def test_missing_policy_diagnostic_applied_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = DummyConfig()
    cfg.data = {"missing_policy": "bfill", "csv_path": "unused.csv"}
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "FundA": [0.1, None],
        }
    )

    calls: list[tuple[Any, Any]] = []

    def fake_apply_missing_policy(frame: pd.DataFrame, *, policy: Any, limit: Any):
        calls.append((policy, limit))
        return frame.fillna(0.0), {"policy": policy, "limit": limit}

    monkeypatch.setattr(mp_engine, "apply_missing_policy", fake_apply_missing_policy)
    monkeypatch.setattr(mp_engine, "_call_pipeline_with_diag", lambda *_, **__: _fake_pipeline_result())

    results = mp_engine.run(cfg, df=df)

    assert calls
    assert calls[0][0] == "bfill"
    diag = results[0]["missing_policy_diagnostic"]
    assert diag["applied"] is True
    assert diag["policy"] == "bfill"
    assert diag["limit"] is None
