from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, Iterable

import pandas as pd

import trend_analysis.multi_period.engine as mp_engine
import trend_analysis.selector as selector_mod


@dataclass
class MinimalConfig:
    """Configuration focused on exercising branch-heavy paths."""

    multi_period: Dict[str, Any] = field(
        default_factory=lambda: {
            "frequency": "M",
            "in_sample_len": 2,
            "out_sample_len": 1,
            "start": "2020-01",
            "end": "2020-03",
        }
    )
    data: Dict[str, Any] = field(default_factory=lambda: {"csv_path": "unused.csv"})
    portfolio: Dict[str, Any] = field(
        default_factory=lambda: {
            "policy": "threshold_hold",
            "random_n": 4,
            "transaction_cost_bps": 0.0,
            "max_turnover": 1.0,
            "threshold_hold": {"target_n": 3, "metric": "Sharpe"},
            "constraints": {
                "max_funds": 1,
                "min_weight": 0.05,
                "max_weight": 0.6,
                "min_weight_strikes": 1,
            },
            "weighting": {"name": "mystery", "params": {}},
            "indices_list": None,
        }
    )
    vol_adjust: Dict[str, Any] = field(default_factory=lambda: {"target_vol": 1.0})
    benchmarks: Dict[str, Any] = field(default_factory=dict)
    run: Dict[str, Any] = field(default_factory=lambda: {"monthly_cost": 0.0})
    seed: int = 99

    def model_dump(self) -> Dict[str, Any]:
        return {
            "multi_period": self.multi_period,
            "portfolio": self.portfolio,
            "vol_adjust": self.vol_adjust,
        }


class SelectorStub:
    def __init__(self, ordering: Iterable[str]) -> None:
        self._ordering = list(ordering)

    def select(self, score_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        ordered = [ix for ix in self._ordering if ix in score_frame.index]
        selected = score_frame.loc[ordered]
        return selected, selected


class WeightingStub:
    """Return intentionally oversized weights to exercise bound logic."""

    def __init__(self) -> None:
        self.updates: list[tuple[pd.Series, int]] = []

    def weight(
        self, selected: pd.DataFrame, date: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        del date
        base = [0.7, 0.6, 0.5]
        values = pd.Series(base[: len(selected)], index=selected.index, dtype=float)
        return values.to_frame("weight")

    def update(self, scores: pd.Series, days: int) -> None:
        self.updates.append((scores.astype(float), int(days)))


class IdentityRebalancer:
    def __init__(self, *_: Any) -> None:
        pass

    def apply_triggers(self, prev_weights: pd.Series, _: pd.DataFrame) -> pd.Series:
        return prev_weights


def _stub_run_analysis(*_: Any, **__: Any) -> Dict[str, Any]:
    return {
        "selected_funds": ["FundA"],
        "in_sample_scaled": pd.DataFrame(),
        "out_sample_scaled": pd.DataFrame(),
        "in_sample_stats": {},
        "out_sample_stats": {},
        "out_sample_stats_raw": {},
        "in_ew_stats": (),
        "out_ew_stats": (),
        "out_ew_stats_raw": (),
        "in_user_stats": (),
        "out_user_stats": (),
        "out_user_stats_raw": (),
        "ew_weights": {},
        "fund_weights": {},
        "benchmark_stats": {},
        "benchmark_ir": {},
        "score_frame": pd.DataFrame(),
        "weight_engine_fallback": None,
    }


def test_run_covers_threshold_hold_branches(monkeypatch):
    cfg = MinimalConfig()
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "FundA": [0.01, 0.02, 0.03],
            "FundB": [0.015, 0.0, 0.01],
            "FundC": [0.02, 0.01, 0.0],
        }
    )

    periods = [
        SimpleNamespace(
            in_start="2020-01",
            in_end="2020-02",
            out_start="2020-03",
            out_end="2020-03",
        )
    ]

    monkeypatch.setattr(mp_engine, "generate_periods", lambda _: periods)
    monkeypatch.setattr(mp_engine, "_run_analysis", _stub_run_analysis)
    monkeypatch.setattr(
        selector_mod,
        "create_selector_by_name",
        lambda *_, **__: SelectorStub(["FundA", "FundB", "FundC"]),
    )
    monkeypatch.setattr(mp_engine, "EqualWeight", lambda: WeightingStub())
    monkeypatch.setattr(mp_engine, "Rebalancer", lambda *_: IdentityRebalancer())

    results = mp_engine.run(cfg, df=df)

    assert len(results) == 1
    entry = results[0]
    assert entry["manager_changes"]  # events recorded from branch coverage
