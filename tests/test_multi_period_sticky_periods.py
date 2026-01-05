from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

import pandas as pd
import pytest

from trend_analysis.multi_period import engine as mp_engine


@dataclass
class StickyConfig:
    multi_period: Dict[str, Any] = field(
        default_factory=lambda: {
            "frequency": "M",
            "in_sample_len": 2,
            "out_sample_len": 1,
            "start": "2020-01",
            "end": "2020-05",
        }
    )
    data: Dict[str, Any] = field(
        default_factory=lambda: {
            "csv_path": "unused.csv",
            "risk_free_column": "RF",
        }
    )
    portfolio: Dict[str, Any] = field(
        default_factory=lambda: {
            "policy": "threshold_hold",
            "random_n": 2,
            "transaction_cost_bps": 0.0,
            "max_turnover": 1.0,
            "threshold_hold": {
                "target_n": 2,
                "metric": "Sharpe",
                "z_exit_soft": -5.0,
                "z_entry_soft": 5.0,
            },
            "constraints": {
                "max_funds": 3,
                "min_weight": 0.0,
                "max_weight": 1.0,
            },
            "weighting": {"name": "equal"},
            "indices_list": [],
        }
    )
    vol_adjust: Dict[str, Any] = field(default_factory=lambda: {"target_vol": 1.0})
    benchmarks: Dict[str, Any] = field(default_factory=dict)
    run: Dict[str, Any] = field(default_factory=lambda: {"monthly_cost": 0.0})
    seed: int = 1

    def model_dump(self) -> Dict[str, Any]:
        return {
            "multi_period": self.multi_period,
            "portfolio": self.portfolio,
            "vol_adjust": self.vol_adjust,
        }


def _sample_df() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=5, freq="ME")
    return pd.DataFrame(
        {
            "Date": dates,
            "A": [0.03, 0.02, 0.01, 0.04, 0.03],
            "B": [0.02, 0.015, 0.02, 0.015, 0.02],
            "C": [0.005, 0.0, -0.005, 0.0, 0.005],
            "RF": [0.001] * len(dates),
        }
    )


def _periods() -> List[Any]:
    return [
        SimpleNamespace(
            in_start="2020-01-31",
            in_end="2020-02-29",
            out_start="2020-03-31",
            out_end="2020-03-31",
        ),
        SimpleNamespace(
            in_start="2020-02-29",
            in_end="2020-03-31",
            out_start="2020-04-30",
            out_end="2020-04-30",
        ),
        SimpleNamespace(
            in_start="2020-03-31",
            in_end="2020-04-30",
            out_start="2020-05-31",
            out_end="2020-05-31",
        ),
    ]


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    def _noop_missing_policy(frame: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
        return frame, {}

    monkeypatch.setattr(mp_engine, "apply_missing_policy", _noop_missing_policy)
    monkeypatch.setattr(mp_engine, "_run_analysis", lambda *args, **kwargs: {})


def _run_with_sticky(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sticky_add: int,
    sticky_drop: int,
    sequence: Iterable[Iterable[str]],
) -> list[dict[str, Any]]:
    cfg = StickyConfig()
    cfg.portfolio["sticky_add_x"] = sticky_add
    cfg.portfolio["sticky_drop_y"] = sticky_drop
    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: _periods())

    seq = [list(s) for s in sequence]

    class SequenceRebalancer:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.calls = 0

        def apply_triggers(self, _prev: pd.Series, _sf: pd.DataFrame, **_kwargs: Any) -> pd.Series:
            idx = min(self.calls, len(seq) - 1)
            holdings = list(seq[idx]) if seq else []
            self.calls += 1
            if not holdings:
                return pd.Series(dtype=float)
            eq = 1.0 / len(holdings)
            return pd.Series({h: eq for h in holdings}, dtype=float)

    monkeypatch.setattr(mp_engine, "Rebalancer", SequenceRebalancer)
    _patch_pipeline(monkeypatch)
    return mp_engine.run(cfg, df=_sample_df())


def _turnover_after_seed(results: list[dict[str, Any]]) -> float:
    return float(sum(float(r.get("turnover", 0.0)) for r in results[1:]))


def test_sticky_add_requires_consecutive_periods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence = [["A", "B", "C"], ["A", "B"]]

    baseline = _run_with_sticky(monkeypatch, sticky_add=1, sticky_drop=1, sequence=sequence)
    delayed = _run_with_sticky(monkeypatch, sticky_add=2, sticky_drop=1, sequence=sequence)

    assert "C" in baseline[1]["selected_funds"]
    assert "C" not in delayed[1]["selected_funds"]
    assert _turnover_after_seed(delayed) < _turnover_after_seed(baseline)


def test_sticky_drop_requires_consecutive_periods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence = [["B"], ["A", "B"]]

    baseline = _run_with_sticky(monkeypatch, sticky_add=1, sticky_drop=1, sequence=sequence)
    delayed = _run_with_sticky(monkeypatch, sticky_add=1, sticky_drop=2, sequence=sequence)

    assert "A" not in baseline[1]["selected_funds"]
    assert "A" in delayed[1]["selected_funds"]
    assert _turnover_after_seed(delayed) < _turnover_after_seed(baseline)


def test_sticky_add_and_drop_reduce_turnover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence = [["A", "B", "C"], ["B"]]

    baseline = _run_with_sticky(monkeypatch, sticky_add=1, sticky_drop=1, sequence=sequence)
    delayed = _run_with_sticky(monkeypatch, sticky_add=2, sticky_drop=2, sequence=sequence)

    assert "A" not in baseline[2]["selected_funds"]
    assert "A" in delayed[2]["selected_funds"]
    assert _turnover_after_seed(delayed) < _turnover_after_seed(baseline)
