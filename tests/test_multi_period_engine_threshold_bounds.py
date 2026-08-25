from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd
import pytest

from trend_analysis.multi_period import engine as mp_engine


@dataclass
class MinimalConfig:
    """Minimal configuration for exercising threshold-hold weight bounds."""

    multi_period: Dict[str, Any] = field(
        default_factory=lambda: {
            "frequency": "M",
            "in_sample_len": 2,
            "out_sample_len": 1,
            "start": "2020-01",
            "end": "2020-04",
        }
    )
    data: Dict[str, Any] = field(
        default_factory=lambda: {
            "csv_path": "unused.csv",
            "allow_risk_free_fallback": True,
        }
    )
    portfolio: Dict[str, Any] = field(
        default_factory=lambda: {
            "policy": "threshold_hold",
            "cost_model": {"per_trade_bps": 0.0, "half_spread_bps": 0},
            "max_turnover": 1.0,
            "threshold_hold": {
                "target_n": 3,
                "metric": "Sharpe",
                "soft_strikes": 1,
                "entry_soft_strikes": 1,
                "z_exit_soft": -5.0,
                "z_entry_soft": -5.0,
            },
            "constraints": {
                "max_funds": 3,
                "min_weight": 0.2,
                "max_weight": 0.55,
                "min_weight_strikes": 1,
            },
            "weighting": {"name": "adaptive_bayes", "params": {}},
            "indices_list": None,
        }
    )
    vol_adjust: Dict[str, Any] = field(default_factory=lambda: {"target_vol": 1.0})
    benchmarks: Dict[str, Any] = field(default_factory=dict)
    run: Dict[str, Any] = field(default_factory=lambda: {"monthly_cost": 0.0})
    seed: int = 123

    def model_dump(self) -> Dict[str, Any]:
        return {
            "multi_period": self.multi_period,
            "portfolio": self.portfolio,
            "vol_adjust": self.vol_adjust,
        }


@dataclass
class DummyPeriod:
    in_start: str
    in_end: str
    out_start: str
    out_end: str


class ScriptedSelector:
    """Selector that keeps the provided score-frame ordering."""

    rank_column = "Sharpe"

    def select(self, score_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return score_frame, score_frame


class ScriptedWeighting:
    """Weighting sequence crafted to exercise bound adjustments."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.calls = 0
        self.sequences: List[Dict[str, float]] = [
            {"Alpha One": 0.6, "Alpha Two": 0.4, "Beta One": 0.2, "Gamma One": 0.1},
            {"Alpha One": 0.4, "Beta One": 0.1, "Gamma One": 0.05},
            {"Alpha One": 0.9, "Beta One": 0.4, "Gamma One": 0.35},
        ]

    def weight(self, selected: pd.DataFrame, date: pd.Timestamp | None = None) -> pd.DataFrame:
        del date
        seq = self.sequences[min(self.calls, len(self.sequences) - 1)]
        self.calls += 1
        weights = pd.Series(
            {idx: seq.get(idx, 0.05) for idx in selected.index},
            index=selected.index,
            dtype=float,
        )
        return weights.to_frame("weight")

    def update(self, scores: pd.Series, days: int) -> None:  # pragma: no cover - inert hook
        pass


class StaticRebalancer:
    """Rebalancer that preserves prior holdings for deterministic tests."""

    def __init__(self, *_cfg: Any) -> None:
        self.calls = 0

    def apply_triggers(self, prev_weights: pd.Series, _sf: pd.DataFrame, **kwargs) -> pd.Series:
        self.calls += 1
        return prev_weights.astype(float)


def test_threshold_hold_weight_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = MinimalConfig()

    dates = pd.to_datetime(
        [
            "2020-01-31",
            "2020-02-29",
            "2020-03-31",
            "2020-04-30",
            "2020-05-31",
        ]
    )
    df = pd.DataFrame(
        {
            "Date": dates,
            "Alpha One": [0.05, 0.04, 0.03, 0.02, 0.01],
            "Alpha Two": [0.06, 0.05, 0.04, 0.03, 0.02],
            "Beta One": [0.02, 0.03, 0.02, 0.01, 0.02],
            "Gamma One": [0.04, 0.05, 0.06, 0.07, 0.08],
        }
    )

    periods = [
        DummyPeriod("2020-01-31", "2020-03-31", "2020-04-30", "2020-04-30"),
        DummyPeriod("2020-02-29", "2020-04-30", "2020-05-31", "2020-05-31"),
    ]

    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)
    monkeypatch.setattr(mp_engine, "AdaptiveBayesWeighting", ScriptedWeighting)
    monkeypatch.setattr(mp_engine, "Rebalancer", StaticRebalancer)

    import trend_analysis.selector as selector_mod

    monkeypatch.setattr(selector_mod, "create_selector_by_name", lambda *a, **k: ScriptedSelector())

    import trend_analysis.core.rank_selection as rank_sel

    metric_maps = {
        "AnnualReturn": {
            "Alpha One": 0.15,
            "Alpha Two": 0.12,
            "Beta One": 0.07,
            "Gamma One": 0.18,
        },
        "Volatility": {
            "Alpha One": 0.25,
            "Alpha Two": 0.2,
            "Beta One": 0.15,
            "Gamma One": 0.3,
        },
        "Sharpe": {
            "Alpha One": 0.9,
            "Alpha Two": 0.8,
            "Beta One": 0.4,
            "Gamma One": 1.1,
        },
        "Sortino": {
            "Alpha One": 1.1,
            "Alpha Two": 0.9,
            "Beta One": 0.45,
            "Gamma One": 1.3,
        },
        "InformationRatio": {
            "Alpha One": 0.6,
            "Alpha Two": 0.5,
            "Beta One": 0.3,
            "Gamma One": 0.9,
        },
        "MaxDrawdown": {
            "Alpha One": -0.12,
            "Alpha Two": -0.11,
            "Beta One": -0.05,
            "Gamma One": -0.09,
        },
    }

    def fake_metric_series(
        _frame: pd.DataFrame, metric: str, _stats_cfg: Any, *, risk_free_override: object
    ) -> pd.Series:
        values = metric_maps[metric]
        return pd.Series(values, dtype=float)

    monkeypatch.setattr(rank_sel, "_compute_metric_series", fake_metric_series)

    records: List[Dict[str, Any]] = []

    def fake_run_analysis(
        _df: pd.DataFrame,
        in_start: str,
        in_end: str,
        out_start: str,
        out_end: str,
        _target_vol: float,
        _monthly_cost: float,
        *,
        custom_weights: Dict[str, float],
        manual_funds: List[str],
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        records.append(
            {
                "weights": dict(custom_weights),
                "funds": list(manual_funds),
                "period": (in_start, out_end),
            }
        )
        return {"metrics": pd.DataFrame(), "details": {}, "seed": cfg.seed}

    monkeypatch.setattr(mp_engine, "_run_analysis_with_diagnostics", fake_run_analysis)

    results = mp_engine.run(cfg, df=df)

    assert len(results) == 2

    # The scripted weighting sequences should normalise within the weight bounds.
    assert records
    first_weights = records[0]["weights"]
    assert set(first_weights) == {"Alpha One", "Beta One", "Gamma One"}
    assert pytest.approx(sum(first_weights.values()), rel=1e-9) == 100.0

    second_weights = records[1]["weights"]
    assert pytest.approx(sum(second_weights.values()), rel=1e-9) == 100.0
    assert set(records[0]["funds"]) == {"Alpha One", "Beta One", "Gamma One"}
    assert set(records[1]["funds"]) == {"Alpha One", "Beta One", "Gamma One"}


def test_threshold_hold_max_active_positions_respects_turnover_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`max_active_positions` binds; the turnover cap may only DELAY it.

    The precedence rule this test encodes, so that the next relaxation has to
    argue with a written rule rather than with a bare number:

    1. The turnover cap wins over the position-count target WITHIN a period. A
       max-active target can require more turnover than the per-period cap
       permits, and forcing an over-cap liquidation to hit the count is wrong.
    2. The position-count target wins ACROSS periods. The cap meters the exit;
       it does not cancel it. So a legacy holding may survive a bounded number
       of transitional periods, and the FINAL record must be at the target.
    3. The count may therefore fall or hold, never rise. A rising count means a
       name the trim zeroed was restored — which is how a deferral that reads as
       "one more period" silently becomes permanent.

    Rule 2 is not free: `_apply_weight_bounds` floors every active position up to
    `min_weight`, so before #6003 the post-cap bounds pass restored the trimmed
    name every period and the count sat at `max_funds` forever. `_apply_weight_bounds`
    now takes an `exiting` set that is exempt from that floor.

    Asserting only `<= max_funds` (as this test did between #5993 and #6003)
    satisfies rule 1 and asserts NOTHING about rules 2 and 3 — `max_funds` is the
    count the fixture already had, so the assertion holds no matter how long the
    trim is deferred, including forever.
    """
    cfg = MinimalConfig()
    cfg.portfolio["constraints"]["max_active_positions"] = 2

    dates = pd.to_datetime(
        [
            "2020-01-31",
            "2020-02-29",
            "2020-03-31",
            "2020-04-30",
            "2020-05-31",
        ]
    )
    df = pd.DataFrame(
        {
            "Date": dates,
            "Alpha One": [0.05, 0.04, 0.03, 0.02, 0.01],
            "Alpha Two": [0.06, 0.05, 0.04, 0.03, 0.02],
            "Beta One": [0.02, 0.03, 0.02, 0.01, 0.02],
            "Gamma One": [0.04, 0.05, 0.06, 0.07, 0.08],
        }
    )

    periods = [
        DummyPeriod("2020-01-31", "2020-03-31", "2020-04-30", "2020-04-30"),
        DummyPeriod("2020-02-29", "2020-04-30", "2020-05-31", "2020-05-31"),
    ]

    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)
    monkeypatch.setattr(mp_engine, "AdaptiveBayesWeighting", ScriptedWeighting)
    monkeypatch.setattr(mp_engine, "Rebalancer", StaticRebalancer)

    import trend_analysis.selector as selector_mod

    monkeypatch.setattr(selector_mod, "create_selector_by_name", lambda *a, **k: ScriptedSelector())

    import trend_analysis.core.rank_selection as rank_sel

    metric_maps = {
        "AnnualReturn": {
            "Alpha One": 0.15,
            "Alpha Two": 0.12,
            "Beta One": 0.07,
            "Gamma One": 0.18,
        },
        "Volatility": {
            "Alpha One": 0.25,
            "Alpha Two": 0.2,
            "Beta One": 0.15,
            "Gamma One": 0.3,
        },
        "Sharpe": {
            "Alpha One": 0.9,
            "Alpha Two": 0.8,
            "Beta One": 0.4,
            "Gamma One": 1.1,
        },
        "Sortino": {
            "Alpha One": 1.1,
            "Alpha Two": 0.9,
            "Beta One": 0.45,
            "Gamma One": 1.3,
        },
        "InformationRatio": {
            "Alpha One": 0.6,
            "Alpha Two": 0.5,
            "Beta One": 0.3,
            "Gamma One": 0.9,
        },
        "MaxDrawdown": {
            "Alpha One": -0.12,
            "Alpha Two": -0.11,
            "Beta One": -0.05,
            "Gamma One": -0.09,
        },
    }

    def fake_metric_series(
        _frame: pd.DataFrame, metric: str, _stats_cfg: Any, *, risk_free_override: object
    ) -> pd.Series:
        values = metric_maps[metric]
        return pd.Series(values, dtype=float)

    monkeypatch.setattr(rank_sel, "_compute_metric_series", fake_metric_series)

    records: List[Dict[str, Any]] = []

    def fake_run_analysis(
        _df: pd.DataFrame,
        in_start: str,
        in_end: str,
        out_start: str,
        out_end: str,
        _target_vol: float,
        _monthly_cost: float,
        *,
        custom_weights: Dict[str, float],
        manual_funds: List[str],
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        records.append(
            {
                "weights": dict(custom_weights),
                "funds": list(manual_funds),
                "period": (in_start, out_end),
            }
        )
        return {"metrics": pd.DataFrame(), "details": {}, "seed": cfg.seed}

    monkeypatch.setattr(mp_engine, "_run_analysis_with_diagnostics", fake_run_analysis)

    results = mp_engine.run(cfg, df=df)

    assert len(results) == 2
    assert records

    max_active = cfg.portfolio["constraints"]["max_active_positions"]
    counts = [len(record["funds"]) for record in records]

    # MEASURED against this fixture under its configured cap (max_turnover=1.0,
    # min_weight=0.2): the transitional rebalance needs ZERO periods -- the target
    # is already met in the first emitted record. Recorded as a constant so that
    # deferring the trim by even one period has to raise this number deliberately,
    # in a diff, with a re-measurement behind it.
    TRANSITIONAL_PERIODS = 0

    # Only the transitional PREFIX may exceed the target. Written as an equality on
    # the offending indices rather than an upper bound, so a record that exceeds it
    # later -- the "deferred forever" case -- cannot pass by being small enough.
    over_target = [index for index, count in enumerate(counts) if count > max_active]
    assert over_target == list(range(TRANSITIONAL_PERIODS)), (
        f"records {over_target} exceed max_active_positions={max_active}; only the "
        f"first {TRANSITIONAL_PERIODS} transitional record(s) may, counts={counts}"
    )

    # The FINAL record is the one that must be at the target. Permitting an
    # over-target count here is what turns "deferred by a period" into "never".
    assert len(records[-1]["funds"]) <= max_active
    assert len(records[-1]["weights"]) <= max_active

    for record in records:
        assert len(record["weights"]) == len(record["funds"])

    # A trim may be METERED by the turnover cap; it may never be REVERSED. A count
    # that rises between records means a name the trim zeroed came back, which is
    # exactly how a bounded deferral becomes an unbounded one.
    assert counts == sorted(
        counts, reverse=True
    ), f"active-position count must be monotonically non-increasing, got {counts}"
