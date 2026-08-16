"""Pin current threshold-hold behavior before the extraction tracked by #5903.

The live function has these decision families: period/history and risk-free setup;
candidate filtering; top-n/top-percent/threshold/random/buy-and-hold seeding;
hard and soft entry/exit; firm de-duplication; cooldown, sticky-period, tenure,
and minimum-fund guards; weighting and bounds; turnover/change budgets and
transaction costs; realised-holding reconciliation; and empty-period assembly.

These are characterization tests.  The literal expectations below describe the
current implementation, including behavior called out as questionable.  They do
not define the desired behavior of the later extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from trend_analysis.multi_period import engine


@dataclass
class ThresholdScenarioConfig:
    multi_period: dict[str, Any] = field(
        default_factory=lambda: {
            "frequency": "M",
            "in_sample_len": 2,
            "out_sample_len": 1,
            "start": "2020-01",
            "end": "2020-05",
        }
    )
    data: dict[str, Any] = field(
        default_factory=lambda: {
            "csv_path": "unused.csv",
            "missing_policy": "ffill",
            "risk_free_column": "RF",
        }
    )
    portfolio: dict[str, Any] = field(
        default_factory=lambda: {
            "policy": "threshold_hold",
            "transaction_cost_bps": 0.0,
            "max_turnover": 1.0,
            "selector": {"params": {"rank_column": "Sharpe"}},
            "rank": {
                "inclusion_approach": "top_n",
                "n": 2,
                "score_by": "Sharpe",
                "transform": "raw",
            },
            "threshold_hold": {
                "target_n": 2,
                "metric": "Sharpe",
                "z_entry_soft": 0.5,
                "z_exit_soft": -0.5,
                "z_exit_hard": -3.0,
                "soft_strikes": 2,
                "entry_soft_strikes": 2,
            },
            "constraints": {
                "max_funds": 2,
                "min_weight": 0.0,
                "max_weight": 1.0,
            },
            "weighting": {"name": "equal"},
            "indices_list": [],
        }
    )
    vol_adjust: dict[str, Any] = field(default_factory=lambda: {"target_vol": 1.0})
    benchmarks: dict[str, Any] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=lambda: {"monthly_cost": 0.0})
    performance: dict[str, Any] = field(default_factory=dict)
    seed: int = 17

    def model_dump(self) -> dict[str, Any]:
        return {
            "multi_period": dict(self.multi_period),
            "portfolio": dict(self.portfolio),
            "vol_adjust": dict(self.vol_adjust),
        }


def _returns_frame() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame(
        {
            "Date": dates,
            "A": [0.05, 0.04, 0.03, 0.02, 0.01, 0.02],
            "B": [0.04, 0.03, 0.02, 0.03, 0.02, 0.01],
            "C": [0.01, 0.015, 0.02, 0.018, 0.017, 0.016],
            "D": [0.03, 0.032, 0.031, 0.029, 0.028, 0.027],
            "E": [0.025, 0.026, 0.027, 0.028, 0.029, 0.03],
            "RF": [0.0] * len(dates),
        }
    )


def _periods(count: int) -> list[SimpleNamespace]:
    all_periods = [
        SimpleNamespace(
            in_start="2020-01", in_end="2020-02", out_start="2020-03", out_end="2020-03"
        ),
        SimpleNamespace(
            in_start="2020-02", in_end="2020-03", out_start="2020-04", out_end="2020-04"
        ),
        SimpleNamespace(
            in_start="2020-03", in_end="2020-04", out_start="2020-05", out_end="2020-05"
        ),
    ]
    return all_periods[:count]


def _patch_scenario(
    monkeypatch: pytest.MonkeyPatch,
    *,
    metric_by_in_end: dict[str, dict[str, float]],
    period_count: int,
    metric_by_name: dict[str, dict[str, float]] | None = None,
) -> None:
    import trend_analysis.core.rank_selection as rank_selection

    monkeypatch.setattr(engine, "generate_periods", lambda _cfg: _periods(period_count))
    monkeypatch.setattr(
        engine,
        "apply_missing_policy",
        lambda frame, *, policy, limit: (frame, {}),
    )
    monkeypatch.setattr(engine, "_run_analysis", lambda *_args, **_kwargs: {})

    def metric_series(frame: pd.DataFrame, metric: str, _cfg: Any) -> pd.Series:
        end_key = pd.Timestamp(frame.index.max()).strftime("%Y-%m")
        values = (metric_by_name or {}).get(metric, metric_by_in_end[end_key])
        return pd.Series(
            {str(column): float(values.get(str(column), 0.0)) for column in frame.columns},
            dtype=float,
        )

    monkeypatch.setattr(rank_selection, "_compute_metric_series", metric_series)


def _snapshot(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "funds": result["selected_funds"],
        "weights": result["fund_weights"],
        "turnover": result["turnover"],
        "cost": result["transaction_cost"],
        "events": [
            (event.get("action"), event.get("manager"), event.get("reason"))
            for event in result["manager_changes"]
        ],
        "tenure": result.get("holding_tenure"),
    }


def test_entry_exit_delay_and_weight_continuity_golden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = ThresholdScenarioConfig()
    cfg.portfolio["transaction_cost_bps"] = 100.0
    _patch_scenario(
        monkeypatch,
        period_count=3,
        metric_by_in_end={
            "2020-02": {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0, "E": -1.0},
            "2020-03": {"A": -1.0, "B": 2.0, "C": 1.0, "D": 3.0, "E": 0.0},
            "2020-04": {"A": -1.0, "B": 2.0, "C": 1.0, "D": 3.0, "E": 0.0},
        },
    )

    results = engine.run(cfg, df=_returns_frame())

    assert [_snapshot(result) for result in results] == [
        {
            "funds": ["A", "B"],
            "weights": {"A": 0.5, "B": 0.5},
            "turnover": 1.0,
            "cost": 0.01,
            "events": [("added", "A", "seed"), ("added", "B", "seed")],
            "tenure": {"A": 1, "B": 1},
        },
        {
            "funds": ["A", "B"],
            "weights": {"A": 0.5, "B": 0.5},
            "turnover": 0.0,
            "cost": 0.0,
            "events": [],
            "tenure": {"A": 2, "B": 2},
        },
        {
            "funds": ["B", "D"],
            "weights": {"B": 0.5, "D": 0.5},
            "turnover": 0.5,
            "cost": 0.005,
            "events": [("added", "D", "z_entry"), ("dropped", "A", "z_exit")],
            "tenure": {"B": 3, "D": 1},
        },
    ]


@pytest.mark.parametrize(
    ("max_turnover", "transaction_cost_bps", "expected"),
    [
        (1.0, 0.0, ({"B": 0.5, "D": 0.5}, 0.5, 0.0)),
        (0.2, 0.0, ({"B": 0.6, "D": 0.2}, 0.4, 0.0)),
        (1.0, 100.0, ({"B": 0.5, "D": 0.5}, 0.5, 0.005)),
        (0.2, 100.0, ({"B": 0.6, "D": 0.2}, 0.4, 0.004)),
    ],
)
def test_turnover_cap_and_transaction_cost_golden(
    monkeypatch: pytest.MonkeyPatch,
    max_turnover: float,
    transaction_cost_bps: float,
    expected: tuple[dict[str, float], float, float],
) -> None:
    cfg = ThresholdScenarioConfig()
    cfg.portfolio["max_turnover"] = max_turnover
    cfg.portfolio["transaction_cost_bps"] = transaction_cost_bps
    cfg.portfolio["threshold_hold"].update(
        {
            "z_entry_soft": 0.0,
            "z_exit_soft": -0.5,
            "z_exit_hard": -1.0,
            "soft_strikes": 1,
            "entry_soft_strikes": 1,
        }
    )
    _patch_scenario(
        monkeypatch,
        period_count=2,
        metric_by_in_end={
            "2020-02": {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0, "E": -1.0},
            "2020-03": {"A": -10.0, "B": 2.0, "C": 1.0, "D": 10.0, "E": 0.0},
        },
    )

    results = engine.run(cfg, df=_returns_frame())

    weights, turnover, cost = expected
    second = _snapshot(results[1])
    assert second["funds"] == ["B", "D"]
    assert second["weights"] == weights
    assert second["turnover"] == pytest.approx(turnover)
    assert second["cost"] == pytest.approx(cost)
    assert second["events"] == [
        ("added", "D", "z_entry"),
        ("dropped", "A", "z_exit_hard"),
    ]


def test_empty_candidate_set_golden(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = ThresholdScenarioConfig()
    cfg.portfolio["rank"]["bottom_k"] = 5
    cfg.portfolio["constraints"]["min_funds"] = 2
    _patch_scenario(
        monkeypatch,
        period_count=1,
        metric_by_in_end={"2020-02": {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0, "E": -1.0}},
    )

    results = engine.run(cfg, df=_returns_frame())

    assert _snapshot(results[0]) == {
        "funds": [],
        "weights": {},
        "turnover": 0.0,
        "cost": 0.0,
        "events": [],
        "tenure": {},
    }
    assert list(results[0]["selection_score_frame"].index) == ["A", "B", "C", "D", "E"]


def test_holdings_carry_but_weights_recompute_each_period_golden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SequencedWeighting:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[str, ...], str]] = []

        def weight(self, selected: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
            self.calls.append((tuple(selected.index), str(date.date())))
            sequences = [(0.8, 0.2), (0.7, 0.3), (0.6, 0.4)]
            values = sequences[min(len(self.calls) - 1, len(sequences) - 1)]
            return pd.DataFrame({"weight": values[: len(selected)]}, index=selected.index)

    weighting = SequencedWeighting()
    monkeypatch.setattr(engine, "EqualWeight", lambda: weighting)
    cfg = ThresholdScenarioConfig()
    _patch_scenario(
        monkeypatch,
        period_count=2,
        metric_by_in_end={
            "2020-02": {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0, "E": -1.0},
            "2020-03": {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0, "E": -1.0},
        },
    )

    results = engine.run(cfg, df=_returns_frame())

    # Seeding makes a historical weighting call, then each period recomputes.
    assert weighting.calls == [
        (("A", "B"), "2020-03-01"),
        (("A", "B"), "2020-03-01"),
        (("A", "B"), "2020-04-01"),
    ]
    assert [_snapshot(result) for result in results] == [
        {
            "funds": ["A", "B"],
            "weights": {"A": 0.7, "B": 0.3},
            "turnover": 1.0,
            "cost": 0.0,
            "events": [("added", "A", "seed"), ("added", "B", "seed")],
            "tenure": {"A": 1, "B": 1},
        },
        {
            "funds": ["A", "B"],
            "weights": {"A": 0.6, "B": 0.4},
            "turnover": 0.1,
            "cost": 0.0,
            "events": [],
            "tenure": {"A": 2, "B": 2},
        },
    ]


def test_single_fund_golden(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = ThresholdScenarioConfig()
    cfg.portfolio["rank"]["n"] = 1
    cfg.portfolio["threshold_hold"]["target_n"] = 1
    cfg.portfolio["constraints"] = {"max_funds": 1, "min_weight": 0.0, "max_weight": 1.0}
    _patch_scenario(
        monkeypatch,
        period_count=1,
        metric_by_in_end={"2020-02": {"A": 1.0}},
    )

    results = engine.run(cfg, df=_returns_frame()[["Date", "A", "RF"]])

    assert _snapshot(results[0]) == {
        "funds": ["A"],
        "weights": {"A": 1.0},
        "turnover": 1.0,
        "cost": 0.0,
        "events": [("added", "A", "seed")],
        "tenure": {"A": 1},
    }


def test_all_funds_below_threshold_produces_empty_portfolio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = ThresholdScenarioConfig()
    cfg.portfolio["rank"].update(
        {
            "inclusion_approach": "threshold",
            "threshold": 100.0,
            "score_by": "Sharpe",
            "transform": "raw",
        }
    )
    _patch_scenario(
        monkeypatch,
        period_count=1,
        metric_by_in_end={"2020-02": {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0, "E": -1.0}},
    )

    results = engine.run(cfg, df=_returns_frame())

    assert _snapshot(results[0]) == {
        "funds": [],
        "weights": {},
        "turnover": 0.0,
        "cost": 0.0,
        "events": [],
        "tenure": {},
    }
    assert results[0]["selection_shortfall"] == {
        "reason": "threshold_hard_gate",
        "threshold": 100.0,
        "eligible_funds": 0,
        "selected_funds": 0,
        "target_funds": 2,
        "min_funds": 0,
    }


def test_threshold_qualifiers_are_not_backfilled_to_minimum_funds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = ThresholdScenarioConfig()
    cfg.portfolio["threshold_hold"].pop("target_n")
    cfg.portfolio["rank"].update(
        {
            "inclusion_approach": "threshold",
            "threshold": 2.0,
            "score_by": "Sharpe",
            "transform": "raw",
        }
    )
    cfg.portfolio["rank"].pop("n")
    cfg.portfolio["constraints"]["max_funds"] = 5
    cfg.portfolio["constraints"]["min_funds"] = 5
    _patch_scenario(
        monkeypatch,
        period_count=1,
        metric_by_in_end={"2020-02": {"A": 2.0, "B": 2.0, "C": 2.0, "D": 1.0, "E": 0.0}},
    )

    results = engine.run(cfg, df=_returns_frame())

    assert _snapshot(results[0]) == {
        "funds": ["A", "B", "C"],
        "weights": {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3},
        "turnover": 1.0,
        "cost": 0.0,
        "events": [
            ("added", "A", "seed"),
            ("added", "B", "seed"),
            ("added", "C", "seed"),
        ],
        "tenure": {"A": 1, "B": 1, "C": 1},
    }
    assert results[0]["selection_shortfall"] == {
        "reason": "threshold_hard_gate",
        "threshold": 2.0,
        "eligible_funds": 3,
        "selected_funds": 3,
        "target_funds": None,
        "min_funds": 5,
    }


@pytest.mark.parametrize(
    ("selection_mode", "extra_portfolio", "expected_funds"),
    [
        (
            "buy_and_hold",
            {"buy_and_hold": {"initial_method": "threshold", "n": 2, "threshold": 0.5}},
            ["A", "B"],
        ),
        ("random", {"random_n": 2}, ["B", "E"]),
    ],
)
def test_alternate_seed_paths_golden(
    monkeypatch: pytest.MonkeyPatch,
    selection_mode: str,
    extra_portfolio: dict[str, Any],
    expected_funds: list[str],
) -> None:
    cfg = ThresholdScenarioConfig()
    cfg.portfolio["selection_mode"] = selection_mode
    cfg.portfolio.update(extra_portfolio)
    _patch_scenario(
        monkeypatch,
        period_count=1,
        metric_by_in_end={"2020-02": {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0, "E": -1.0}},
    )

    results = engine.run(cfg, df=_returns_frame())

    assert _snapshot(results[0]) == {
        "funds": expected_funds,
        "weights": {fund: 0.5 for fund in expected_funds},
        "turnover": 1.0,
        "cost": 0.0,
        "events": [("added", fund, "seed") for fund in expected_funds],
        "tenure": {fund: 1 for fund in expected_funds},
    }


def test_blended_seed_uses_metric_specific_scores_golden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = ThresholdScenarioConfig()
    cfg.portfolio["rank"].update(
        {
            "inclusion_approach": "top_pct",
            "pct": 0.4,
            "score_by": "blended",
            "blended_weights": {"Sharpe": 0.4, "MaxDrawdown": 0.6},
            "transform": "raw",
        }
    )
    fallback = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}
    _patch_scenario(
        monkeypatch,
        period_count=1,
        metric_by_in_end={"2020-02": fallback},
        metric_by_name={
            "Sharpe": {"A": 5.0, "B": 1.0, "C": 4.0, "D": 2.0, "E": 3.0},
            "MaxDrawdown": {"A": 0.0, "B": -5.0, "C": -4.0, "D": -1.0, "E": -2.0},
        },
    )

    results = engine.run(cfg, df=_returns_frame())

    assert _snapshot(results[0]) == {
        "funds": ["C", "B"],
        "weights": {"C": 0.5, "B": 0.5},
        "turnover": 1.0,
        "cost": 0.0,
        "events": [("added", "C", "seed"), ("added", "B", "seed")],
        "tenure": {"C": 1, "B": 1},
    }


def test_zscore_threshold_differs_from_raw_ascending_selection_golden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = ThresholdScenarioConfig()
    cfg.portfolio["rank"].update(
        {
            "inclusion_approach": "threshold",
            "threshold": 0.0,
            "score_by": "MaxDrawdown",
            "transform": "zscore",
        }
    )
    fallback = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}
    _patch_scenario(
        monkeypatch,
        period_count=1,
        metric_by_in_end={"2020-02": fallback},
        metric_by_name={
            "Sharpe": {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0, "E": -1.0},
            "MaxDrawdown": {"A": 0.0, "B": -5.0, "C": -4.0, "D": -1.0, "E": -2.0},
        },
    )

    results = engine.run(cfg, df=_returns_frame())

    # CHARACTERIZATION: z-score transformation currently treats the larger
    # standardized MaxDrawdown as better; raw ascending selection would choose B/C.
    assert _snapshot(results[0]) == {
        "funds": ["A", "D"],
        "weights": {"A": 0.5, "D": 0.5},
        "turnover": 1.0,
        "cost": 0.0,
        "events": [("added", "A", "seed"), ("added", "D", "seed")],
        "tenure": {"A": 1, "D": 1},
    }


def test_top_percent_seed_golden(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = ThresholdScenarioConfig()
    cfg.portfolio["rank"].update(
        {
            "inclusion_approach": "top_pct",
            "pct": 0.4,
            "score_by": "Sharpe",
            "transform": "raw",
        }
    )
    cfg.portfolio["threshold_hold"]["target_n"] = 4
    cfg.portfolio["constraints"]["max_funds"] = 5
    _patch_scenario(
        monkeypatch,
        period_count=1,
        metric_by_in_end={"2020-02": {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0, "E": -1.0}},
    )

    results = engine.run(cfg, df=_returns_frame())

    assert _snapshot(results[0]) == {
        "funds": ["A", "B"],
        "weights": {"A": 0.5, "B": 0.5},
        "turnover": 1.0,
        "cost": 0.0,
        "events": [("added", "A", "seed"), ("added", "B", "seed")],
        "tenure": {"A": 1, "B": 1},
    }
