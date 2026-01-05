import types
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
import pytest

from trend_analysis.multi_period import engine


class DummySelector:
    """Minimal selector returning the provided score frame."""

    rank_column = "rank"

    def __init__(self, order: Iterable[str] | None = None) -> None:
        self._order = list(order) if order is not None else None

    def select(self, score_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._order is not None:
            score_frame = score_frame.loc[self._order]
        return score_frame, score_frame


class DummyWeighting(engine.BaseWeighting):
    """Weighting implementation that produces deterministic equal weights."""

    def __init__(self) -> None:
        self.updates: list[tuple[pd.Series, int]] = []

    def weight(self, df: pd.DataFrame, date: pd.Timestamp | None = None) -> pd.DataFrame:
        del date
        if df.empty:
            return pd.DataFrame({"weight": pd.Series(dtype=float)})
        w = np.full(len(df.index), 1.0 / len(df.index))
        return pd.DataFrame({"weight": w}, index=df.index)

    def update(self, scores: pd.Series, days: int) -> None:  # pragma: no cover - thin
        self.updates.append((scores.copy(), days))


class DummyConfig:
    """Small configuration stub satisfying the engine's interface."""

    def __init__(self, policy: str = "") -> None:
        self.data: Dict[str, Any] = {"csv_path": "unused.csv"}
        self.portfolio: Dict[str, Any] = {
            "policy": policy,
            "rank": {},
            "random_n": 3,
            "custom_weights": None,
        }
        self.vol_adjust: Dict[str, Any] = {"target_vol": 1.0}
        self.run: Dict[str, Any] = {"monthly_cost": 0.0}
        self.performance: Dict[str, Any] = {}
        self.benchmarks: Dict[str, Any] = {}
        self.multi_period: Dict[str, Any] = {
            "frequency": "monthly",
            "in_sample_len": 1,
            "out_sample_len": 1,
            "start": "2020-01-31",
            "end": "2020-02-29",
        }
        self.seed = 7

    def model_dump(self) -> Dict[str, Any]:  # pragma: no cover - simple helper
        return {"multi_period": dict(self.multi_period)}


def _make_price_frame(date: str, values: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({"Date": [pd.Timestamp(date)], **values})


def test_compute_turnover_state_tracks_union_alignment() -> None:
    prev_idx = np.array(["FundA", "FundB"], dtype=object)
    prev_vals = np.array([0.6, 0.4], dtype=float)
    new_weights = pd.Series({"FundA": 0.2, "FundC": 0.8})

    turnover, next_idx, next_vals = engine._compute_turnover_state(prev_idx, prev_vals, new_weights)

    union = pd.Index(["FundA", "FundC", "FundB"])
    expected_turnover = float(
        np.abs(
            new_weights.reindex(union, fill_value=0.0)
            - pd.Series(prev_vals, index=prev_idx).reindex(union, fill_value=0.0)
        ).sum()
    )

    assert turnover == pytest.approx(expected_turnover)
    assert list(next_idx) == ["FundA", "FundC"]
    assert list(next_vals) == [0.2, 0.8]


def test_run_schedule_applies_rebalance_strategies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = ["2021-01-31", "2021-02-28"]
    frames = {
        d: pd.DataFrame(
            {
                "rank": [1.0, 2.0],
                "metric": [0.4, 0.2],
            },
            index=["FundA", "FundB"],
        )
        for d in dates
    }

    selector = DummySelector()
    weighting = DummyWeighting()

    captured: list[Dict[str, Any]] = []

    def fake_apply(strategies, params, current, target, *, scores=None):
        captured.append(
            {
                "strategies": strategies,
                "params": params,
                "current": current.copy(),
                "target": target.copy(),
                "scores": scores.copy() if scores is not None else None,
            }
        )
        return (target * 0 + 0.5, 1.23)

    monkeypatch.setattr(engine, "apply_rebalancing_strategies", fake_apply)

    portfolio = engine.run_schedule(
        frames,
        selector,
        weighting,
        rank_column="rank",
        rebalance_strategies=["threshold"],
        rebalance_params={"threshold": {"param": 5}},
    )

    assert len(portfolio.history) == len(dates)
    assert all(isinstance(v, pd.Series) for v in portfolio.history.values())
    assert [call["scores"].tolist() for call in captured] == [[1.0, 2.0], [1.0, 2.0]]
    # Weighting.update should be invoked once per period
    assert [days for _, days in weighting.updates] == [0, 28]


def test_portfolio_rebalance_accepts_series() -> None:
    """The Portfolio helper should store series inputs without re-wrapping."""

    pf = engine.Portfolio()
    weights = pd.Series({"FundA": 0.6, "FundB": 0.4}, dtype=float)

    pf.rebalance("2021-03-31", weights, turnover=0.12, cost=0.005)

    assert set(pf.history) == {"2021-03-31"}
    stored = pf.history["2021-03-31"]
    assert isinstance(stored, pd.Series)
    assert stored.to_dict() == {"FundA": 0.6, "FundB": 0.4}
    assert pf.turnover["2021-03-31"] == pytest.approx(0.12)
    assert pf.costs["2021-03-31"] == pytest.approx(0.005)
    assert pf.total_rebalance_costs == pytest.approx(0.005)


def test_run_price_frames_validation_errors() -> None:
    cfg = DummyConfig()
    df = pd.DataFrame({"Date": [pd.Timestamp("2020-01-31")], "Fund": [0.1]})

    with pytest.raises(TypeError):
        engine.run(cfg, df=df, price_frames={"2020-01": "not a frame"})

    with pytest.raises(ValueError):
        engine.run(
            cfg,
            df=df,
            price_frames={"2020-01": pd.DataFrame({"Other": [1]})},
        )

    with pytest.raises(ValueError):
        engine.run(cfg, df=df, price_frames={})


def test_run_combines_price_frames_and_calls_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = DummyConfig()

    frames = {
        "2020-01": _make_price_frame("2020-01-31", {"FundX": 0.1}),
        "2020-02": _make_price_frame("2020-02-29", {"FundX": 0.2}),
    }

    periods = [
        types.SimpleNamespace(
            in_start="2020-01-31",
            in_end="2020-01-31",
            out_start="2020-02-29",
            out_end="2020-02-29",
        )
    ]

    monkeypatch.setattr(engine, "generate_periods", lambda *_: periods)

    captured: list[pd.DataFrame] = []

    def fake_run_analysis(df, *_args, **_kwargs):
        captured.append(df.copy())
        return {"out_ew_stats": {}, "out_user_stats": {}}

    monkeypatch.setattr(engine, "_run_analysis", fake_run_analysis)

    results = engine.run(cfg, df=None, price_frames=frames)

    assert captured, "_run_analysis should be called"
    combined_df = captured[0]
    assert combined_df["Date"].tolist() == [
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-29"),
    ]
    assert len(results) == 1
    assert results[0]["period"] == (
        "2020-01-31",
        "2020-01-31",
        "2020-02-29",
        "2020-02-29",
    )
