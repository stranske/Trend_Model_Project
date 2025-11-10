"""Additional branch coverage for ``trend_analysis.multi_period.engine``."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from trend_analysis.multi_period import engine as mp_engine


class DebugSelector:
    """Selector that returns the incoming score frame unchanged."""

    rank_column = "Sharpe"

    def select(self, score_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return score_frame, score_frame


class CyclingWeighting:
    """Weighting stub that cycles deterministic weight patterns."""

    def __init__(self) -> None:
        self._calls = 0
        self._patterns = [
            {"FundA": 0.6, "FundB": 0.4},
            {"FundB": 0.55, "FundC": 0.45},
        ]

    def weight(
        self, selected: pd.DataFrame, date: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        del date
        pattern = self._patterns[min(self._calls, len(self._patterns) - 1)]
        self._calls += 1
        weights = pd.Series(
            {idx: pattern.get(idx, 0.25) for idx in selected.index},
            index=selected.index,
            dtype=float,
        )
        weights /= weights.sum()
        return weights.to_frame("weight")

    def update(self, scores: pd.Series, days: int) -> None:  # pragma: no cover - hook
        assert days >= 0
        assert not scores.empty


class DummyConfig(SimpleNamespace):
    """Minimal configuration object exposing ``model_dump`` and dict-like attrs."""

    def model_dump(self) -> dict[str, object]:
        return {
            "multi_period": self.multi_period,
            "portfolio": self.portfolio,
            "vol_adjust": self.vol_adjust,
        }


def _base_config() -> DummyConfig:
    return DummyConfig(
        multi_period={
            "frequency": "M",
            "in_sample_len": 1,
            "out_sample_len": 1,
            "start": "2020-01",
            "end": "2020-02",
        },
        data={"csv_path": "unused.csv"},
        portfolio={
            "policy": "threshold_hold",
            "transaction_cost_bps": 0.0,
            "max_turnover": 1.0,
            "threshold_hold": {
                "target_n": 1,
                "metric": "Sharpe",
                "soft_strikes": 1,
                "entry_soft_strikes": 1,
                "min_weight": 0.2,
                "max_weight": 0.6,
                "min_weight_strikes": 2,
            },
            "constraints": {
                "max_funds": 2,
                "min_weight": 0.2,
                "max_weight": 0.6,
                "min_weight_strikes": 2,
            },
            "weighting": {"name": "adaptive_bayes", "params": {}},
            "indices_list": None,
        },
        vol_adjust={"target_vol": 1.0, "window": 12},
        benchmarks={},
        run={"monthly_cost": 0.0},
        seed=123,
    )


def test_run_schedule_debug_turnover_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabling the debug env flag should execute the turnover validation block."""

    monkeypatch.setattr(
        mp_engine.os,
        "getenv",
        lambda key: "1" if key == "DEBUG_TURNOVER_VALIDATE" else None,
    )
    calls: list[tuple[object, object]] = []

    original_isclose = mp_engine.np.isclose

    def recording_isclose(a: object, b: object, *, rtol: float, atol: float) -> bool:
        calls.append((a, b))
        return bool(original_isclose(a, b, rtol=rtol, atol=atol))

    monkeypatch.setattr(mp_engine.np, "isclose", recording_isclose)
    score_frames = {
        "2020-01-31": pd.DataFrame({"Sharpe": [1.0, 0.5]}, index=["FundA", "FundB"]),
        "2020-02-29": pd.DataFrame({"Sharpe": [0.7, 1.2]}, index=["FundB", "FundC"]),
    }

    portfolio = mp_engine.run_schedule(
        score_frames,
        selector=DebugSelector(),
        weighting=CyclingWeighting(),
        rank_column="Sharpe",
        rebalancer=None,
        rebalance_strategies=None,
        rebalance_params=None,
    )

    assert set(portfolio.history) == {"2020-01-31", "2020-02-29"}
    assert portfolio.turnover["2020-01-31"] >= 0.0
    assert portfolio.turnover["2020-02-29"] >= 0.0
    assert calls, "debug turnover validator should compare expected vs actual"

    monkeypatch.setattr(mp_engine.np, "isclose", original_isclose)


def test_run_rejects_empty_price_frames() -> None:
    """Passing an empty price frame mapping should raise a ValueError."""

    cfg = _base_config()
    with pytest.raises(ValueError, match="price_frames is empty"):
        mp_engine.run(cfg, df=None, price_frames={})


def test_run_requires_csv_path_when_dataframe_missing() -> None:
    """When no DataFrame is supplied the CSV path must be configured."""

    cfg = _base_config()
    cfg.data = {}
    with pytest.raises(KeyError, match=r"cfg.data\['csv_path']"):
        mp_engine.run(cfg, df=None, price_frames=None)


def test_run_requires_date_column() -> None:
    """A DataFrame without a Date column should be rejected."""

    cfg = _base_config()
    df = pd.DataFrame({"Alpha": [0.01, 0.02]})

    with pytest.raises(ValueError, match="must contain a 'Date' column"):
        mp_engine.run(cfg, df=df, price_frames=None)


def test_run_missing_policy_rejects_empty_cleaned_frame() -> None:
    """If the missing-data policy yields no assets the engine must error."""

    cfg = _base_config()
    dates = pd.date_range("2020-01-31", periods=3, freq="M")
    df = pd.DataFrame({"Date": dates, "Alpha": [float("nan")] * 3})

    with pytest.raises(ValueError, match="Missing-data policy removed all assets"):
        mp_engine.run(cfg, df=df, price_frames=None)


def test_run_combines_price_frames_and_returns_period_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Providing price frames should allow the non-threshold pipeline path to execute."""

    cfg = _base_config()
    cfg.portfolio["policy"] = "all"
    cfg.portfolio.setdefault("random_n", 2)
    cfg.portfolio.setdefault("custom_weights", None)
    cfg.portfolio.setdefault("rank", None)
    cfg.portfolio.setdefault("manual_list", None)
    cfg.portfolio.setdefault("indices_list", None)
    cfg.portfolio.setdefault("max_turnover", 1.0)
    cfg.performance = {"enable_cache": False, "incremental_cov": False}

    price_frames = {
        "2020-01": pd.DataFrame(
            {
                "Date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
                "Alpha": [0.01, 0.02],
            }
        ),
        "2020-03": pd.DataFrame(
            {
                "Date": pd.to_datetime(["2020-03-31", "2020-04-30"]),
                "Alpha": [0.03, 0.04],
            }
        ),
    }

    periods = [
        SimpleNamespace(
            in_start="2020-01-31",
            in_end="2020-02-29",
            out_start="2020-03-31",
            out_end="2020-04-30",
        )
    ]

    monkeypatch.setattr(mp_engine, "generate_periods", lambda *_args: periods)
    monkeypatch.setattr(mp_engine, "_run_analysis", lambda *a, **k: {"summary": "ok"})

    results = mp_engine.run(cfg, df=None, price_frames=price_frames)

    assert results
    assert results[0]["period"] == (
        "2020-01-31",
        "2020-02-29",
        "2020-03-31",
        "2020-04-30",
    )
