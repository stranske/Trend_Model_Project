from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trend_analysis.multi_period import engine


def test_portfolio_rebalance_dataframe_weight_column():
    portfolio = engine.Portfolio()
    weights = pd.DataFrame({"weight": [0.3, 0.7]}, index=["FundA", "FundB"])

    portfolio.rebalance("2024-01-31", weights, turnover=0.15, cost=0.02)

    key = "2024-01-31"
    assert key in portfolio.history
    assert np.isclose(portfolio.turnover[key], 0.15)
    assert np.isclose(portfolio.costs[key], 0.02)
    assert np.isclose(portfolio.history[key].sum(), 1.0)
    assert np.isclose(portfolio.total_rebalance_costs, 0.02)


def test_compute_turnover_state_aligns_weights():
    first = pd.Series([0.6, 0.4], index=["FundA", "FundB"], dtype=float)
    turnover, prev_idx, prev_vals = engine._compute_turnover_state(None, None, first)

    assert np.isclose(turnover, np.abs(first).sum())
    assert list(prev_idx) == ["FundA", "FundB"]

    second = pd.Series([0.2, 0.8], index=["FundB", "FundC"], dtype=float)
    turnover2, _, _ = engine._compute_turnover_state(prev_idx, prev_vals, second)

    aligned_prev = first.reindex(["FundB", "FundC", "FundA"], fill_value=0.0)
    aligned_new = second.reindex(["FundB", "FundC", "FundA"], fill_value=0.0)
    expected = np.abs(aligned_new - aligned_prev).sum()
    assert np.isclose(turnover2, expected)


class DummySelector:
    column = "score"

    def __init__(self) -> None:
        self.calls: list[pd.DataFrame] = []

    def select(self, score_frame: pd.DataFrame):
        self.calls.append(score_frame)
        selected = score_frame.copy()
        selected["weight"] = np.linspace(0.6, 0.4, len(score_frame))
        return selected, score_frame


class DummyWeighting:
    def __init__(self) -> None:
        self.update_calls: list[tuple[pd.Series, int]] = []

    def weight(
        self, selected: pd.DataFrame, date: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        del date
        weights = selected[["weight"]].astype(float)
        weights["weight"] = weights["weight"].to_numpy() / weights["weight"].sum()
        return weights

    def update(self, scores: pd.Series, days: int) -> None:
        self.update_calls.append((scores, days))


class DummyRebalancer:
    def __init__(self) -> None:
        self.calls: list[tuple[pd.Series, list[str]]] = []

    def apply_triggers(
        self, prev_weights: pd.Series, score_frame: pd.DataFrame, **kwargs
    ) -> pd.Series:
        self.calls.append((prev_weights.copy(), score_frame.index.tolist()))
        return prev_weights.sort_index()


def _make_score_frames() -> dict[str, pd.DataFrame]:
    dates = ["2024-01-31", "2024-02-29"]
    frames = {}
    for idx, date in enumerate(dates):
        frames[date] = pd.DataFrame(
            {
                "score": [1.0 + idx, 0.5 + idx],
            },
            index=["FundA", "FundB"],
        )
    return frames


def test_run_schedule_with_strategies_and_rebalancer(monkeypatch):
    selector = DummySelector()
    weighting = DummyWeighting()
    rebalancer = DummyRebalancer()

    def fake_apply(
        strategies,
        params,
        current_weights,
        target_weights,
        *,
        scores=None,
        cash_policy=None,
    ):
        del strategies, params, current_weights, scores, cash_policy
        normalised = target_weights / target_weights.sum()
        return normalised, 0.25

    monkeypatch.setattr(engine, "apply_rebalancing_strategies", fake_apply)

    pf = engine.run_schedule(
        _make_score_frames(),
        selector,
        weighting,
        rank_column="score",
        rebalancer=rebalancer,
        rebalance_strategies=["demo"],
        rebalance_params={"demo": {}},
    )

    assert set(pf.history.keys()) == {"2024-01-31", "2024-02-29"}
    assert np.isclose(pf.total_rebalance_costs, 0.5)
    # update called twice with appropriate horizons (0 days on first iteration)
    days = [call[1] for call in weighting.update_calls]
    assert days[0] == 0 and days[1] > 0
    assert rebalancer.calls  # rebalancer triggered


def test_run_schedule_without_strategies_uses_turnover_state():
    selector = DummySelector()
    weighting = DummyWeighting()

    pf = engine.run_schedule(_make_score_frames(), selector, weighting)

    first = pf.history["2024-01-31"]
    second = pf.history["2024-02-29"]
    aligned = second.reindex(first.index, fill_value=0.0)
    expected_turnover = np.abs(aligned - first).sum()
    assert np.isclose(pf.turnover["2024-02-29"], expected_turnover)
    assert np.isclose(pf.total_rebalance_costs, 0.0)


class DummyCfg:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.portfolio: dict[str, object] = {
            "policy": "",
            "selection_mode": "all",
            "random_n": 2,
            "rank": {},
            "custom_weights": None,
            "manual_list": None,
            "indices_list": None,
        }
        self.vol_adjust: dict[str, float] = {"target_vol": 1.0}
        self.run: dict[str, float] = {"monthly_cost": 0.0}
        self.benchmarks: dict[str, str] = {}
        self.performance: dict[str, object] = {
            "enable_cache": True,
            "incremental_cov": False,
        }
        self.seed = 123

    def model_dump(self) -> dict[str, object]:
        return {}


def test_run_price_frames_validation_errors():
    cfg = DummyCfg()

    with pytest.raises(TypeError):
        engine.run(cfg, price_frames=[])  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        engine.run(cfg, price_frames={"2024-01": pd.Series([1, 2, 3])})

    with pytest.raises(ValueError):
        frame = pd.DataFrame({"Foo": [1, 2, 3]})
        engine.run(cfg, price_frames={"2024-01": frame})

    with pytest.raises(ValueError):
        engine.run(cfg, price_frames={})


def test_run_requires_csv_path_when_df_missing():
    cfg = DummyCfg()
    cfg.performance = {}

    with pytest.raises(KeyError):
        engine.run(cfg)


def test_run_with_price_frames_builds_covariance(monkeypatch):
    cfg = DummyCfg()

    dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
    price_frame = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.01, 0.02, 0.015],
            "FundB": [0.005, 0.01, 0.0],
        }
    )

    price_frames = {"2020-01": price_frame}

    period = SimpleNamespace(
        in_start="2020-01",
        in_end="2020-02",
        out_start="2020-03",
        out_end="2020-03",
    )

    monkeypatch.setattr(engine, "generate_periods", lambda _: [period])

    def fake_run_analysis(*args, **kwargs):
        return {"out_ew_stats": {"sharpe": 1.0}, "out_user_stats": {"sharpe": 0.9}}

    monkeypatch.setattr(engine, "_run_analysis", fake_run_analysis)

    results = engine.run(cfg, price_frames=price_frames)

    assert len(results) == 1
    result = results[0]
    assert result["period"] == (
        "2020-01",
        "2020-02",
        "2020-03",
        "2020-03",
    )
    assert "cov_diag" in result
    assert len(result["cov_diag"]) == 2
    assert "cache_stats" in result
