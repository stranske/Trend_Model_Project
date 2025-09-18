from pathlib import Path

import pandas as pd
import pytest
import yaml  # type: ignore[import-untyped]

from trend_analysis.config import Config
from trend_analysis.multi_period import Portfolio
from trend_analysis.multi_period import engine as mp_engine
from trend_analysis.multi_period import run as run_mp
from trend_analysis.multi_period import run_schedule
from trend_analysis.multi_period.replacer import Rebalancer
from trend_analysis.multi_period.scheduler import generate_periods
from trend_analysis.selector import RankSelector
from trend_analysis.weighting import AdaptiveBayesWeighting, BaseWeighting, EqualWeight


def make_df():
    dates = pd.date_range("1990-01-31", periods=12, freq="ME")
    return pd.DataFrame(
        {
            "Date": dates,
            "A": 0.01,
            "B": 0.02,
        }
    )


def test_multi_period_run_returns_results():
    cfg_data = yaml.safe_load(Path("config/defaults.yml").read_text())
    cfg_data["multi_period"] = {
        "frequency": "M",
        "in_sample_len": 2,
        "out_sample_len": 1,
        "start": "1990-01",
        "end": "1990-12",
    }
    cfg = Config(**cfg_data)
    df = make_df()
    out = run_mp(cfg, df)
    assert isinstance(out, list)
    assert out, "no period results returned"
    assert len(out) > 1, "Multi-period run produced only a single period"


def test_run_schedule_generates_weight_history():
    sf = pd.read_csv(Path("tests/fixtures/score_frame_2025-06-30.csv"), index_col=0)
    frames = {"2025-06-30": sf, "2025-07-31": sf}
    selector = RankSelector(top_n=1, rank_column="Sharpe")
    weighting = EqualWeight()
    portfolio = run_schedule(frames, selector, weighting, rank_column="Sharpe")
    assert isinstance(portfolio, Portfolio)
    assert list(portfolio.history.keys()) == ["2025-06-30", "2025-07-31"]
    assert list(portfolio.history["2025-06-30"].index) == ["A"]


def test_run_schedule_updates_weighting():
    sf = pd.read_csv(Path("tests/fixtures/score_frame_2025-06-30.csv"), index_col=0)
    frames = {"2025-06-30": sf, "2025-07-31": sf}
    selector = RankSelector(top_n=2, rank_column="Sharpe")
    weighting = AdaptiveBayesWeighting(max_w=None)
    portfolio = run_schedule(frames, selector, weighting, rank_column="Sharpe")
    w1 = portfolio.history["2025-06-30"]
    w2 = portfolio.history["2025-07-31"]
    assert w2.loc["A"] > w1.loc["A"]


def test_run_with_price_frames_none():
    """Test run function with price_frames=None (default behavior)."""
    cfg_data = yaml.safe_load(Path("config/defaults.yml").read_text())
    cfg_data["multi_period"] = {
        "frequency": "M",
        "in_sample_len": 2,
        "out_sample_len": 1,
        "start": "1990-01",
        "end": "1990-12",
    }
    cfg = Config(**cfg_data)
    df = make_df()
    # Test with price_frames=None (default)
    out = run_mp(cfg, df, price_frames=None)
    assert isinstance(out, list)
    assert len(out) > 1, "Multi-period run produced only a single period"


def test_run_with_price_frames_provided():
    """Test run function with price_frames provided."""
    cfg_data = yaml.safe_load(Path("config/defaults.yml").read_text())
    cfg_data["multi_period"] = {
        "frequency": "M",
        "in_sample_len": 2,
        "out_sample_len": 1,
        "start": "1990-01",
        "end": "1990-12",
    }
    cfg = Config(**cfg_data)

    # Create sample price_frames with returns data (similar to make_df format)
    dates = pd.date_range("1990-01-31", periods=6, freq="ME")
    price_frames = {
        "1990-01": pd.DataFrame(
            {
                "Date": dates[:3],
                "A": [0.01, 0.02, 0.01],  # Returns, not prices
                "B": [0.02, 0.01, 0.03],
            }
        ),
        "1990-02": pd.DataFrame(
            {
                "Date": dates[3:6],
                "A": [0.015, 0.025, 0.01],
                "B": [0.02, 0.015, 0.025],
            }
        ),
    }

    out = run_mp(cfg, df=None, price_frames=price_frames)
    assert isinstance(out, list)
    assert len(out) > 0, "Multi-period run with price_frames produced no results"


def test_run_with_invalid_price_frames():
    """Test run function with invalid price_frames raises appropriate
    errors."""
    cfg_data = yaml.safe_load(Path("config/defaults.yml").read_text())
    cfg_data["multi_period"] = {
        "frequency": "M",
        "in_sample_len": 2,
        "out_sample_len": 1,
        "start": "1990-01",
        "end": "1990-12",
    }
    cfg = Config(**cfg_data)

    # Test with non-dict price_frames
    try:
        run_mp(cfg, df=None, price_frames="invalid")
        assert False, "Should have raised TypeError for non-dict price_frames"
    except TypeError as e:
        assert "price_frames must be a dict" in str(e)

    # Test with non-DataFrame values
    try:
        run_mp(cfg, df=None, price_frames={"2020-01": "not_a_dataframe"})
        assert False, "Should have raised TypeError for non-DataFrame values"
    except TypeError as e:
        assert "must be a pandas DataFrame" in str(e)

    # Test with missing required columns
    try:
        price_frames = {"2020-01": pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})}
        run_mp(cfg, df=None, price_frames=price_frames)
        assert False, "Should have raised ValueError for missing Date column"
    except ValueError as e:
        assert "missing required columns" in str(e)
        assert "Date" in str(e)

    # Test with empty price_frames
    try:
        run_mp(cfg, df=None, price_frames={})
        assert False, "Should have raised ValueError for empty price_frames"
    except ValueError as e:
        assert "price_frames is empty" in str(e)


def test_generate_periods_respects_boundaries():
    # Use relative dates for maintainability
    start = (pd.Timestamp.today() - pd.offsets.MonthBegin(5)).strftime("%Y-%m")
    end = pd.Timestamp.today().strftime("%Y-%m")
    cfg = {
        "multi_period": {
            "frequency": "M",
            "in_sample_len": 2,
            "out_sample_len": 1,
            "start": start,
            "end": end,
        }
    }
    periods = generate_periods(cfg)
    # The number of periods depends on the date range and window sizes
    total = len(pd.period_range(start, end, freq="M"))
    in_len = cfg["multi_period"]["in_sample_len"]
    out_len = cfg["multi_period"]["out_sample_len"]
    expected_periods = ((total - (in_len + out_len)) // out_len) + 1
    assert len(periods) == expected_periods
    prev_start = None
    for pt in periods:
        in_start = pd.to_datetime(pt.in_start)
        in_end = pd.to_datetime(pt.in_end)
        out_start = pd.to_datetime(pt.out_start)
        out_end = pd.to_datetime(pt.out_end)
        # window lengths
        in_len = len(pd.period_range(in_start, in_end, freq="M"))
        out_len = len(pd.period_range(out_start, out_end, freq="M"))
        assert in_len == 2
        assert out_len == 1
        # out-of-sample begins after in-sample ends
        assert in_end < out_start
        if prev_start is not None:
            # next period starts exactly one month after previous start
            assert in_start == prev_start + pd.offsets.MonthBegin(1)
        prev_start = in_start


def test_run_schedule_with_rebalancer_replaces_funds():
    sf1 = pd.DataFrame({"zscore": [2.0, 1.5, -0.5]}, index=["A", "B", "C"])
    sf2 = pd.DataFrame({"zscore": [-1.5, 0.5, 2.0]}, index=["A", "B", "C"])
    # Use a fixed reference date for deterministic tests
    end_of_month = pd.Timestamp("2023-01-31")
    prev_month_end = end_of_month - pd.offsets.MonthEnd(1)
    frames = {prev_month_end: sf1, end_of_month: sf2}
    selector = RankSelector(top_n=2, rank_column="zscore")
    weighting = EqualWeight()
    cfg = {
        "portfolio": {
            "threshold_hold": {"soft_strikes": 1, "entry_soft_strikes": 1},
            "constraints": {"max_funds": 2},
        }
    }
    reb = Rebalancer(cfg)
    pf = run_schedule(
        frames,
        selector,
        weighting,
        rank_column="zscore",
        rebalancer=reb,
    )
    w1 = pf.history[str(prev_month_end.date())]
    w2 = pf.history[str(end_of_month.date())]
    assert set(w1.index) == {"A", "B"}
    assert set(w2.index) == {"B", "C"}


def test_portfolio_rebalance_accepts_mapping():
    pf = Portfolio()
    pf.rebalance(
        "2024-01-31", {"Fund A": 0.25, "Fund B": 0.75}, turnover=0.1, cost=0.02
    )
    key = "2024-01-31"
    assert set(pf.history[key].index) == {"Fund A", "Fund B"}
    assert pf.history[key].dtype == float
    assert pf.turnover[key] == pytest.approx(0.1)
    assert pf.costs[key] == pytest.approx(0.02)
    assert pf.total_rebalance_costs == pytest.approx(0.02)


def test_run_schedule_calls_rebalance_strategies_and_updates(monkeypatch):
    dates = [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")]
    sf1 = pd.DataFrame(
        {"Sharpe": [1.0, 0.5], "weight": [0.6, 0.4]}, index=["Fund A", "Fund B"]
    )
    sf2 = pd.DataFrame(
        {"Sharpe": [0.4, 1.2], "weight": [0.5, 0.5]}, index=["Fund A", "Fund B"]
    )
    frames = {dates[0]: sf1, dates[1]: sf2}

    class DummySelector:
        rank_column = "Sharpe"

        def select(
            self, score_frame: pd.DataFrame
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
            return score_frame, score_frame

    class DummyWeighting(BaseWeighting):
        def __init__(self) -> None:
            self.updates: list[tuple[pd.Series, int]] = []

        def weight(self, selected: pd.DataFrame) -> pd.DataFrame:
            weights = pd.Series([0.6, 0.4], index=selected.index, dtype=float)
            return pd.DataFrame({"weight": weights})

        def update(self, scores: pd.Series, days: int) -> None:
            self.updates.append((scores.copy(), days))

    class DummyRebalancer:
        def __init__(self) -> None:
            self.calls: list[pd.Series] = []

        def apply_triggers(
            self, prev_weights: pd.Series, score_frame: pd.DataFrame
        ) -> pd.Series:
            self.calls.append(prev_weights.copy())
            return prev_weights.sort_index()

    dummy_weighting = DummyWeighting()
    dummy_rebalancer = DummyRebalancer()

    weights_sequence = iter(
        [
            (pd.Series({"Fund A": 0.7, "Fund B": 0.3}, dtype=float), 0.123),
            (pd.Series({"Fund A": 0.5, "Fund B": 0.5}, dtype=float), 0.456),
        ]
    )

    def fake_apply(strategies, params, current_weights, target_weights, **kwargs):
        assert strategies == ["dummy"]
        assert params == {"dummy": {"alpha": 0.5}}
        assert list(current_weights.index) == ["Fund A", "Fund B"]
        assert list(target_weights.index) == ["Fund A", "Fund B"]
        assert kwargs["scores"].name == "Sharpe"
        return next(weights_sequence)

    monkeypatch.setattr(mp_engine, "apply_rebalancing_strategies", fake_apply)

    pf = run_schedule(
        frames,
        DummySelector(),
        dummy_weighting,
        rank_column="Sharpe",
        rebalancer=dummy_rebalancer,
        rebalance_strategies=["dummy"],
        rebalance_params={"dummy": {"alpha": 0.5}},
    )

    # Rebalancer used the initial weights and fake strategy output stored the cost
    assert dummy_rebalancer.calls, "rebalancer should have been invoked"
    assert pf.costs["2024-01-31"] == pytest.approx(0.123)
    assert pf.costs["2024-02-29"] == pytest.approx(0.456)
    assert pf.turnover["2024-02-29"] == pytest.approx(0.4)
    assert pf.total_rebalance_costs == pytest.approx(0.123 + 0.456)

    # Weighting update ran for both periods with increasing day offsets
    assert len(dummy_weighting.updates) == 2
    days_between = dummy_weighting.updates[1][1]
    assert days_between > 0


def test_run_requires_csv_path_when_df_missing():
    cfg_data = yaml.safe_load(Path("config/defaults.yml").read_text())
    cfg_data["multi_period"] = {
        "frequency": "M",
        "in_sample_len": 1,
        "out_sample_len": 1,
        "start": "1990-01",
        "end": "1990-03",
    }
    cfg_data["data"].pop("csv_path", None)
    cfg = Config(**cfg_data)
    with pytest.raises(KeyError):
        run_mp(cfg, df=None)


def test_run_raises_file_not_found_when_loader_returns_none(monkeypatch):
    cfg_data = yaml.safe_load(Path("config/defaults.yml").read_text())
    cfg_data["multi_period"] = {
        "frequency": "M",
        "in_sample_len": 1,
        "out_sample_len": 1,
        "start": "1990-01",
        "end": "1990-03",
    }
    cfg_data["data"]["csv_path"] = "missing.csv"
    cfg = Config(**cfg_data)

    monkeypatch.setattr(mp_engine, "load_csv", lambda path: None)

    with pytest.raises(FileNotFoundError):
        run_mp(cfg, df=None)
