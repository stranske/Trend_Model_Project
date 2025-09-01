import pandas as pd
import yaml  # type: ignore[import-untyped]
from pathlib import Path
from trend_analysis.config import Config
from trend_analysis.multi_period import run as run_mp, run_schedule, Portfolio
from trend_analysis.selector import RankSelector
from trend_analysis.weighting import EqualWeight, AdaptiveBayesWeighting


def make_df():
    dates = pd.date_range("1990-01-31", periods=12, freq="M")
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
    dates = pd.date_range("1990-01-31", periods=6, freq="M")
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
    """Test run function with invalid price_frames raises appropriate errors."""
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
