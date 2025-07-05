import pandas as pd
import yaml
from pathlib import Path
from trend_analysis.config import Config
from trend_analysis.multi_period import run as run_mp, run_schedule, Portfolio
from trend_analysis.selector import RankSelector
from trend_analysis.weighting import EqualWeight, AdaptiveBayesWeighting


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
