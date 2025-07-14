import yaml
from pathlib import Path
from trend_analysis.multi_period.engine import run

CFG = yaml.safe_load(Path("config/defaults.yml").read_text())


def test_engine_generates_periods():
    res = run(CFG)
    periods = res["periods"]
    assert res["n_periods"] == len(periods)
    first = periods[0]
    assert first.in_start < first.out_start
