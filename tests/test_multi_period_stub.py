import warnings
from pathlib import Path

import yaml

from trend_analysis.multi_period.scheduler import generate_periods

CFG = yaml.safe_load(Path("config/defaults.yml").read_text())


def test_scheduler_generates_periods():
    periods = generate_periods(CFG)
    assert periods, "Scheduler returned empty list"
    first = periods[0]
    assert first.in_start < first.out_start, "Period tuple ordering incorrect"


def test_frequency_alias_resolves_without_warning():
    cfg = yaml.safe_load(Path("config/defaults.yml").read_text())
    cfg["multi_period"]["frequency"] = "M"
    with warnings.catch_warnings(record=True) as w:
        _ = generate_periods(cfg)
    assert not w, "Unexpected warnings when using frequency 'M'"
