"""Regression coverage for the shared portfolio turnover ceiling."""

import json

import pytest

from trend_analysis.config.model import PortfolioSettings
from trend_analysis.config.models import Config
from trend_analysis.config.turnover import MAX_TURNOVER_CEILING
from trend_analysis.util.paths import proj_path


@pytest.mark.parametrize("value", [0.5, 1.0, 1.5, 2.0, 2.5])
def test_max_turnover_ceiling_is_single_sourced(value: float) -> None:
    schema = json.loads((proj_path() / "config.schema.json").read_text())
    maximum = schema["properties"]["portfolio"]["properties"]["max_turnover"]["maximum"]
    settings = {
        "rebalance_calendar": "NYSE",
        "max_turnover": value,
        "cost_model": {"per_trade_bps": 5, "half_spread_bps": 0},
    }
    accepted = value <= maximum
    for validator in (
        lambda: Config(
            version="test",
            data={},
            preprocessing={},
            vol_adjust={},
            sample_split={},
            portfolio=settings,
            metrics={},
            regime={},
            export={},
            performance={},
            run={},
        ),
        lambda: PortfolioSettings.model_validate(settings),
    ):
        if accepted:
            validator()
        else:
            with pytest.raises(ValueError):
                validator()


def test_schema_ceiling_matches_the_constant() -> None:
    schema = json.loads((proj_path() / "config.schema.json").read_text())
    max_turnover = schema["properties"]["portfolio"]["properties"]["max_turnover"]
    assert max_turnover["maximum"] == MAX_TURNOVER_CEILING
    assert max_turnover["constraints"]["maximum"] == MAX_TURNOVER_CEILING
    assert max_turnover["additionalProperties"]["maximum"] == MAX_TURNOVER_CEILING
