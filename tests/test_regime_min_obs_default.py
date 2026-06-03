from pathlib import Path

import yaml

from trend_analysis.regimes import RegimeSettings, normalise_settings


def test_regime_min_observations_default_matches_shipped_config() -> None:
    defaults_path = Path("config/defaults.yml")
    defaults = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))
    shipped_min_obs = defaults["regime"]["min_observations"]

    assert RegimeSettings().min_obs == shipped_min_obs
    assert normalise_settings(None).min_obs == shipped_min_obs
    assert normalise_settings({}).min_obs == shipped_min_obs
