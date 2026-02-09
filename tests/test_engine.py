import pandas as pd
import pytest

import trend_analysis.multi_period.engine as mp_engine
from trend.config_schema import CoreConfigError
from trend_analysis.regimes import normalise_settings


def _sample_frame() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    return pd.DataFrame({"Alpha": [0.01, 0.02, 0.03]}, index=dates)


def test_resolve_max_turnover_cap_none_message() -> None:
    df = _sample_frame()
    settings = normalise_settings(None)

    with pytest.raises(CoreConfigError) as excinfo:
        mp_engine._resolve_max_turnover_cap(
            df,
            max_turnover_cfg=None,
            regime_settings=settings,
            benchmarks_cfg=None,
            regime_frequency="M",
            regime_ppy=12,
        )
    message = str(excinfo.value)
    assert "None" in message
    assert (
        "numeric scalars: int/float/numpy numeric types, or a valid regime mapping"
        in message
    )


def test_resolve_max_turnover_cap_string_message() -> None:
    df = _sample_frame()
    settings = normalise_settings(None)

    with pytest.raises(CoreConfigError) as excinfo:
        mp_engine._resolve_max_turnover_cap(
            df,
            max_turnover_cfg="abc",
            regime_settings=settings,
            benchmarks_cfg=None,
            regime_frequency="M",
            regime_ppy=12,
        )
    message = str(excinfo.value)
    assert "'abc'" in message
    assert (
        "numeric scalars: int/float/numpy numeric types, or a valid regime mapping"
        in message
    )
