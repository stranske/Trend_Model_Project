import numpy as np
import pandas as pd
import pytest

import trend_analysis.multi_period.engine as mp_engine
from trend.config_schema import CoreConfigError
from trend_analysis.regimes import normalise_settings


def _sample_frame() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    return pd.DataFrame({"Alpha": [0.01, 0.02, 0.03]}, index=dates)


def test_resolve_max_turnover_cap_accepts_numeric_scalars() -> None:
    df = _sample_frame()
    settings = normalise_settings(None)

    for value in (0.5, 1, np.float64(0.2)):
        resolved = mp_engine._resolve_max_turnover_cap(
            df,
            max_turnover_cfg=value,
            regime_settings=settings,
            benchmarks_cfg=None,
            regime_frequency="M",
            regime_ppy=12,
        )
        assert resolved == pytest.approx(float(value))


def test_resolve_max_turnover_cap_non_numeric_raises() -> None:
    class Dummy:
        def __repr__(self) -> str:
            return "<dummy>"

    df = _sample_frame()
    settings = normalise_settings(None)

    for value in ("abc", Dummy(), None):
        with pytest.raises(CoreConfigError) as excinfo:
            mp_engine._resolve_max_turnover_cap(
                df,
                max_turnover_cfg=value,
                regime_settings=settings,
                benchmarks_cfg=None,
                regime_frequency="M",
                regime_ppy=12,
            )
        message = str(excinfo.value)
        assert (
            "numeric scalars: int/float/numpy numeric types, or a valid regime mapping" in message
        )
        assert repr(value) in message


def test_resolve_max_turnover_cap_mapping_uses_regime_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _sample_frame()
    settings = normalise_settings(
        {
            "enabled": True,
            "proxy": "Alpha",
            "risk_on_label": "calm",
            "risk_off_label": "stress",
            "default_label": "calm",
        }
    )

    monkeypatch.setattr(mp_engine, "_resolve_regime_label_for_window", lambda *a, **k: "calm")

    resolved = mp_engine._resolve_max_turnover_cap(
        df,
        max_turnover_cfg={"calm": 0.3, "stress": 0.1},
        regime_settings=settings,
        benchmarks_cfg=None,
        regime_frequency="M",
        regime_ppy=12,
    )

    assert resolved == pytest.approx(0.3)
