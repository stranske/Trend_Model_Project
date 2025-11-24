import numpy as np
import pandas as pd

from trend_analysis.config import Config
from trend_analysis.multi_period import run as run_mp


def _synthetic_df(periods=10, assets=5, seed=0):
    rng = np.random.default_rng(seed)
    # Use 'ME' (month end) instead of deprecated 'M'
    dates = pd.date_range("2020-01-31", periods=periods, freq="ME")
    data = rng.normal(scale=0.01, size=(periods, assets))
    cols = [f"F{i}" for i in range(assets)]
    df = pd.DataFrame(data, columns=cols)
    df.insert(0, "Date", dates)
    df.insert(1, "RF", 0.0)
    return df


def _base_cfg():
    return Config(
        version="0.1.0",
        data={"risk_free_column": "RF"},
        preprocessing={},
        vol_adjust={},
        sample_split={},
        portfolio={},
        benchmarks={},
        metrics={},
        export={},
        performance={},
        run={},
    )


def test_engine_incremental_cov_diag_consistency():
    df = _synthetic_df(30, 6)
    base = _base_cfg()
    # Minimal multi_period parameters similar to defaults
    base.multi_period = {
        "frequency": "M",
        "in_sample_len": 5,
        "out_sample_len": 1,
        "start": "2020-01",
        "end": "2020-12",
    }

    # Run with incremental disabled
    cfg_full = _base_cfg()
    cfg_full.multi_period = base.multi_period
    cfg_full.performance = {"enable_cache": True, "incremental_cov": False}
    res_full = run_mp(cfg_full, df=df)
    diag_full = [r.get("cov_diag") for r in res_full if "cov_diag" in r]

    # Run with incremental enabled
    cfg_inc = _base_cfg()
    cfg_inc.multi_period = base.multi_period
    cfg_inc.performance = {"enable_cache": True, "incremental_cov": True}
    res_inc = run_mp(cfg_inc, df=df)
    diag_inc = [r.get("cov_diag") for r in res_inc if "cov_diag" in r]

    assert len(diag_full) == len(diag_inc) > 0
    for a, b in zip(diag_full, diag_inc):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)


def test_engine_incremental_cov_missing_values_alignment():
    df = _synthetic_df(30, 5)
    # Introduce gaps that require forward-fill and zero backfill
    df.loc[5, "F1"] = np.nan
    df.loc[6, "F2"] = np.nan
    df.loc[9, "F3"] = np.nan
    df.loc[15:16, "F4"] = np.nan
    base = _base_cfg()
    base.multi_period = {
        "frequency": "M",
        "in_sample_len": 5,
        "out_sample_len": 1,
        "start": "2020-01",
        "end": "2020-12",
    }

    cfg_full = _base_cfg()
    cfg_full.multi_period = base.multi_period
    cfg_full.performance = {"enable_cache": True, "incremental_cov": False}
    res_full = run_mp(cfg_full, df=df)
    diag_full = [r.get("cov_diag") for r in res_full if "cov_diag" in r]

    cfg_inc = _base_cfg()
    cfg_inc.multi_period = base.multi_period
    cfg_inc.performance = {
        "enable_cache": True,
        "incremental_cov": True,
        "shift_detection_max_steps": 4,
    }
    res_inc = run_mp(cfg_inc, df=df)
    diag_inc = [r.get("cov_diag") for r in res_inc if "cov_diag" in r]

    assert len(diag_full) == len(diag_inc) > 0
    for a, b in zip(diag_full, diag_inc):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)
