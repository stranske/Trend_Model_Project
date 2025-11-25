from pathlib import Path

import pandas as pd

from trend_analysis.api import run_simulation
from trend_analysis.config import Config
from trend_analysis.logging import (
    get_default_log_path,
    init_run_logger,
    logfile_to_frame,
)


def _make_df():
    dates = pd.date_range("2022-01-31", periods=14, freq="ME")
    return pd.DataFrame(
        {
            "Date": dates,
            "Fund_A": 0.01,
            "Fund_B": 0.012,
            "Fund_C": 0.009,
            "SPX": 0.011,
            "RF": 0.001,
        }
    )


def _make_cfg():
    return Config(
        version="1",
        data={"risk_free_column": "RF"},
        preprocessing={},
        vol_adjust={"target_vol": 1.0},
        sample_split={
            "in_start": "2022-01",
            "in_end": "2022-07",
            "out_start": "2022-08",
            "out_end": "2022-10",
        },
        portfolio={"selection_mode": "all", "weighting_scheme": "equal"},
        benchmarks={"spx": "SPX"},
        metrics={},
        export={},
        run={},
    )


def test_granular_steps_logged(tmp_path: Path):
    df = _make_df()
    cfg = _make_cfg()
    run_id = "intlog123"
    log_path = get_default_log_path(run_id, base=tmp_path)
    init_run_logger(run_id, log_path)
    # Attach run_id dynamically (config model forbids unknown fields)
    try:
        object.__setattr__(cfg, "run_id", run_id)  # type: ignore[arg-type]
    except Exception:
        pass
    run_simulation(cfg, df)
    assert log_path.exists()
    frame = logfile_to_frame(log_path)
    # Assert that selection, weighting & benchmarks steps appear (weighting always logged now)
    steps = set(frame["step"].tolist())
    assert {"selection", "benchmarks", "weighting"}.issubset(steps)
