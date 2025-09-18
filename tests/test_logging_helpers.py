from pathlib import Path

from trend_analysis.logging import (
    init_run_logger,
    log_step,
    logfile_to_frame,
    error_summary,
)


def test_logfile_helpers(tmp_path: Path):
    run_id = "rhelper"
    log_path = tmp_path / "run.jsonl"
    init_run_logger(run_id, log_path)
    log_step(run_id, "load", "Loaded dataset", rows=10)
    log_step(run_id, "selection", "Selected managers", count=3)
    log_step(run_id, "selection", "ERROR occurred", level="ERROR", detail="boom")
    df = logfile_to_frame(log_path)
    assert not df.empty
    assert set(["ts", "run_id", "step", "level", "msg"]).issubset(df.columns)
    # Ensure ordering newest first
    assert df.iloc[0]["ts"] >= df.iloc[-1]["ts"]
    errs = error_summary(log_path)
    if not errs.empty:
        # Only one distinct error message expected
        assert errs.iloc[0]["count"] >= 1
        assert errs.iloc[0]["msg"] in df[df.level == "ERROR"]["msg"].unique()
