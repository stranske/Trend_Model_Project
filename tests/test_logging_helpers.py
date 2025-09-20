import logging
from pathlib import Path

from trend_analysis.logging import (RUN_LOGGER_NAME, error_summary,
                                    get_default_log_path, init_run_logger,
                                    latest_errors, log_step, logfile_to_frame)


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


def test_logfile_to_frame_flattens_scalar_extras(tmp_path: Path) -> None:
    run_id = "flatten"
    log_path = tmp_path / "flatten.jsonl"
    init_run_logger(run_id, log_path)

    # Emit multiple records sharing the same scalar extra so the helper promotes
    # it to a dedicated column. Include another extra present on only one line
    # to confirm it is ignored when lengths do not match the total row count.
    for i in range(3):
        log_step(run_id, "step", f"message-{i}", foo=i)
    log_step(run_id, "step", "final", foo=99, bar="solo")

    # Drop the initial logger_initialised entry so all rows share the ``foo`` extra.
    with log_path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
    with log_path.open("w", encoding="utf-8") as fh:
        fh.writelines(lines[1:])

    df = logfile_to_frame(log_path)

    # Column should be promoted because every record supplied ``foo``.
    assert "foo" in df.columns
    # Latest record first so values appear in reverse chronological order.
    assert df.loc[0, "foo"] == 99
    assert set(df["foo"].tolist()) == {0, 1, 2, 99}

    # ``bar`` appears once so the helper should leave it nested inside ``extra``.
    assert "bar" not in df.columns
    extras = [extra for extra in df["extra"].tolist() if isinstance(extra, dict)]
    assert any("bar" in extra for extra in extras)


def test_latest_errors_and_default_path(tmp_path: Path) -> None:
    run_id = "err-log"
    log_path = get_default_log_path(run_id, base=tmp_path)
    logger = init_run_logger(run_id, log_path)

    log_step(run_id, "prep", "starting")
    logger.warning("warn", extra={"run_id": run_id, "step": "prep", "code": 1})
    logger.error("boom", extra={"run_id": run_id, "step": "run", "code": 2})

    errors = latest_errors(log_path, limit=5)
    assert errors
    assert errors[-1]["msg"] == "boom"
    assert errors[-1]["extra"]["code"] == 2

    summary = error_summary(log_path)
    assert not summary.empty
    assert summary.iloc[0]["msg"] == "boom"


def test_log_step_no_handlers_is_noop() -> None:
    logger = logging.getLogger(RUN_LOGGER_NAME)
    handlers = list(logger.handlers)
    try:
        for handler in handlers:
            logger.removeHandler(handler)
        logger.propagate = False

        # Should silently do nothing because no structured handler is registered.
        log_step("no-run", "step", "message")

        # No handlers should have been added implicitly.
        assert not logger.handlers
    finally:
        for handler in handlers:
            logger.addHandler(handler)
