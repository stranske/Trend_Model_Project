"""Tests for the structured logging helpers.

These tests exercise the JSONL writer, read helpers and aggregation utilities
implemented in :mod:`trend_analysis.logging`.  The module under test powers
the pipeline run logging experience, so we focus on verifying that:

* a run logger can be initialised in isolation and writes JSON Lines output;
* ``log_step`` is a safe no-op when no handlers are registered;
* helper utilities can read, flatten and aggregate the emitted records.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from trend_analysis.logging import (
    RUN_LOGGER_NAME,
    error_summary,
    get_default_log_path,
    init_run_logger,
    iter_jsonl,
    latest_errors,
    log_step,
    logfile_to_frame,
)


@pytest.fixture
def isolated_run_logger() -> Iterator[None]:
    """Ensure each test interacts with a clean run logger instance."""

    logger = logging.getLogger(RUN_LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.handlers.clear()
    logger.propagate = False
    try:
        yield
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        logger.handlers.clear()


def test_log_step_without_handlers_is_noop(isolated_run_logger: None) -> None:
    """Calling ``log_step`` without initialising the logger should not
    crash."""

    logger = logging.getLogger(RUN_LOGGER_NAME)
    assert not logger.handlers

    # Should simply return; exercising the defensive branch where no handlers
    # are configured yet.
    log_step("run-123", "bootstrap", "starting")

    assert not logger.handlers


def test_jsonl_logging_roundtrip(isolated_run_logger: None, tmp_path: Path) -> None:
    """Full integration test for the JSONL logging helpers."""

    log_path = tmp_path / "logs" / "run.jsonl"

    # Initialising the logger should create the JSONL file and emit the
    # ``logger_initialised`` line immediately.
    init_run_logger("run-456", log_path)
    assert log_path.exists()

    # Emit two structured log lines with consistent ``user`` metadata so the
    # DataFrame helper can flatten it into a dedicated column.
    log_step("run-456", "load_data", "fetched", user="alice", rows=120)
    log_step(
        "run-456",
        "load_data",
        "failed",
        level="error",
        user="alice",
        reason="timeout",
    )

    records = list(iter_jsonl(log_path))
    assert [r["msg"] for r in records][-1] == "failed"
    assert records[-1]["level"] == "ERROR"

    # ``latest_errors`` should surface the error entry we just emitted.
    latest = latest_errors(log_path)
    assert len(latest) == 1
    assert latest[0]["msg"] == "failed"

    # ``logfile_to_frame`` returns most recent entries first and flattens the
    # shared ``user`` metadata into its own column.
    frame = logfile_to_frame(log_path, limit=2)
    assert len(frame) == 2
    assert list(frame["msg"]) == ["failed", "fetched"]
    assert list(frame["user"]) == ["alice", "alice"]

    # ``error_summary`` aggregates ERROR rows across the entire log file.
    summary = error_summary(log_path)
    assert summary["msg"].tolist() == ["failed"]
    assert summary["count"].tolist() == [1]
    assert summary["last_ts"].iloc[0] == pytest.approx(records[-1]["ts"])


def test_get_default_log_path_creates_directory(tmp_path: Path) -> None:
    """The helper should materialise the base directory when required."""

    target = get_default_log_path("abc123", base=tmp_path)
    assert target.name == "run_abc123.jsonl"
    assert target.parent == tmp_path
    assert target.parent.exists()


def test_logfile_to_frame_empty(tmp_path: Path) -> None:
    """An empty log file should return an empty, schema-compatible frame."""

    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("\n")

    frame = logfile_to_frame(empty_path, limit=5)
    expected_cols = ["ts", "run_id", "step", "level", "msg"]
    assert list(frame.columns) == expected_cols
    assert frame.empty


def test_logfile_to_frame_handles_missing_and_nested_extra(tmp_path: Path) -> None:
    """The reader should gracefully handle logs without ``extra`` metadata."""

    path = tmp_path / "manual.jsonl"
    entries = [
        {
            "ts": 1.0,
            "run_id": "r1",
            "step": "init",
            "level": "INFO",
            "msg": "with-extra",
            "extra": {"user": "alice", "meta": {"foo": 1}},
        },
        {"ts": 2.0, "run_id": "r1", "step": "init", "level": "INFO", "msg": "no-extra"},
    ]
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

    # Limiting to the first row exercises the branch where no ``extra`` column
    # exists in the DataFrame.
    no_extra = logfile_to_frame(path, limit=1)
    assert "extra" not in no_extra.columns

    # Using the full file introduces an ``extra`` column containing both dict
    # and non-dict entries; nested structures should not be flattened.
    frame = logfile_to_frame(path, limit=None)
    assert "extra" in frame.columns
    assert "meta" not in frame.columns


def test_error_summary_empty_conditions(tmp_path: Path) -> None:
    """Edge cases: missing file, empty log and logs without ERROR lines."""

    missing = tmp_path / "missing.jsonl"
    assert error_summary(missing).empty

    empty_log = tmp_path / "only_newline.jsonl"
    empty_log.write_text("\n", encoding="utf-8")
    assert error_summary(empty_log).empty

    info_only = tmp_path / "info_only.jsonl"
    info_only.write_text(
        json.dumps(
            {
                "ts": 3.0,
                "run_id": "r2",
                "step": "load",
                "level": "INFO",
                "msg": "ok",
                "extra": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert error_summary(info_only).empty
