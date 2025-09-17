import json
import logging
from pathlib import Path

from trend_analysis.logging import RunLogger, start_run_logger


def _read_entries(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_run_logger_writes_jsonl(tmp_path: Path) -> None:
    run_logger = RunLogger(run_id="test", log_dir=tmp_path)
    try:
        run_logger.log("setup", "starting", level="INFO", foo=1)
        run_logger.log("step", "warning", level="WARNING", data={"bar": 2})
    finally:
        run_logger.close()

    entries = _read_entries(tmp_path / "test.jsonl")
    assert entries[0]["run_id"] == "test"
    assert entries[0]["step"] == "setup"
    assert entries[0]["level"] == "INFO"
    assert entries[1]["level"] == "WARNING"
    assert entries[1]["extra"]["data"] == {"bar": 2}


def test_run_log_handler_captures_logging(tmp_path: Path) -> None:
    with start_run_logger(run_id="handler", log_dir=tmp_path) as run_logger:
        logger = logging.getLogger("trend_analysis.test")
        logger.warning("Something happened", extra={"step": "phase"})
        run_logger.log("custom", "direct call")

    entries = _read_entries(tmp_path / "handler.jsonl")
    assert any(entry["step"] == "phase" for entry in entries)
    assert any(entry["message"] == "direct call" for entry in entries)


def test_start_run_logger_generates_run_id(tmp_path: Path) -> None:
    with start_run_logger(log_dir=tmp_path) as run_logger:
        run_logger.log("step", "message")
        file_path = run_logger.path

    assert file_path.exists()
    entries = _read_entries(file_path)
    assert entries[0]["run_id"] == run_logger.run_id
