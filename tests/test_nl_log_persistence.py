from __future__ import annotations

from datetime import date, datetime, timezone

from trend_analysis.config.patch import ConfigPatch, PatchOperation
from trend_analysis.config.validation import ValidationResult
from trend_analysis.llm.nl_logging import (
    NLOperationLog,
    rotate_nl_logs,
    write_nl_log,
)


def _sample_log(timestamp: datetime) -> NLOperationLog:
    patch = ConfigPatch(
        operations=[PatchOperation(op="set", path="analysis.top_n", value=12)],
        summary="Update top_n selection",
    )
    validation = ValidationResult(valid=True, errors=[], warnings=[])
    return NLOperationLog(
        request_id="req-abc",
        timestamp=timestamp,
        operation="nl_to_patch",
        input_hash="hash-123",
        prompt_template="template",
        prompt_variables={"instruction": "Set top_n to 12"},
        model_output='{"operations": [], "summary": "noop"}',
        parsed_patch=patch,
        validation_result=validation,
        error=None,
        duration_ms=12.5,
        model_name="unit-test-model",
        temperature=0.0,
        token_usage={"prompt_tokens": 5, "completion_tokens": 7},
    )


def test_write_nl_log_creates_daily_file(tmp_path) -> None:
    timestamp = datetime(2025, 1, 5, 12, 0, tzinfo=timezone.utc)
    entry = _sample_log(timestamp)

    log_path = write_nl_log(entry, base_dir=tmp_path, rotate=False)

    assert log_path == tmp_path / "nl_ops_2025-01-05.jsonl"
    assert log_path.exists()
    contents = log_path.read_text(encoding="utf-8")
    assert '"request_id":"req-abc"' in contents


def test_rotate_nl_logs_removes_older_files(tmp_path) -> None:
    for day in range(1, 32):
        log_path = tmp_path / f"nl_ops_2025-01-{day:02d}.jsonl"
        log_path.write_text("{}", encoding="utf-8")

    removed = rotate_nl_logs(base_dir=tmp_path, keep_days=30, today=date(2025, 1, 31))

    assert (tmp_path / "nl_ops_2025-01-01.jsonl") in removed
    assert not (tmp_path / "nl_ops_2025-01-01.jsonl").exists()
    assert (tmp_path / "nl_ops_2025-01-02.jsonl").exists()


def test_write_nl_log_rotates_by_default(tmp_path) -> None:
    old_path = tmp_path / "nl_ops_2025-01-01.jsonl"
    old_path.write_text("{}", encoding="utf-8")
    timestamp = datetime(2025, 2, 1, 8, 0, tzinfo=timezone.utc)
    entry = _sample_log(timestamp)

    write_nl_log(entry, base_dir=tmp_path, now=timestamp)

    assert not old_path.exists()
