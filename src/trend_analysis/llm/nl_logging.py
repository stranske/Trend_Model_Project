"""Structured logging models for NL operations."""

from __future__ import annotations

import json
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from trend_analysis.config.patch import ConfigPatch
from trend_analysis.config.validation import ValidationResult

_NL_LOG_PREFIX = "nl_ops_"
_NL_LOG_DIR = Path(".trend_nl_logs")
_NL_LOG_SUFFIX = ".jsonl"
_NL_LOG_LOCK = threading.Lock()


class NLOperationLog(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str
    timestamp: datetime
    operation: Literal["nl_to_patch", "apply_patch", "validate", "run"]
    input_hash: str
    prompt_template: str
    prompt_variables: dict[str, Any]
    model_output: str | None
    parsed_patch: ConfigPatch | None
    validation_result: ValidationResult | None
    error: str | None
    duration_ms: float
    model_name: str
    temperature: float
    token_usage: dict[str, Any] | None


def _log_date_from_timestamp(timestamp: datetime) -> date:
    if timestamp.tzinfo is None:
        return timestamp.date()
    return timestamp.astimezone(timezone.utc).date()


def get_nl_log_path(
    *,
    log_date: date | None = None,
    base_dir: Path | None = None,
) -> Path:
    target_dir = Path(base_dir) if base_dir is not None else _NL_LOG_DIR
    active_date = log_date or datetime.now(timezone.utc).date()
    return target_dir / f"{_NL_LOG_PREFIX}{active_date.isoformat()}{_NL_LOG_SUFFIX}"


def rotate_nl_logs(
    *,
    base_dir: Path | None = None,
    keep_days: int = 30,
    today: date | None = None,
) -> list[Path]:
    target_dir = Path(base_dir) if base_dir is not None else _NL_LOG_DIR
    if not target_dir.exists():
        return []
    active_date = today or datetime.now(timezone.utc).date()
    cutoff = active_date - timedelta(days=keep_days)
    removed: list[Path] = []
    for path in target_dir.glob(f"{_NL_LOG_PREFIX}*{_NL_LOG_SUFFIX}"):
        stem = path.stem
        if not stem.startswith(_NL_LOG_PREFIX):
            continue
        date_str = stem[len(_NL_LOG_PREFIX) :]
        try:
            log_date = date.fromisoformat(date_str)
        except ValueError:
            continue
        if log_date <= cutoff:
            try:
                path.unlink()
                removed.append(path)
            except FileNotFoundError:
                continue
    return removed


def write_nl_log(
    entry: NLOperationLog,
    *,
    base_dir: Path | None = None,
    rotate: bool = True,
    keep_days: int = 30,
    now: datetime | None = None,
) -> Path:
    target_dir = Path(base_dir) if base_dir is not None else _NL_LOG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    if rotate:
        rotate_nl_logs(
            base_dir=target_dir,
            keep_days=keep_days,
            today=(now or datetime.now(timezone.utc)).date(),
        )
    log_date = _log_date_from_timestamp(entry.timestamp)
    log_path = get_nl_log_path(log_date=log_date, base_dir=target_dir)
    payload = entry.model_dump(mode="json")
    line = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), default=str)
    with _NL_LOG_LOCK:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return log_path


__all__ = ["NLOperationLog", "get_nl_log_path", "rotate_nl_logs", "write_nl_log"]
