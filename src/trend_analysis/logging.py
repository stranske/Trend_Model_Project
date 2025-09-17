"""Structured JSONL logging utilities for analysis runs."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

__all__ = [
    "RunLogMetadata",
    "RunLogger",
    "RunLogHandler",
    "start_run_logger",
]


_DEFAULT_LOG_DIR = Path.cwd() / "run_logs"
_SAFE_TYPES = (str, int, float, bool, type(None))


def _coerce_json(value: Any) -> Any:
    """Convert arbitrary Python objects into JSON-serialisable structures."""

    if isinstance(value, _SAFE_TYPES):
        return value
    if isinstance(value, Mapping):
        return {str(key): _coerce_json(val) for key, val in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        return [_coerce_json(item) for item in value]
    return repr(value)


@dataclass(slots=True)
class RunLogMetadata:
    """Metadata describing a structured log file for a run."""

    run_id: str
    path: Path


class RunLogger:
    """Utility that writes structured JSONL log entries."""

    def __init__(self, run_id: str | None = None, log_dir: Path | None = None) -> None:
        self.run_id = run_id or uuid.uuid4().hex
        self._log_dir = Path(log_dir) if log_dir is not None else _DEFAULT_LOG_DIR
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._log_dir / f"{self.run_id}.jsonl"
        self._lock = threading.Lock()
        # Touch the file so it exists for tailers even before the first write.
        self._path.touch(exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        # Nothing to close when using per-write file handles.
        pass

    def log(
        self,
        step: str,
        message: str,
        *,
        level: str = "INFO",
        **extra: Any,
    ) -> None:
        """Write a structured log entry."""

        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "step": step,
            "level": level,
            "message": message,
        }
        if extra:
            payload["extra"] = _coerce_json(extra)
        entry = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as stream:
                stream.write(entry + "\n")
                stream.flush()


class RunLogHandler(logging.Handler):
    """Logging handler that forwards records into a :class:`RunLogger`."""

    _BASE_ATTRS = {
        "name",
        "msg",
        "message",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
    }

    def __init__(self, run_logger: RunLogger, default_step: str = "python") -> None:
        super().__init__()
        self._run_logger = run_logger
        self._default_step = default_step
        self.setLevel(logging.INFO)
        self._formatter = logging.Formatter()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive fallback
            message = str(record.msg)
        step = getattr(record, "step", None) or getattr(record, "run_step", None)
        level = record.levelname
        extra: dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key in self._BASE_ATTRS or key.startswith("_"):
                continue
            extra[key] = value
        if record.exc_info:
            extra["exception"] = self._formatter.formatException(record.exc_info)
        # Avoid passing duplicate 'step' keys via **extra
        extra.pop("step", None)
        extra.pop("run_step", None)
        self._run_logger.log(
            step or self._default_step,
            message,
            level=level,
            **extra,
        )

    def close(self) -> None:  # pragma: no cover - thin wrapper
        try:
            self.flush()
        finally:
            super().close()


@contextmanager
def start_run_logger(
    run_id: str | None = None,
    log_dir: Path | None = None,
) -> Iterator[RunLogger]:
    """Context manager that wires a :class:`RunLogger` into the logging system."""

    run_logger = RunLogger(run_id=run_id, log_dir=log_dir)
    handler = RunLogHandler(run_logger)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        yield run_logger
    finally:
        root_logger.removeHandler(handler)
        handler.close()
        run_logger.close()
