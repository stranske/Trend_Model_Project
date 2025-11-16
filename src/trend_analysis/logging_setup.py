"""Central logging configuration for Trend Model scripts and CLIs."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = _REPO_ROOT / "perf" / "runs"
LOG_FORMAT = "%(asctime)sZ %(levelname)s [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
_HANDLER_FLAG = "_trend_perf_log_handler"


def _resolve_level(level: int | str) -> int:
    if isinstance(level, str):
        resolved = logging.getLevelName(level.upper())
        if isinstance(resolved, int):
            return resolved
        raise ValueError(f"Unknown log level: {level}")
    return int(level)


def _clear_existing_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_FLAG, False):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # pragma: no cover - defensive cleanup
                pass


def setup_logging(
    *,
    level: int | str = logging.INFO,
    console_level: Optional[int | str] = None,
    timestamp: Optional[str] = None,
    app_name: str = "app",
) -> Path:
    """Configure root logging and return the log file path.

    Parameters
    ----------
    level:
        Default log level for file + console handlers. Accepts ``int`` or
        ``logging``-compatible names such as ``"INFO"``.
    console_level:
        Optional override for the console handler level.
    timestamp:
        Optional run identifier. When omitted, the helper generates a UTC
        ``YYYYMMDD-HHMMSS`` value.
    app_name:
        Filename (without extension) to use for the log file.
    """

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{app_name}.log"

    logger = logging.getLogger()
    logger.setLevel(_resolve_level(level))
    _clear_existing_handlers(logger)

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(_resolve_level(level))
    setattr(file_handler, _HANDLER_FLAG, True)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_level = (
        _resolve_level(console_level)
        if console_level is not None
        else _resolve_level(level)
    )
    stream_handler.setLevel(stream_level)
    setattr(stream_handler, _HANDLER_FLAG, True)
    logger.addHandler(stream_handler)

    logger.info("Logging initialised", extra={"log_path": str(log_path)})
    return log_path


__all__ = ["setup_logging", "RUNS_ROOT"]
