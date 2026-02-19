from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as _pd

RUN_LOGGER_NAME = "trend_analysis.runlog"
_lock = threading.Lock()


@dataclass(slots=True)
class LogRecordSchema:
    ts: float
    run_id: str
    step: str
    level: str
    msg: str
    extra: dict[str, Any]


class JsonlHandler(logging.Handler):
    """A minimal JSONL file handler.

    Writes one JSON object per line.  Thread-safe via a coarse lock
    because logging volume here is modest (a few dozen lines per run).
    """

    def __init__(self, path: Path, max_size_kb: int | None = 1024):  # pragma: no cover - trivial
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_size_kb = max_size_kb

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - IO
        try:
            payload = {
                "ts": record.created,
                "run_id": getattr(record, "run_id", "unknown"),
                "step": getattr(record, "step", getattr(record, "funcName", "")),
                "level": record.levelname,
                "msg": record.getMessage(),
                "extra": {
                    k: v
                    for k, v in record.__dict__.items()
                    if k
                    not in {
                        "msg",
                        "args",
                        "levelname",
                        "levelno",
                        "msecs",
                        "relativeCreated",
                        "name",
                        "thread",
                        "threadName",
                        "processName",
                        "process",
                        "pathname",
                        "filename",
                        "module",
                        "exc_info",
                        "exc_text",
                        "stack_info",
                        "lineno",
                        "created",
                        "funcName",
                        "ts",
                        "run_id",
                        "step",
                    }
                },
            }
            line = json.dumps(payload, separators=(",", ":"))
            with _lock:
                # Rotate if exceeding size (simple single backup strategy)
                if self.max_size_kb and self.path.exists():
                    if self.path.stat().st_size > self.max_size_kb * 1024:
                        backup = self.path.with_suffix(self.path.suffix + ".1")  # e.g. run.jsonl.1
                        try:
                            if backup.exists():
                                backup.unlink()
                            self.path.rename(backup)
                        except Exception:  # pragma: no cover
                            pass
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception:  # pragma: no cover - defensive
            pass


def init_run_logger(run_id: str, log_path: Path) -> logging.Logger:
    """Initialise (or reset) the structured run logger.

    Safe to call multiple times; existing JSONL handlers are replaced.
    """

    logger = logging.getLogger(RUN_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    # Remove prior JsonlHandlers to avoid duplicate lines
    for h in list(logger.handlers):  # pragma: no cover - simple iteration
        if isinstance(h, JsonlHandler):
            logger.removeHandler(h)
    handler = JsonlHandler(log_path)
    logger.addHandler(handler)
    # Lightweight console echo for debugging (optional, not structured)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(logging.WARNING)
        logger.addHandler(sh)
    logger.propagate = False
    logger.info("logger_initialised", extra={"run_id": run_id, "step": "init"})
    return logger


def log_step(run_id: str, step: str, message: str, level: str = "INFO", **extra: Any) -> None:
    """Emit a structured log line for a pipeline step.

    Parameters
    ----------
    run_id : str
        Correlates all lines belonging to the same execution.
    step : str
        Machine-friendly step identifier (e.g. 'load_data', 'selection').
    message : str
        Human-readable message.
    level : str, default 'INFO'
        Logging level name.
    **extra : Any
        Additional JSON-serialisable context.
    """

    logger = logging.getLogger(RUN_LOGGER_NAME)
    if not logger.handlers:  # No structured handler registered; skip silently.
        return
    lvl = getattr(logging, level.upper(), logging.INFO)
    try:
        logger.log(lvl, message, extra={"run_id": run_id, "step": step, **extra})
    except Exception:  # pragma: no cover - defensive
        pass


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file (best-effort)."""

    if not Path(path).exists():  # pragma: no cover - guard
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:  # pragma: no cover
                continue


def latest_errors(path: Path, limit: int = 20) -> list[dict[str, Any]]:
    """Return up to `limit` ERROR-level lines (most recent first)."""

    lines = list(iter_jsonl(path))
    errs = [r for r in lines if r.get("level") == "ERROR"]
    return errs[-limit:]


def get_default_log_path(run_id: str, base: Optional[os.PathLike[str]] = None) -> Path:
    base_dir = Path(base) if base else Path("outputs") / "logs"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"run_{run_id}.jsonl"


__all__ = [
    "RUN_LOGGER_NAME",
    "init_run_logger",
    "log_step",
    "iter_jsonl",
    "latest_errors",
    "get_default_log_path",
    "logfile_to_frame",
    "error_summary",
]


def logfile_to_frame(path: Path, limit: int | None = 500) -> _pd.DataFrame:
    """Return a DataFrame of log lines (most recent first).

    Parameters
    ----------
    path : Path
        JSONL log path.
    limit : int | None
        If provided, restrict to last N lines.
    """

    if not path.exists():  # pragma: no cover - guard
        return _pd.DataFrame(columns=["ts", "run_id", "step", "level", "msg"])
    lines = list(iter_jsonl(path))
    if limit is not None:
        lines = lines[-limit:]
    if not lines:
        return _pd.DataFrame(columns=["ts", "run_id", "step", "level", "msg"])
    df = _pd.DataFrame(lines)

    # Flatten extra keys lightly for convenience (only scalar)
    def _scalar(v: Any) -> bool:
        return isinstance(v, (str, int, float, bool)) or v is None

    if "extra" in df.columns:
        extra_cols: dict[str, list[Any]] = {}
        for obj in df["extra"]:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if _scalar(v):
                        extra_cols.setdefault(k, []).append(v)
        for k, col_vals in extra_cols.items():
            if len(col_vals) == len(df):
                df[k] = col_vals
    return df.sort_values("ts", ascending=False).reset_index(drop=True)


def error_summary(path: Path) -> _pd.DataFrame:
    """Aggregate ERROR lines by message with counts and last timestamp."""
    if not path.exists():  # pragma: no cover
        return _pd.DataFrame(columns=["msg", "count", "last_ts"])
    df = logfile_to_frame(path, limit=None)
    if df.empty:
        return _pd.DataFrame(columns=["msg", "count", "last_ts"])
    errs = df[df["level"] == "ERROR"]
    if errs.empty:
        return _pd.DataFrame(columns=["msg", "count", "last_ts"])
    agg = (
        errs.groupby("msg")
        .agg(count=("msg", "size"), last_ts=("ts", "max"))
        .reset_index()
        .sort_values(["count", "last_ts"], ascending=[False, False])
    )
    return agg
