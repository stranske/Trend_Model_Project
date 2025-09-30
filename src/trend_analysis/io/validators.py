"""Data validation helpers for uploaded market data."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from .market_data import MarketDataValidationError, validate_market_data

# Canonical mapping from human-readable frequency labels to pandas codes.
FREQUENCY_MAP: Dict[str, str] = {
    "daily": "D",
    "business-daily": "B",
    "weekly": "W",
    "monthly": "M",
    "quarterly": "Q",
    "annual": "Y",
    "irregular": "irregular",
}

REVERSE_FREQUENCY_MAP: Dict[str, str] = {
    code: label for label, code in FREQUENCY_MAP.items()
}


def detect_frequency(df: pd.DataFrame) -> str:
    """Best-effort frequency detection that mirrors legacy behaviour."""

    def _normalise_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
        if "Date" in frame.columns:
            idx = pd.to_datetime(frame["Date"], errors="coerce")
        elif isinstance(frame.index, pd.PeriodIndex):
            idx = frame.index.to_timestamp(how="end")
        elif isinstance(frame.index, pd.DatetimeIndex):
            idx = frame.index
        else:
            return pd.DatetimeIndex([])
        return pd.DatetimeIndex(idx.dropna())

    try:
        validated = validate_market_data(df)
    except MarketDataValidationError:
        idx = _normalise_index(df)
        if len(idx) < 2:
            return "unknown"

        freq_code = idx.freqstr
        if freq_code is None:
            try:
                freq_code = pd.infer_freq(idx)
            except ValueError:
                freq_code = None

        if freq_code is None:
            diffs = idx.to_series().diff().dropna()
            if diffs.empty:
                return "unknown"
            counts = diffs.value_counts(normalize=True)
            top_share = float(counts.iloc[0])
            top_delta = counts.index[0]
            if top_share < 0.8:
                preview = ", ".join(str(delta) for delta in counts.index[:3])
                return f"irregular ({preview})"
            freq_code = to_offset(top_delta).freqstr

        label = REVERSE_FREQUENCY_MAP.get(freq_code, "irregular")
        if label == "irregular" and freq_code:
            prefix = freq_code.split("-")[0]
            label = REVERSE_FREQUENCY_MAP.get(prefix, label)
            if label == "irregular" and prefix:
                label = REVERSE_FREQUENCY_MAP.get(prefix[:1], label)
        if label == "irregular" and freq_code not in {None, "irregular"}:
            return f"irregular ({freq_code})"
        return label

    metadata = validated.attrs.get("market_data", {})
    label = metadata.get("frequency")
    if isinstance(label, str):
        return label
    freq_code = metadata.get("frequency_code")
    if isinstance(freq_code, str):
        label = REVERSE_FREQUENCY_MAP.get(freq_code, "irregular")
        if label == "irregular" and freq_code:
            prefix = freq_code.split("-")[0]
            label = REVERSE_FREQUENCY_MAP.get(prefix, label)
            if label == "irregular" and prefix:
                label = REVERSE_FREQUENCY_MAP.get(prefix[:1], label)
        return label if label != "irregular" else freq_code
    return "unknown"


class ValidationResult:
    """Result of schema validation with detailed feedback."""

    def __init__(
        self,
        is_valid: bool,
        issues: List[str],
        warnings: List[str],
        frequency: Optional[str] = None,
        date_range: Optional[Tuple[str, str]] = None,
    ):
        self.is_valid = is_valid
        self.issues = issues
        self.warnings = warnings
        self.frequency = frequency
        self.date_range = date_range

    def get_report(self) -> str:
        """Generate a human-readable validation report."""
        lines: List[str] = []
        if self.is_valid:
            lines.append("✅ Schema validation passed!")
            if self.frequency:
                lines.append(f"📊 Detected frequency: {self.frequency}")
            if self.date_range:
                lines.append(
                    f"📅 Date range: {self.date_range[0]} to {self.date_range[1]}"
                )
        else:
            lines.append("❌ Schema validation failed!")

        if self.issues:
            lines.append("\n🔴 Issues that must be fixed:")
            for issue in self.issues:
                lines.append(f"  • {issue}")

        if self.warnings:
            lines.append("\n🟡 Warnings:")
            for warning in self.warnings:
                lines.append(f"  • {warning}")

        return "\n".join(lines)

    def __iter__(self) -> Iterator[str]:  # pragma: no cover - trivial
        return iter(self.issues)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.issues)

    def __eq__(self, other: object) -> bool:  # pragma: no cover - trivial
        if isinstance(other, (list, tuple)):
            return list(self.issues) == list(other)
        return object.__eq__(self, other)


def _serialise_timestamp(value: object) -> Optional[str]:
    if isinstance(value, pd.Timestamp):
        return value.tz_localize(None).isoformat()
    return None


def _read_from_path(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"File not found: '{path}'")
    if path.is_dir():
        raise ValueError(f"Path is a directory, not a file: '{path}'")

    suffix = path.suffix.lower()
    try:
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except PermissionError as exc:
        raise ValueError(f"Permission denied accessing file: '{path}'") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"File contains no data: '{path}'") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(
            f"Failed to parse file (corrupted or invalid format): '{path}'"
        ) from exc
    except FileNotFoundError:
        raise ValueError(f"File not found: '{path}'") from None
    except ImportError as exc:
        raise ValueError(f"Missing optional dependency while reading '{path}': {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to read file: '{path}'") from exc


def _read_from_buffer(file_like: Any) -> Tuple[pd.DataFrame, str]:
    name = getattr(file_like, "name", "") or "upload"
    suffix = Path(name).suffix.lower()

    try:
        if suffix in {".xlsx", ".xls"}:
            data = file_like.read()
            df = pd.read_excel(io.BytesIO(data))
        elif suffix in {".parquet", ".pq"}:
            data = file_like.read()
            df = pd.read_parquet(io.BytesIO(data))
        else:
            df = pd.read_csv(file_like)
    except PermissionError as exc:
        raise ValueError(f"Permission denied accessing file: '{name}'") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"File contains no data: '{name}'") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(
            f"Failed to parse file (corrupted or invalid format): '{name}'"
        ) from exc
    except IsADirectoryError as exc:
        raise ValueError(f"Path is a directory, not a file: '{name}'") from exc
    except ImportError as exc:
        raise ValueError(
            f"Missing optional dependency while reading '{name}': {exc}"
        ) from exc
    except Exception as exc:
        raise ValueError(f"Failed to read file: '{name}'") from exc
    finally:
        try:
            file_like.seek(0)
        except Exception:  # pragma: no cover - best effort
            pass

    return df, name


def validate_returns_schema(df: pd.DataFrame) -> ValidationResult:
    """Validate that a DataFrame conforms to the expected returns schema."""

    try:
        validated = validate_market_data(df)
    except MarketDataValidationError as exc:
        return ValidationResult(False, [str(exc)], [])

    metadata = validated.attrs.get("market_data", {})
    start = _serialise_timestamp(metadata.get("start"))
    end = _serialise_timestamp(metadata.get("end"))
    date_range = (start, end) if start and end else None
    frequency = metadata.get("frequency")

    return ValidationResult(True, [], [], frequency, date_range)


def load_and_validate_upload(file_like: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load an uploaded file (CSV, Excel, Parquet) and validate its schema."""

    if isinstance(file_like, (str, Path)):
        path = Path(file_like)
        df = _read_from_path(path)
        origin = f"upload:{path}"
        name = str(path)
    else:
        df, name = _read_from_buffer(file_like)
        origin = name

    try:
        validated = validate_market_data(df, origin=origin)
    except MarketDataValidationError as exc:
        raise ValueError(f"Schema validation failed:\n{exc}") from exc

    metadata = dict(validated.attrs.get("market_data", {}))
    frequency = metadata.get("frequency")
    freq_code = metadata.get("frequency_code")
    start = _serialise_timestamp(metadata.get("start"))
    end = _serialise_timestamp(metadata.get("end"))
    date_range = (start, end) if start and end else None

    if frequency and freq_code:
        FREQUENCY_MAP.setdefault(frequency, freq_code)

    validation = ValidationResult(True, [], [], frequency, date_range)

    meta: Dict[str, Any] = {
        "original_columns": list(validated.columns),
        "n_rows": len(validated),
        "validation": validation,
    }
    if frequency:
        meta["frequency"] = frequency
    if freq_code:
        meta["frequency_code"] = freq_code
    if date_range:
        meta["date_range"] = date_range
    mode = metadata.get("mode")
    if mode:
        meta["mode"] = mode

    return validated, meta


def create_sample_template() -> pd.DataFrame:
    """Create a sample returns template DataFrame."""

    dates = pd.date_range(start="2023-01-31", end="2023-12-31", freq="ME")
    np.random.seed(42)
    n_funds = 5
    sample_data: Dict[str, Any] = {"Date": dates}

    for i in range(1, n_funds + 1):
        returns = np.random.normal(0.008, 0.03, len(dates))
        sample_data[f"Fund_{i:02d}"] = returns

    benchmark_returns = np.random.normal(0.007, 0.025, len(dates))
    sample_data["SPX_Benchmark"] = benchmark_returns

    return pd.DataFrame(sample_data)


__all__ = [
    "FREQUENCY_MAP",
    "detect_frequency",
    "ValidationResult",
    "validate_returns_schema",
    "load_and_validate_upload",
    "create_sample_template",
]
