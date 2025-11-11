"""Compatibility helpers for legacy validator entry points."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import numpy as np
import pandas as pd

from .market_data import (
    MarketDataMetadata,
    MarketDataMode,
    MarketDataValidationError,
    ValidatedMarketData,
    attach_metadata,
    classify_frequency,
    validate_market_data,
)


@dataclass(slots=True)
class _ValidationSummary:
    """Compute validation warnings reused across call sites."""

    metadata: MarketDataMetadata
    frame: pd.DataFrame

    def warnings(self) -> List[str]:
        warnings: list[str] = []
        rows = self.metadata.rows
        if rows < 12:
            warnings.append(
                f"Dataset is quite small ({rows} periods) – consider a longer history."
            )
        for column in self.frame.columns:
            valid = self.frame[column].notna().sum()
            if rows and valid / rows <= 0.5:
                warnings.append(
                    f"Column '{column}' has >50% missing values ({valid}/{rows} valid)."
                )
        if self.metadata.frequency_missing_periods > 0:
            warnings.append(
                "Date index contains "
                f"{self.metadata.frequency_missing_periods} missing {self.metadata.frequency_label} periods "
                f"(max gap {self.metadata.frequency_max_gap_periods})."
            )
        if self.metadata.missing_policy_dropped:
            dropped = ", ".join(sorted(self.metadata.missing_policy_dropped))
            warnings.append(
                "Missing-data policy dropped columns: "
                f"{dropped} (policy={self.metadata.missing_policy})."
            )
        if self.metadata.missing_policy_summary and (
            self.metadata.frequency_missing_periods > 0
            or bool(self.metadata.missing_policy_filled)
            or bool(self.metadata.missing_policy_dropped)
        ):
            warnings.append(
                "Missing-data policy applied: "
                f"{self.metadata.missing_policy_summary}."
            )
        return warnings


class ValidationResult:
    """Backwards-compatible structure returned by
    ``validate_returns_schema``."""

    def __init__(
        self,
        is_valid: bool,
        issues: Iterable[str] | None,
        warnings: Iterable[str] | None,
        frequency: str | None = None,
        date_range: Tuple[str, str] | None = None,
        metadata: MarketDataMetadata | None = None,
    ) -> None:
        self.is_valid = is_valid
        self.issues = list(issues or [])
        self.warnings = list(warnings or [])
        self.frequency = frequency
        self.date_range = date_range
        self.metadata = metadata
        self.mode: MarketDataMode | None = metadata.mode if metadata else None

    def get_report(self) -> str:
        lines = []
        if self.is_valid:
            lines.append("✅ Schema validation passed!")
            if self.frequency:
                lines.append(f"📊 Detected frequency: {self.frequency}")
            if self.date_range:
                lines.append(
                    f"📅 Date range: {self.date_range[0]} to {self.date_range[1]}"
                )
            if self.mode:
                lines.append(f"📈 Detected mode: {self.mode.value}")
            if (
                self.metadata
                and self.metadata.missing_policy_summary
                and (
                    self.metadata.frequency_missing_periods > 0
                    or bool(self.metadata.missing_policy_filled)
                    or bool(self.metadata.missing_policy_dropped)
                )
            ):
                lines.append(
                    "🧹 Missing data policy: " f"{self.metadata.missing_policy_summary}"
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

    def __iter__(self) -> Iterator[str]:  # pragma: no cover - compatibility shim
        return iter(self.issues)

    def __len__(self) -> int:  # pragma: no cover - compatibility shim
        return len(self.issues)

    def __eq__(self, other: object) -> bool:  # pragma: no cover - compatibility shim
        if isinstance(other, (list, tuple)):
            return list(self.issues) == list(other)
        return object.__eq__(self, other)


def detect_frequency(df: pd.DataFrame) -> str:
    """Best-effort frequency detection for legacy callers."""

    if not isinstance(df.index, pd.DatetimeIndex) or len(df.index) < 2:
        return "unknown"
    try:
        info = classify_frequency(df.index)
    except MarketDataValidationError as exc:
        user_msg = (getattr(exc, "user_message", "") or str(exc) or "").strip()
        issues = [str(issue).strip() for issue in getattr(exc, "issues", []) if issue]
        detail_parts = [part for part in [user_msg, "; ".join(issues)] if part]
        detail = "; ".join(part for part in detail_parts if part)
        if detail:
            return f"irregular ({detail})"
        return "irregular"
    label = str(info.get("label") or "unknown")
    code = str(info.get("code") or "")
    issues = info.get("issues") or []
    message = str(info.get("message") or "")
    if isinstance(issues, str):
        issues = [issues]
    irregular_hint = any(
        isinstance(item, str) and "irregular" in item.lower() for item in issues
    ) or ("irregular" in message.lower())
    if irregular_hint:
        details = [message.strip()] if message else []
        details.extend(
            item.strip() for item in issues if isinstance(item, str) and item.strip()
        )
        detail_text = "; ".join(d for d in details if d)
        if detail_text:
            return f"irregular ({detail_text})"
        return "irregular"
    if label == "unknown" and code not in {"", "UNKNOWN"}:
        # Provide the compact code if available (e.g. D/W/M)
        return code
    return label


def _build_result(validated: ValidatedMarketData) -> ValidationResult:
    summary = _ValidationSummary(validated.metadata, validated.frame)
    warnings = summary.warnings()
    return ValidationResult(
        True,
        [],
        warnings,
        frequency=validated.metadata.frequency_label,
        date_range=validated.metadata.date_range,
        metadata=validated.metadata,
    )


def validate_returns_schema(df: pd.DataFrame) -> ValidationResult:
    """Validate a DataFrame and return a legacy ``ValidationResult``."""

    try:
        validated = validate_market_data(df)
    except MarketDataValidationError as exc:
        issues = list(exc.issues) or [exc.user_message]
        return ValidationResult(False, issues, [])
    return _build_result(validated)


def _read_uploaded_file(file_like: Any) -> Tuple[pd.DataFrame, str]:
    name = getattr(file_like, "name", None)
    lower_name = name.lower() if isinstance(name, str) else ""

    if isinstance(file_like, (str, Path)):
        path = Path(file_like)
        if not path.exists():
            raise ValueError(f"File not found: '{path}'")
        if path.is_dir():
            raise ValueError(f"Path is a directory, not a file: '{path}'")
        try:
            if path.suffix.lower() in {".xlsx", ".xls"}:
                return pd.read_excel(path), str(path)
            if path.suffix.lower() in {".parquet", ".pq"}:
                return pd.read_parquet(path), str(path)
            return pd.read_csv(path), str(path)
        except Exception as exc:
            raise ValueError(f"Failed to read file: '{path}'") from exc

    try:
        if hasattr(file_like, "read"):
            if lower_name.endswith((".xlsx", ".xls")):
                data = file_like.read()
                buf = io.BytesIO(data)
                frame = pd.read_excel(buf)
            elif lower_name.endswith((".parquet", ".pq")):
                data = file_like.read()
                buf = io.BytesIO(data)
                frame = pd.read_parquet(buf)
            else:
                frame = pd.read_csv(file_like)
            try:
                file_like.seek(0)
            except Exception:  # pragma: no cover - not all streams support seek
                pass
            return frame, lower_name or "uploaded file"
    except FileNotFoundError:
        raise ValueError(f"File not found: '{lower_name or file_like}'")
    except PermissionError:
        raise ValueError(
            f"Permission denied accessing file: '{lower_name or file_like}'"
        )
    except IsADirectoryError:
        raise ValueError(
            f"Path is a directory, not a file: '{lower_name or file_like}'"
        )
    except pd.errors.EmptyDataError:
        raise ValueError(f"File contains no data: '{lower_name or file_like}'")
    except pd.errors.ParserError:
        raise ValueError(
            f"Failed to parse file (corrupted or invalid format): '{lower_name or file_like}'"
        )
    except Exception as exc:
        raise ValueError(f"Failed to read file: '{lower_name or file_like}'") from exc

    if lower_name:
        try:
            frame = pd.read_csv(file_like)
            return frame, lower_name
        except FileNotFoundError:
            raise ValueError(f"File not found: '{lower_name or file_like}'")
        except PermissionError:
            raise ValueError(
                f"Permission denied accessing file: '{lower_name or file_like}'"
            )
        except IsADirectoryError:
            raise ValueError(
                f"Path is a directory, not a file: '{lower_name or file_like}'"
            )
        except pd.errors.EmptyDataError:
            raise ValueError(f"File contains no data: '{lower_name or file_like}'")
        except pd.errors.ParserError:
            raise ValueError(
                f"Failed to parse file (corrupted or invalid format): '{lower_name or file_like}'"
            )
        except Exception as exc:
            raise ValueError(
                f"Failed to read file: '{lower_name or file_like}'"
            ) from exc

    raise ValueError("Unsupported upload source")


def load_and_validate_upload(file_like: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load uploaded content, validate it, and attach metadata."""

    frame, source = _read_uploaded_file(file_like)
    try:
        validated = validate_market_data(frame, source=source)
    except MarketDataValidationError as exc:
        raise MarketDataValidationError(exc.user_message, issues=exc.issues) from exc

    attach_metadata(validated.frame, validated.metadata)
    result = _build_result(validated)
    meta: Dict[str, Any] = {
        "metadata": validated.metadata,
        "validation": result,
        "n_rows": validated.metadata.rows,
        "original_columns": list(validated.metadata.columns),
        "mode": validated.metadata.mode.value,
        "frequency": validated.metadata.frequency_label,
        "date_range": validated.metadata.date_range,
    }
    return validated.frame, meta


def create_sample_template() -> pd.DataFrame:
    """Create a sample returns template with realistic data."""

    dates = pd.date_range(start="2023-01-31", periods=12, freq="ME")
    rng = np.random.default_rng(42)
    data: Dict[str, Any] = {"Date": dates}
    for idx in range(1, 6):
        data[f"Fund_{idx:02d}"] = rng.normal(0.008, 0.03, len(dates))
    data["SPX_Benchmark"] = rng.normal(0.007, 0.025, len(dates))
    return pd.DataFrame(data)
