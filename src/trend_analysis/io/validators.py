"""Internal validation adapters retained for isolated regression coverage.

New IO call sites use :mod:`trend_analysis.io.market_data` or
:mod:`trend_analysis.io.ui_ingest`; this module is intentionally not exported
from the public ``trend_analysis.io`` surface.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .market_data import (
    MarketDataMetadata,
    MarketDataValidationError,
    ValidatedMarketData,
    attach_metadata,
    validate_market_data,
)


@dataclass(slots=True)
class _ValidationSummary:
    """Compute validation warnings reused across call sites."""

    metadata: MarketDataMetadata
    frame: pd.DataFrame

    def warnings(self) -> list[str]:
        warnings: list[str] = []
        rows = self.metadata.rows
        if rows < 12:
            warnings.append(f"Dataset is quite small ({rows} periods) – consider a longer history.")
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
            warnings.append(f"Missing-data policy applied: {self.metadata.missing_policy_summary}.")
        return warnings


def _validation_report(validated: ValidatedMarketData) -> dict[str, Any]:
    summary = _ValidationSummary(validated.metadata, validated.frame)
    return {
        "is_valid": True,
        "issues": [],
        "warnings": summary.warnings(),
        "frequency": validated.metadata.frequency_label,
        "date_range": validated.metadata.date_range,
        "mode": validated.metadata.mode.value,
        "metadata": validated.metadata,
    }


def _read_uploaded_file(file_like: Any) -> tuple[pd.DataFrame, str]:
    """Read every upload shape through one format and error boundary."""

    name = getattr(file_like, "name", None)
    lower_name = name.lower() if isinstance(name, str) else ""

    if isinstance(file_like, (str, Path)):
        path = Path(file_like)
        if not path.exists():
            raise ValueError(f"File not found: '{path}'")
        if path.is_dir():
            raise ValueError(f"Path is a directory, not a file: '{path}'")
        reader_input: Any = path
        source = str(path)
        suffix = path.suffix.lower()
    else:
        if not hasattr(file_like, "read") and not lower_name:
            raise ValueError("Unsupported upload source")
        reader_input = file_like
        source = lower_name or "uploaded file"
        suffix = Path(lower_name).suffix.lower()

    try:
        if suffix in {".xlsx", ".xls"}:
            if isinstance(reader_input, Path):
                frame = pd.read_excel(reader_input)
            else:
                frame = pd.read_excel(io.BytesIO(reader_input.read()))
        elif suffix in {".parquet", ".pq"}:
            if isinstance(reader_input, Path):
                frame = pd.read_parquet(reader_input)
            else:
                frame = pd.read_parquet(io.BytesIO(reader_input.read()))
        else:
            frame = pd.read_csv(reader_input)
    except FileNotFoundError as exc:
        raise ValueError(f"File not found: '{source}'") from exc
    except PermissionError as exc:
        raise ValueError(f"Permission denied accessing file: '{source}'") from exc
    except IsADirectoryError as exc:
        raise ValueError(f"Path is a directory, not a file: '{source}'") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"File contains no data: '{source}'") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Failed to parse file (corrupted or invalid format): '{source}'") from exc
    except Exception as exc:
        raise ValueError(f"Failed to read file: '{source}'") from exc
    finally:
        if not isinstance(reader_input, Path) and hasattr(reader_input, "seek"):
            try:
                reader_input.seek(0)
            except Exception:  # pragma: no cover - not all streams support seek
                pass

    return frame, source


def load_and_validate_upload(file_like: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load uploaded content, validate it, and attach metadata."""

    frame, source = _read_uploaded_file(file_like)
    try:
        validated = validate_market_data(frame, source=source)
    except MarketDataValidationError as exc:
        raise MarketDataValidationError(exc.user_message, issues=exc.issues) from exc

    attach_metadata(validated.frame, validated.metadata)
    meta: dict[str, Any] = {
        "metadata": validated.metadata.model_dump(mode="json"),
        "validation": _validation_report(validated),
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
    data: dict[str, Any] = {"Date": dates}
    for idx in range(1, 6):
        data[f"Fund_{idx:02d}"] = rng.normal(0.008, 0.03, len(dates))
    data["SPX_Benchmark"] = rng.normal(0.007, 0.025, len(dates))
    return pd.DataFrame(data)
